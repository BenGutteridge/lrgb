import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_scatter import scatter

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer

from param_calcs import get_k_neighbourhoods


class DRewGatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, t, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.share_weights = bool('share' in cfg.gnn.layer_type)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        # self.C = pyg_nn.Linear(in_dim, out_dim, bias=True) # leave for now
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        if self.share_weights:
            self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
            self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)
        else:
            k_neighbourhoods = get_k_neighbourhoods(t)
            self.B = nn.ModuleDict({str(k): pyg_nn.Linear(in_dim, out_dim, bias=True) for k in k_neighbourhoods})
            self.E = nn.ModuleDict({str(k): pyg_nn.Linear(in_dim, out_dim, bias=True) for k in k_neighbourhoods})


        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.e = None

        self.nu = cfg.nu if cfg.nu != -1 else float('inf')

    def forward(self, t, xs, batch): # needs to take current layer and custom x list
        x, edge_index = batch.x, batch.edge_index
        # e = batch.edge_attr
        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]     # not currently used
        edge_index      : [2, n_edges]
        """

        if self.residual:
            x_in = x
            # e_in = e

        k_neighbourhoods = get_k_neighbourhoods(t)
        delay = lambda k : max(k-self.nu, 0)
        if self.share_weights:
            B = lambda _ : self.B
        else:
            B = lambda k : self.B[k]

        x_states = [] 
        for k in k_neighbourhoods:
            x_states.append(t-delay(k))
        x_states = set(sorted(x_states)) # remove dupes
        Bx, Ex = {}, {} # ones which use the 'target' node and may require delay
        
        # for l in x_states: # makes dict of necessary Bx^{t-\tau}
        #     Bx[l] = self.B(xs[l])
        #     Ex[l] = self.E(xs[l])

        for k in k_neighbourhoods:
            Bx[k] = B(k)(xs[t-delay(k)])
            Ex[k] = B(k)(xs[t-delay(k)])
        
        Ax = self.A(x)
        # Ce = self.C(e)x
        Dx = self.D(x) # these use the local node i and do not require varying k-neighbourhoods

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        # make Dx_i, Ex_j 
        i_idxs, j_idxs = edge_index[1,:], edge_index[0,:] # 0 is j, 1 is i by pytorch's convention
        node_dim = 0 # default is -2 which is equivalent
        Dx_i = {k : Dx.index_select(node_dim, i_idxs[batch.edge_attr==k]) for k in k_neighbourhoods} # local node, always current timestep

        # # Ex, Bx only need indexing by different amounts of delay, _j correspond to the k-neighbourhood adjacency and so only need indexing by k (t-delay(k) always the same for each k)
        # Ex_j = {k : Ex[t-delay(k)].index_select(node_dim, j_idxs[batch.edge_attr==k]) for k in k_neighbourhoods}
        # Bx_j = {k : Bx[t-delay(k)].index_select(node_dim, j_idxs[batch.edge_attr==k]) for k in k_neighbourhoods}

        Ex_j = {k : Ex[k].index_select(node_dim, j_idxs[batch.edge_attr==k]) for k in k_neighbourhoods}
        Bx_j = {k : Bx[k].index_select(node_dim, j_idxs[batch.edge_attr==k]) for k in k_neighbourhoods}
        
        if pe_LapPE: # TODO
            PE_i = pe_LapPE.index_select(node_dim, i_idxs)
            PE_j = pe_LapPE.index_select(node_dim, i_idxs)

        # MESSAGES
        sigma_ij = {}
        for k in k_neighbourhoods:
            e_ij_k = Dx_i[k] + Ex_j[k] # + Ce
            sigma_ij[k] = torch.sigmoid(e_ij_k)

        # # Handling for Equivariant and Stable PE using LapPE
        # # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        # self.e = e_ij
        
        # AGGREGATE
        dim_size = Bx[list(Bx.keys())[0]].shape[0]  # or None ??   <--- Double check this [BG: their note not mine]
        alpha = torch.ones(len(k_neighbourhoods)) # TODO implement alpha params
        alpha = F.softmax(alpha, dim=0)
        if not cfg.agg_weights.convex_combo: alpha = alpha * alpha.shape[0]
        x = 0
        for k_i, k in enumerate(k_neighbourhoods):
            sum_sigma_x = sigma_ij[k] * Bx_j[k]
            numerator_eta_xj = scatter(sum_sigma_x, i_idxs[batch.edge_attr==k], 
                                    0, None, dim_size,
                                    reduce='sum')
            sum_sigma = sigma_ij[k]
            denominator_eta_xj = scatter(sum_sigma, i_idxs[batch.edge_attr==k],
                                        0, None, dim_size,
                                        reduce='sum')
            eta_xj = (numerator_eta_xj / (denominator_eta_xj + 1e-6))
            x = x + alpha[k_i] * eta_xj

        # UPDATE
        x = Ax + x
        x = self.bn_node_x(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # e = e_ij
        # e = self.bn_edge_e(e)
        # e = F.relu(e)
        # e = F.dropout(e, self.dropout, training=self.training)
        if self.residual:
            x = x_in + x
            # e = e_in + e
        batch.x = x
        # batch.edge_attr = e

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


class DRewGatedGCNGraphGymLayer(nn.Module):
    """GatedGCN layer.
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = DRewGatedGCNLayer(in_dim=layer_config.dim_in,
                                   out_dim=layer_config.dim_out,
                                   dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                   residual=False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                   **kwargs)

    def forward(self, batch):
        return self.model(batch)


register_layer('drewgatedgcnconv', DRewGatedGCNGraphGymLayer)
register_layer('share_drewgatedgcnconv', DRewGatedGCNGraphGymLayer)
