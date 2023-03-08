import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym import register
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
# from .utils import init_khop_GCN
from .utils import init_DRewGCN
sort_and_removes_dupes = lambda mylist : sorted(list(dict.fromkeys(mylist)))
custom_heads = ['jk_maxpool_graph']
from param_calcs import get_k_neighbourhoods

# @register_stage('delay_gnn')      # xt+1 = f(x)       (NON-RESIDUAL)
class DelayGNNStage(nn.Module):
    """
    Stage that stack GNN layers and includes a 1-hop skip (Delay GNN for max K = 2)

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        has_act = True
        self.use_bn, self.dropout = False, 0
        assert cfg.gnn.dropout == 0, 'Error: not supported'
        assert cfg.alt_postlayer.bn or cfg.alt_postlayer.act, 'Error: only intended for use with alt_postlayer: bn=%d, act=%d' % (int(cfg.alt_postlayer.bn), int(cfg.alt_postlayer.act))
        if cfg.alt_postlayer.bn: cfg.gnn.batchnorm = False 
        if cfg.alt_postlayer.act: has_act = False # so standard setup is not used

        self.num_layers, use_weights = cfg.gnn.layers_mp, cfg.agg_weights.use
        self.nu = cfg.nu if cfg.nu != -1 else float('inf')
        W_kt = {}
        if use_weights: alpha_t = []
        t0 = 0
        post_layer = []
        for t in range(t0, num_layers):
            d_in = dim_in if t == 0 else dim_out
            k_neighbourhoods = get_k_neighbourhoods(t)
            for k in k_neighbourhoods:
                W_kt["k=%d, t=%d" % (k,t)] = GNNLayer(d_in, dim_out, has_act) # regular GCN layers
                # if use_weights: alpha_t.append(torch.nn.Parameter(torch.randn(len(k_neighbourhoods)), requires_grad=True)) # random init from normal dist
                if use_weights: alpha_t.append(torch.nn.Parameter(torch.ones(len(k_neighbourhoods)), requires_grad=True)) # unity init
            layer_wrapper = [] # adding post-summation batch norm and/or activation
            if cfg.alt_postlayer.bn:
                layer_wrapper.append(
                    nn.BatchNorm1d(dim_out,
                               eps=cfg.bn.eps,
                               momentum=cfg.bn.mom))
            if cfg.alt_postlayer.act:
                layer_wrapper.append(register.act_dict[cfg.gnn.act])
            post_layer.append(nn.Sequential(*layer_wrapper))
            
        self.W_kt = nn.ModuleDict(W_kt)
        self.post_layer = nn.ModuleList(post_layer)
        if use_weights: self.alpha_t = nn.ParameterList(alpha_t)

    def forward(self, batch, dirichlet_energy=False):
        """
        x_{t+1} = x_t + f(x_t, x_{t-1})
        first pass: uses regular edge index for each layer
        """

        # new k-hop method: efficient
        # k-hop adj matrix
        A = lambda k : batch.edge_index[:, batch.edge_attr==k]
        W = lambda k, t : self.W_kt["k=%d, t=%d"%(k,t)]

        # run through layers
        t, x = 0, [] # length t list with x_0, x_1, ..., x_t
        # modules = self.children()
        for t in range(self.num_layers):
            x.append(batch.x)
            batch.x = torch.zeros_like(x[t])
            k_neighbourhoods = get_k_neighbourhoods(t)
            alpha = self.alpha_t[t] if cfg.agg_weights.use else torch.ones(len(k_neighbourhoods)) # learned weighting or equal weighting
            alpha = F.softmax(alpha, dim=0)
            alpha = alpha if cfg.agg_weights.convex_combo else alpha * len(k_neighbourhoods) # convex comb, or scale by no. of terms (e.g. unity weights for agg_weights.use=False)
            for i, k in enumerate(k_neighbourhoods):
                if A(k).shape[1] > 0: # iff there are edges of type k
                    delay = max(k-self.nu,0)
                    if cfg.nu_v2:
                        delay = int((k-1)//self.nu)
                    batch.x = batch.x + alpha[i] * W(k,t)(batch, x[t-delay], A(k)).x
            batch.x = x[t] + self.post_layer[t](batch.x)
            # batch.x = x[t] + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm: # normalises after every layer
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        if dirichlet_energy:
            L, energies = get_laplacian(A(1)), []
            for x_i in x+[batch.x]:
                energies.append(dirichlet(x_i, L))
            batch.dirichlet_energies = np.array(energies)
        if 'jk' in cfg.gnn.head:
            return batch, x # for heads using Jumping Knowledge at final layer
        return batch

register_stage('delay_gnn_alt_postlayer', DelayGNNStage)

import numpy as np

def dirichlet(x, L):
    """takes in list of node features (nxd) 
    at each layer and outputs array of dirichlet energies"""
    x = tonp(x)
    assert x.shape[0] == L.shape[0] == L.shape[1]
    # print('x: ', type(x))
    # print('L: ', type(L))
    E = np.dot(np.dot(x.T, L), x)
    E = np.trace(E) / np.linalg.norm(x, ord='fro')**2
    return E

def get_laplacian(edge_index):
    L = pyg.utils.get_laplacian(edge_index, normalization='sym')[0]
    L = pyg.utils.to_dense_adj(L).squeeze() # from index format to matrix
    return tonp(L)

def tonp(tsr):
    if isinstance(tsr, np.ndarray):
        return tsr
    elif isinstance(tsr, np.matrix):
        return np.array(tsr)
    # elif isinstance(tsr, scipy.sparse.csc.csc_matrix):
    #     return np.array(tsr.todense())

    assert isinstance(tsr, torch.Tensor)
    tsr = tsr.cpu()
    assert isinstance(tsr, torch.Tensor)

    try:
        arr = tsr.numpy()
    except TypeError:
        arr = tsr.detach().to_dense().numpy()
    except:
        arr = tsr.detach().numpy()

    assert isinstance(arr, np.ndarray)
    return arr