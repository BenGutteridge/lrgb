import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
from .utils import init_khop_GCN_lite

class RelationalDelayGNNLiteStage(nn.Module):
    """
    Stage that stack GNN layers and includes a 1-hop skip (Delay GNN for max K = 2)

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self = init_khop_GCN_lite(self, dim_in, dim_out, num_layers, skip_first_hop=True) # skip L=0 since using custom A_{k=1}
        print('Using \nu_{kt} for k up to %d' % num_layers)
        print('Edge types: ', cfg.edge_types, '\nAdding edge types to model...')
        W_edge, nu_edge = {}
        for e in cfg.edge_types:
            for t in range(num_layers):
                W_edge['t=%d, e=%s'%(t,e)] = GNNLayer(dim_in, dim_out)
                nu_edge['k=1, t=%d, e=%s'%(t,e)] = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.W_edge = nn.ModuleDict(W_edge)
        self.nu_edge = nn.ParameterDict(nu_edge)

    def forward(self, batch):
        """
        x_{t+1} = x_t + f(x_t, x_{t-1})
        first pass: uses regular edge index for each layer
        """
        # adjacency matrices (inc for edge types for k=1)
        A = lambda k : batch.edge_index[:, batch.edge_attr[:,0]==k] # edge attr now includes both k-hop and edge type
        A_edge = lambda e : batch.edge_index[:, batch.edge_attr[:,1]==int(e)] # using -1 to distinguish k>1 hop edges
        # accessing weight matrices and nu scalars
        W = lambda t : self.W_t["t=%d"%t]
        nu = lambda k, t : self.nu_kt["k=%d, t=%d"%(k,t)]
        W_edge = lambda e, t : self.W_edge["t=%d, e=%s"%(t,e)]
        nu_edge = lambda e, t : self.nu_edge["k=1, t=%d, e=%s"%(t,e)]

        # run through layers
        t, x = 0, [] # length t list with x_0, x_1, ..., x_t
        for t in range(self.num_layers):
            x.append(batch.x)
            # k = 1 : use custom A_{k=1} for each edge type
            batch.x = torch.zeros_like(x[t])
            for e in cfg.edge_types:
                batch.x = batch.x + nu_edge(e, t) * W_edge(e,t)(batch, x[t], A_edge(e)).x
            # k > 1 
            for k in range(2, (t+1)+1):
                if A(k).shape[1] > 0: # prevents adding I*W*H (bc of self added connections to zero adj)
                    delay = max(k-self.rbar,0)
                    if cfg.rbar_v2:
                        delay = int((k-1)//self.rbar)
                    batch.x = batch.x + nu(k,t) * W(t)(batch, x[t-delay], A(k)).x
            batch.x = x[t] + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm: # normalises after every layer
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch

register_stage('rel_delay_gnn_lite', RelationalDelayGNNLiteStage)