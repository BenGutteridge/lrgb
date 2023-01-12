import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
# from .utils import init_khop_GCN
from .utils import init_khop_GCN_v2


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
        # self = init_khop_GCN(self, dim_in, dim_out, num_layers)
        self = init_khop_GCN_v2(self, dim_in, dim_out, num_layers)

    def forward(self, batch, dirichlet_energy=False):
        """
        x_{t+1} = x_t + f(x_t, x_{t-1})
        first pass: uses regular edge index for each layer
        """
        # # old k-hop method: inefficient
        # from graphgym.ben_utils import get_k_hop_adjacencies
        # k_hop_edges, _ = get_k_hop_adjacencies(batch.edge_index, self.max_k)
        # A = lambda k : k_hop_edges[k-1]

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
            for k in range(1, (t+1)+1):
                # W = next(modules)
                if A(k).shape[1] > 0: # iff there are edges of type k
                    delay = max(k-self.rbar,0)
                    if cfg.rbar_v2:
                        delay = int((k-1)//self.rbar)
                    batch.x = batch.x + W(k,t)(batch, x[t-delay], A(k)).x
            batch.x = x[t] + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm: # normalises after every layer
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        if dirichlet_energy:
            L, energies = get_laplacian(A(1)), []
            for x_i in x+[batch.x]:
                energies.append(dirichlet(x_i, L))
            batch.dirichlet_energies = np.array(energies)
        return batch

register_stage('delay_gnn', DelayGNNStage)

import numpy as np

def dirichlet(x, L):
    """takes in list of node features (nxd) 
    at each layer and outputs array of dirichlet energies"""
    x = tonp(x)
    assert x.shape[0] == L.shape[0] == L.shape[1]
    print('x: ', type(x))
    print('L: ', type(L))
    E = np.dot(np.dot(x.T, L), x)
    E = np.trace(E) / np.linalg.norm(x, ord='fro')
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