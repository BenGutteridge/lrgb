import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
# from .utils import init_khop_GCN
from .utils import init_khop_GCN_v2
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
        # self = init_khop_GCN(self, dim_in, dim_out, num_layers)
        self = init_khop_GCN_v2(self, dim_in, dim_out, num_layers)

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
            alpha = self.alpha_t[t] if cfg.use_agg_weights else torch.ones(len(k_neighbourhoods)) # learned weighting or equal weighting
            alpha = F.softmax(alpha)
            for i, k in enumerate(k_neighbourhoods):
                if A(k).shape[1] > 0: # iff there are edges of type k
                    delay = max(k-self.nu,0)
                    if cfg.nu_v2:
                        delay = int((k-1)//self.nu)
                    batch.x = batch.x + alpha[i] * W(k,t)(batch, x[t-delay], A(k)).x
            batch.x = x[t] + nn.ReLU()(batch.x)
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

register_stage('delay_gnn', DelayGNNStage)

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