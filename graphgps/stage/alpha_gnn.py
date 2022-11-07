import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
from .utils import init_khop_nondynamic_GCN

# @register_stage('delay_gnn')      # xt+1 = f(x)       (NON-RESIDUAL)
class AlphaGNNStage(nn.Module):
    """
    \alpha_GNN: 
    The nondynamic version of r*GCN - no dynamically added components

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        # W_t only, not W_{k,t}
        # alpha_k sums to 1 and weights Sk
        # all Sk used at every layer - nondynamic
        alpha = num_layers if cfg.alpha==1 else cfg.alpha
        print('Running alphaGNN, alpha = ', alpha)
        self = init_khop_nondynamic_GCN(self, 
                                        dim_in, dim_out, 
                                        num_layers, 
                                        max_k=alpha)
        for t in range(num_layers):
            assert len(self.W[t]) == alpha

    def forward(self, batch):
        """
        Eq (2) in paper writeup (so far)
        Sum of all S_k.W_{k,t} for k < alpha
        """
        # k-hop adj matrix
        A = lambda k : batch.edge_index[:, batch.edge_attr==k]
        W = lambda t,k : self.W[t][k-1] # W(t,k)
        alpha = self.max_k
        # run through layers
        t = 0
        for t in range(self.num_layers):
            x = batch.x
            batch.x = torch.zeros_like(x)
            for k in range(1, alpha+1):
                batch.x = batch.x + W(t,k)(batch, x, A(k)).x
            batch.x = x + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm: # normalises after every layer
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch

register_stage('alpha_gnn', AlphaGNNStage)