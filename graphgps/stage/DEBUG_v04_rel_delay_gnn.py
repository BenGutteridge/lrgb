import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
from .utils import init_khop_GCN, init_khop_GCN_v2, add_edge_types_to_model

class RelationalDelayGNNStage_v4(nn.Module):
    """
    Stage that stack GNN layers and includes a 1-hop skip (Delay GNN for max K = 2)

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self = init_khop_GCN_v2(self, dim_in, dim_out, num_layers, skip_first_hop=True) # skip L=0 since using custom A_{k=1}
        print('Edge types: ', cfg.edge_types, '\nAdding edge types to model...')
        # self = add_edge_types_to_model(self, cfg.edge_types, dim_in, dim_out)
        #####
        print("N.B. NOT CURRENTLY USING EDGE TYPES FOR DEBUGGING")
        self.W_kt['k=1, t=0'] = GNNLayer(dim_in, dim_out)
        #####

    def forward(self, batch):
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
        A = lambda k : batch.edge_index[:, batch.edge_attr[:,0]==k] # edge attr now includes both k-hop and edge type
        # A_edge = lambda e : batch.edge_index[:, batch.edge_attr[:,1]==int(e)] # using -1 to distinguish k>1 hop edges
        W = lambda k, t : self.W_kt["k=%d, t=%d"%(k,t)]

        # run through layers
        t, x = 0, [] # length t list with x_0, x_1, ..., x_t
        for t in range(self.num_layers):
            x.append(batch.x)
            # k = 1
            batch.x = torch.zeros_like(x[t])
            # for e in self.edge_types: # a list of strings
            #     batch.x = batch.x + self.W_edge[e](batch, x[t], A_edge(e)).x
            # k > 1 
            for k in range(1, (t+1)+1):
                if A(k).shape[1] > 0: # prevents adding I*W*H (bc of self added connections to zero adj)
                    delay = max(k-self.rbar,0)
                    if cfg.rbar_v2:
                        delay = int((k-1)//self.rbar)
                    batch.x = batch.x + W(k,t)(batch, x[t-delay], A(k)).x    # n.b. W edits batch.x in-place. Not a problem here though
            batch.x = x[t] + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm: # normalises after every layer
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch

register_stage('rel_delay_gnn', RelationalDelayGNNStage_v4)