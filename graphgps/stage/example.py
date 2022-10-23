import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage

# from torch_geometric.graphgym.models.layer import GeneralLayer # using custom

from torch_geometric.graphgym.models.layer import LayerConfig, new_layer_config
def GNNLayer(dim_in, dim_out, has_act=True):
    """
    Wrapper for a GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    return GeneralLayer(
        cfg.gnn.layer_type,
        layer_config=new_layer_config(dim_in, dim_out, 1, has_act=has_act,
                                      has_bias=False, cfg=cfg))

# # OLD
# def GNNLayer(dim_in, dim_out, has_act=True):
#     return GeneralLayer(cfg.gnn.layer_type, dim_in, dim_out, has_act)


class GNNStackStage(nn.Module):
    '''Simple Stage that stack GNN layers'''
    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch


register_stage('example', GNNStackStage)


# ####################
# # Delay skip

# import torch

# # @register_stage('delay_gnn')      # xt+1 = f(x)       (NON-RESIDUAL)
# class DelayGNNStage(nn.Module):
#     """
#     Stage that stack GNN layers and includes a 1-hop skip (Delay GNN for max K = 2)

#     Args:
#         dim_in (int): Input dimension
#         dim_out (int): Output dimension
#         num_layers (int): Number of GNN layers
#     """
#     def __init__(self, dim_in, dim_out, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         self.max_k = cfg.gnn.layers_mp # cfg.delay.max_k
#         for t in range(num_layers):
#             d_in = dim_in if t == 0 else dim_out
#             K = min(self.max_k, t+1)
#             for k in range(1, K+1):
#                 W = GNNLayer(d_in, dim_out) # regular GCN layers
#                 self.add_module('W_k{}_t{}'.format(k,t), W)

#     def forward(self, batch):
#         """
#         x_{t+1} = x_t + f(x_t, x_{t-1})
#         first pass: uses regular edge index for each layer
#         """
#         # # old k-hop method: inefficient
#         # from graphgym.ben_utils import get_k_hop_adjacencies
#         # k_hop_edges, _ = get_k_hop_adjacencies(batch.edge_index, self.max_k)
#         # A = lambda k : k_hop_edges[k-1]

#         # new k-hop method: efficient
#         # k-hop adj matrix
#         A = lambda k : batch.edge_index[:, batch.edge_attr==k]
        
#         # run through layers
#         t, x = 0, [] # length t list with x_0, x_1, ..., x_t
#         modules = self.children()
#         for t in range(self.num_layers):
#             x.append(batch.x)
#             batch.x = torch.zeros_like(x[t])
#             for k in range(1, (t+1)+1):
#                 W = next(modules)
#                 batch.x = batch.x + W(batch, x[t+1-k], A(k)).x
#             batch.x = x[t] + nn.ReLU()(batch.x)
#             if cfg.gnn.l2norm: # normalises after every layer
#                 batch.x = F.normalize(batch.x, p=2, dim=-1)
#         return batch

# register_stage('delay_gnn', DelayGNNStage)

# ###########

# # @register_stage('kgnn')      # to compare with DelayGCN: all x is x(t), no 'delay', same params
# class KGNNStage(nn.Module):
#     """
#     NO DELAY ELEMENT

#     Args:
#         dim_in (int): Input dimension
#         dim_out (int): Output dimension
#         num_layers (int): Number of GNN layers
#     """
#     def __init__(self, dim_in, dim_out, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         self.max_k = cfg.gnn.layers_mp # cfg.delay.max_k
#         for t in range(num_layers):
#             d_in = dim_in if t == 0 else dim_out
#             K = min(self.max_k, t+1)
#             for k in range(1, K+1):
#                 W = GNNLayer(d_in, dim_out) # regular GCN layers
#                 self.add_module('W_k{}_t{}'.format(k,t), W)

#     def forward(self, batch):
#         """
#         x_{t+1} = x_t + f(x_t)
#         first pass: uses regular edge index for each layer
#         """
#         # # old k-hop method: inefficient
#         # from graphgym.ben_utils import get_k_hop_adjacencies
#         # k_hop_edges, _ = get_k_hop_adjacencies(batch.edge_index, self.max_k)
#         # A = lambda k : k_hop_edges[k-1]

#         # new k-hop method: efficient
#         # k-hop adj matrix
#         A = lambda k : batch.edge_index[:, batch.edge_attr==k]
        
#         # run through layers
#         t = 0
#         modules = self.children()
#         for t in range(self.num_layers):
#             x = batch.x
#             batch.x = torch.zeros_like(x)
#             for k in range(1, (t+1)+1):
#                 W = next(modules)
#                 batch.x = batch.x + W(batch, x, A(k)).x
#             batch.x = x + nn.ReLU()(batch.x)
#             if cfg.gnn.l2norm: # normalises after every layer
#                 batch.x = F.normalize(batch.x, p=2, dim=-1)
#         return batch

# register_stage('kgnn', KGNNStage)

###########

from torch_geometric.graphgym import register

# my General Layer
class GeneralLayer(nn.Module):
    """
    General wrapper for layers edited to be suitable for DelayGCN

    Args:
        name (string): Name of the layer in registered :obj:`layer_dict`
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation after the layer
        has_bn (bool):  Whether has BatchNorm in the layer
        has_l2norm (bool): Wheter has L2 normalization after the layer
        **kwargs (optional): Additional args
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.has_l2norm = layer_config.has_l2norm
        has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not has_bn
        self.layer = register.layer_dict[name](layer_config, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(layer_config.dim_out, eps=layer_config.bn_eps,
                               momentum=layer_config.bn_mom))
        if layer_config.dropout > 0:
            layer_wrapper.append(
                nn.Dropout(p=layer_config.dropout,
                           inplace=layer_config.mem_inplace))
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch, x, edge):
        batch = self.layer(batch, x, edge)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch

