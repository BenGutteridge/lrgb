# import torch.nn as nn
# import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.register import register_stage
# import torch
from .example import GNNLayer

def empty_func():
  pass

def init_khop_GCN(model, dim_in, dim_out, num_layers):
  model.num_layers = num_layers
  model.max_k = cfg.gnn.layers_mp # cfg.delay.max_k
  for t in range(num_layers):
      d_in = dim_in if t == 0 else dim_out
      K = min(model.max_k, t+1)
      for k in range(1, K+1):
          W = GNNLayer(d_in, dim_out) # regular GCN layers
          model.add_module('W_k{}_t{}'.format(k,t), W)
  return model
