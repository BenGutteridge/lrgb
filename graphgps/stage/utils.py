import torch.nn as nn
# import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
sort_and_removes_dupes = lambda mylist : sorted(list(dict.fromkeys(mylist)))
from param_calcs import get_k_neighbourhoods


def init_DRewGCN(model, dim_in, dim_out, num_layers, skip_first_hop=False):
  """The (nu)DRew-GCN param initialiser, used for delay_gnn"""
  model.num_layers, use_weights = num_layers, cfg.use_agg_weights
  model.nu = cfg.nu if cfg.nu != -1 else float('inf')
  W_kt = {}
  if use_weights: alpha_t = []
  t0 = 1 if skip_first_hop else 0
  for t in range(t0, num_layers):
    d_in = dim_in if t == 0 else dim_out
    k_neighbourhoods = get_k_neighbourhoods(t)
    for k in k_neighbourhoods:
      W_kt["k=%d, t=%d" % (k,t)] = GNNLayer(d_in, dim_out) # regular GCN layers
    if use_weights: alpha_t.append(torch.nn.Parameter(torch.randn(len(k_neighbourhoods)), requires_grad=True)) # TODO: is to.device needed after randn()?
  model.W_kt = nn.ModuleDict(W_kt)
  if use_weights: model.alpha_t = nn.ParameterList(alpha_t)
  return model

def init_shareDRewGCN(model, dim_in, dim_out, num_layers, skip_first_hop=False):
  """The (nu)DRew-GCN param initialiser, but with weight sharing"""
  model.num_layers, use_weights = num_layers, cfg.use_agg_weights
  model.nu = cfg.nu if cfg.nu != -1 else float('inf')
  W_t = {}
  if use_weights: alpha_t = []
  t0 = 1 if skip_first_hop else 0
  for t in range(t0, num_layers):
    d_in = dim_in if t == 0 else dim_out
    k_neighbourhoods = get_k_neighbourhoods(t)
    W_t["t=%d" % (t)] = GNNLayer(d_in, dim_out) # regular GCN layers
    if use_weights: alpha_t.append(torch.nn.Parameter(torch.randn(len(k_neighbourhoods)), requires_grad=True)) # TODO: is to.device needed after randn()?
  model.W_t = nn.ModuleDict(W_t)
  if use_weights: model.alpha_t = nn.ParameterList(alpha_t)
  return model

########## Retired ##########

def init_khop_nondynamic_GCN(model, dim_in, dim_out, num_layers, max_k=None):
  """For the non-dynamic k-hop model: alpha_k_gnn.
  W_t, alpha_k (sums to 1)"""
  assert num_layers == cfg.gnn.layers_mp
  model.num_layers = num_layers
  model.max_k = cfg.gnn.layers_mp if max_k is None else max_k
  if cfg.max_graph_diameter <= model.max_k:
    print("Warning: max_graph_diameter = %d; <= max_k, so setting max_k to max_graph_diameter" % cfg.max_graph_diameter)
    model.max_k = cfg.max_graph_diameter
  # set hidden_dim if using fixed param count
  if cfg.fixed_params:
    assert dim_in == dim_out # if this isn't the case I've goofed somewhere
    n_params = cfg.fixed_mp_params_num
    num_W = model.max_k*((model.max_k+1)/2 + (num_layers - model.max_k))
    dim_out = (n_params/num_W)**0.5
    print('Using fixed mp param count of %d: hidden_dim = %d' % (n_params, dim_in))
  # make the W_k
  W = []
  for t in range(num_layers):
    W.append(nn.ModuleList([GNNLayer(dim_in, dim_out) for _ in range(model.max_k)])) # W_{k,t}
  model.W = nn.ModuleList(W)
  return model

def init_khop_GCN_lite(model, dim_in, dim_out, num_layers, max_k=None, skip_first_hop=False):
  """Using weight sharing over k - i.e. \\nu_{k,t}W_t rather than W_{k,t}"""
  assert num_layers == cfg.gnn.layers_mp
  model.num_layers = num_layers
  model.max_k = cfg.gnn.layers_mp if max_k is None else max_k
  # set hidden_dim if using fixed param count
  if cfg.fixed_params:
    assert dim_in == dim_out # if this isn't the case I've goofed somewhere
    n_params = cfg.fixed_mp_params_num
    dim_out = (n_params/num_layers)**0.5
    print('Using fixed mp param count of %d: hidden_dim = %d' % (n_params, dim_out))
  W_t = {}
  nu_kt = {}
  if cfg.nu == -1: # can't set inf in cfg
    model.nu = float('inf')
  else:
    model.nu = cfg.nu # default 1
  t0 = 1 if skip_first_hop else 0
  for t in range(t0, num_layers):
      d_in = dim_in if t == 0 else dim_out
      W_t["t=%d" % t] = GNNLayer(d_in, dim_out)
      K = min(model.max_k, t+1)
      for k in range(1, K+1):
          nu_kt["k=%d, t=%d" % (k,t)] = nn.Parameter(torch.ones(1), requires_grad=True)
  model.W_t = nn.ModuleDict(W_t)
  model.nu_kt = nn.ParameterDict(nu_kt)
  return model

# # retired
# def init_khop_GCN(model, dim_in, dim_out, num_layers, max_k=None):
#   """The k-hop GCN param initialiser, used for k_gnn and delay_gnn"""
#   model.num_layers = num_layers
#   model.max_k = cfg.gnn.layers_mp if max_k is None else max_k
#   if cfg.nu == -1: # can't set inf in cfg
#     model.nu = float('inf')
#   else:
#     model.nu = cfg.nu # default 1
#   for t in range(num_layers):
#       d_in = dim_in if t == 0 else dim_out
#       K = min(model.max_k, t+1)
#       for k in range(1, K+1):
#           W = GNNLayer(d_in, dim_out) # regular GCN layers
#           model.add_module('W_k{}_t{}'.format(k,t), W)
#   return model

# # retired
# def init_khop_LiteGCN(model, dim_in, dim_out, num_layers):
#   """The lightweight version of the k-hop GCN, with nu_{k,t}W_k instead of W_{k,t}, and W instead of W(t).
#   Will be used for delite_gnn and klite_gnn"""
#   model.num_layers = num_layers
#   model.max_k = cfg.gnn.layers_mp # cfg.delay.max_k
#   # make the W_k
#   W = []
#   for k in range(model.max_k):
#       W.append(GNNLayer(dim_in, dim_out))
#   model.W = nn.ModuleList(W)
#   # make K*T nu_{k,t} scalars
#   nu = {}
#   for t in range(num_layers):
#     for k in range(1, min(model.max_k, t+1)+1):
#       nu['%d,%d'%(k,t)] = nn.parameter.Parameter(torch.Tensor(1))
#   model.nu = nn.ParameterDict(nu)

#   return model

## stuff about max_k: no longer want this
  # if max_k is None:
  #   model.max_k = num_layers
  # else:
  #   model.max_k = max_k
  #   print('WARNING: Using max_k = %d; was this intentional?' % max_k)
  # if cfg.max_graph_diameter <= model.max_k:
  #   print("Warning: max_graph_diameter = %d; <= max_k, so setting max_k to max_graph_diameter" % cfg.max_graph_diameter)
  #   model.max_k = cfg.max_graph_diameter