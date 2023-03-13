# parameter calculators

from ogb.utils.features import get_atom_feature_dims
from graphgps.encoder.voc_superpixels_encoder import VOC_node_input_dim
from torch_geometric.graphgym.config import cfg
from graphgps.drew_utils import get_task_id, set_jumping_knowledge
sort_and_removes_dupes = lambda mylist : sorted(list(dict.fromkeys(mylist)))

def set_d_fixed_params(cfg):
  set_jumping_knowledge()
  N = cfg.fixed_params.N
  if N > 0:
    cfg.gnn.dim_inner = return_hidden_dim(N)
    print('Hidden dim manually set to %d for fixed param count of %dk' % (cfg.gnn.dim_inner, int(N/1000)))
  else:
      print('Using given hidden dim of %d' % cfg.gnn.dim_inner)

def get_k_neighbourhoods(t):
  rho_l = max(1, t+1+1-cfg.rho)
  rho_nbhs = list(range(rho_l, t+1+1)) # rolling rho
  rho_nbhs = [((k % cfg.rho_max) + cfg.k_max) if k>cfg.rho_max else k for k in rho_nbhs]
  sp_nbhs = list(range(1, min(t+1, cfg.k_max)+1))
  return sort_and_removes_dupes(rho_nbhs + sp_nbhs)

def get_num_fc_drew(L):
  """Base number of FC layers in DRew MP"""
  num_fc = 0
  print('k_max = %02d, rho = %02d\nLayer k-neighbourhoods:' % (cfg.k_max, cfg.rho))
  assert cfg.rho < L and cfg.rho >= 0, "Error: rho >= L or < 0"
  assert cfg.k_max >= 0, 'Error: k_max < 0'
  for t in range(L): # ignores skipped first layer for relational. TODO: sort
    k_nbhs = get_k_neighbourhoods(t)
    toprint = ' '.join([str(i).ljust(2) if i in k_nbhs else 'X'.ljust(2) for i in range(1, k_nbhs[-1]+1)])
    print('\t%02d: %s' % (t, toprint))
    num_fc += len(k_nbhs)
  return num_fc

def return_hidden_dim(N):
  """Return hidden dimension for MPNN based on """
  # number of FC layers in message passing
  N *= 0.99 # a little spare
  L = cfg.gnn.layers_mp
  if cfg.gnn.stage_type.startswith('delay_gnn') & (cfg.model.type == 'gnn'):
    num_fc = get_num_fc_drew(L)
  elif cfg.gnn.stage_type == 'delay_share_gnn': # weight sharing - only one W mp per layer
    num_fc = L
  elif cfg.gnn.layer_type in ['share_drewgatedgcnconv', 'gatedgcnconv_noedge']:
    num_fc = 4*L # A,B,D,E (no C currently)
  elif cfg.model.type == 'alpha_gated_gnn':
    num_fc = 2*min(L, cfg.k_max)*L + 2*L
  elif cfg.gnn.layer_type == 'drewgatedgcnconv':
    num_fc = 2*L + get_num_fc_drew(L)*2 # A,D and B_{k},E_{k}
  elif cfg.gnn.layer_type == 'gatedgcnconv':
    num_fc = 5*L # A,B,C,D,E #TODO check. surely C won't be d**2, it'll be d*|E|?
  elif cfg.gnn.layer_type in 'gcnconv':
    num_fc = L
  elif cfg.gnn.stage_type == 'alpha_gnn':
    num_fc = min(L, cfg.k_max) * L
  else:
    raise ValueError('Unknown stage/layer type combination; stage_type: {0}, layer_type: {1}'.format(cfg.gnn.stage_type, cfg.gnn.layer_type))
  
  # accounting for concatenation-based jumping knowledge MLPs in the head
  if 'cat_jk' in cfg.gnn.head:
    n_jk_terms = cfg.gnn.layers_mp
    if 'rho' in cfg.gnn.head: n_jk_terms -= (cfg.rho + cfg.k_max - 1)
    num_fc += n_jk_terms 

  # other params and summation
  post_mp = cfg.gnn.layers_post_mp - 1       # 2-layer MLP at end -- not counting final layer to num classes
  num_bn = cfg.gnn.batchnorm * num_fc        # batch norm layers
  task = get_task_id()
  if task == 'pept':
    node_embed = sum(get_atom_feature_dims())
    head = 10 # number of classes -- 11 for struct, close enough
    d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+head, -N)
  elif task == 'voc':
    node_embed = VOC_node_input_dim
    head = 21 # number of classes
    d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+head+post_mp, -N)
  elif task == 'coco':
    node_embed = VOC_node_input_dim
    head = 81 # number of classes
    d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+head+post_mp, -N)
  elif task == 'pcqm':
    node_embed = sum(get_atom_feature_dims())
    post_mp += 1 # head is a fc, post-mp layer
    d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+post_mp, -N)
  else:
    raise ValueError('Unknown dataset format {0}'.format(cfg.dataset.format))
  
  return round(d)

def solve_quadratic(a,b,c):
  # Solve the quadratic equation ax**2 + bx + c = 0
  d = (b**2) - (4*a*c)
  # find two solutions
  sol1 = (-b-d**.5)/(2*a)
  sol2 = (-b+d**.5)/(2*a)
  if sol1 > 0 and sol2 < 0:
    return sol1
  else:
    return sol2

