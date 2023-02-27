# parameter calculators

from ogb.utils.features import get_atom_feature_dims
from graphgps.encoder.voc_superpixels_encoder import VOC_node_input_dim
from torch_geometric.graphgym.config import cfg
from graphgps.ben_utils import get_task_id, set_jumping_knowledge
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
  rho_nbhs = list(range(max(1, t+1+1-cfg.rho), t+1+1))
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
  # rho = cfg.rho if cfg.rho != -1 else 1e6 # inplace of inf to avoid 0*inf=nan error later
  # Lq = min(rho, L)    # number of quadratically scaling param layers
  # Ll = max(0, L-rho)  # number of linearly scaling param layers
  # num_fc = (Lq**2+Lq)/2 + rho*Ll # number of d**2 layers
  # if cfg.rho > 0:
  #   num_fc += cfg.k_max*Ll # if the k=1 connection is maintained throughout
  return num_fc

def return_hidden_dim(N):
  """Return hidden dimension for MPNN based on """
  # number of FC layers in message passing
  N *= 0.99 # a little spare
  L = cfg.gnn.layers_mp
  if cfg.gnn.stage_type == 'delay_gnn':
    num_fc = get_num_fc_drew(L)
  elif cfg.gnn.stage_type == 'delay_share_gnn': # weight sharing - only one W mp per layer
    num_fc = L
  elif cfg.gnn.layer_type in 'gcnconv':
    num_fc = L
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
  elif sol2 > 0 and sol1 < 0:
    return sol2
  else:
    sols = [sol1, sol2]
    choice = input('Solutions are {0} and {1}, which one would you like to use (Use 0/1)?\n'.format(sols))
    return sols[choice]

# def pept_drew(N, L, rho):
#   """Calculate hidden dim for peptides func and struct"""
#   num_fc = get_num_fc_drew(L, rho)
#   num_bn = cfg.gnn.batchnorm * num_fc        # batch norm layers
#   atom_embeds = sum(get_atom_feature_dims()) # atom embeddings
#   head = 10                                  # number of classes -- 11 for struct, close enough
#   d = solve_quadratic(num_fc, num_bn+atom_embeds+head, -N)
#   return round(d)

# def pept_gcn(N, L, rho):
#   """Calculate hidden dim for peptides func and struct"""
#   num_fc = L
#   num_bn = cfg.gnn.batchnorm * num_fc        # batch norm layers
#   atom_embeds = sum(get_atom_feature_dims()) # atom embeddings
#   head = 10                                  # number of classes -- 11 for struct, close enough
#   d = solve_quadratic(num_fc, num_bn+atom_embeds+head, -N)
#   return round(d)

# def voc_drew(N, L, rho):
#   """Calculate hidden dim for voc superpixels"""
#   num_fc = get_num_fc_drew(L, rho)
#   num_bn = cfg.gnn.batchnorm * num_fc        # batch norm layers
#   post_mp = cfg.gnn.layers_post_mp - 1       # 2-layer MLP at end -- not counting final layer to num classes
#   node_embed = VOC_node_input_dim            # atom embeddings
#   head = 21                                  # number of classes for voc
#   d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+head+post_mp, -N)
#   return round(d)


# def coco_drew(N, L, rho):
#   """Calculate hidden dim for coco superpixels"""
#   num_fc = get_num_fc_drew(L, rho)
#   num_bn = cfg.gnn.batchnorm * num_fc        # batch norm layers
#   post_mp = cfg.gnn.layers_post_mp - 1       # 2-layer MLP at end -- not counting final layer to num classes
#   node_embed = VOC_node_input_dim            # atom embeddings
#   head = 81                                  # number of classes for voc
#   d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+head+post_mp, -N)
#   return round(d)

# def pcqmcontact_drew(N, L, rho):
#   """Calculate hidden dim for pcqm contact"""
#   num_fc = get_num_fc_drew(L, rho)
#   num_bn = cfg.gnn.batchnorm * num_fc        # batch norm layers
#   post_mp = cfg.gnn.layers_post_mp           # MLPs at end -- inlcudes head since this is link pred
#   node_embed = sum(get_atom_feature_dims())
#   d = solve_quadratic(num_fc+post_mp, num_bn+node_embed+post_mp, -N)
#   return round(d)

# def pcqm_gcn(N, L, rho):
#   """
#   (L+1+0.5)*d**2 for MP layers, a 1-layer MLP at end, and ~0.5d^2 for node embedding
#   """
#   N*=.98
#   return round((N/(L+1+0.5))**0.5)

# def pcqm_drew(N, L, rho):
#   """
#   (L**2+L+2+1)/2 * d**2 for MP layers, 1-layer MLP at end, and ~0.5d^2 for node embedding
#   """
#   Lq = min(rho, L) # number of quadratically scaling param layers
#   Ll = max(0, L-rho) # number of linearly scaling param layers
#   total = (Lq**2+Lq)/2 + rho*Ll + 1 + 0.5
#   d = (N/total)**0.5
#   return round(d)

# calc_dict = {
#   'voc_gcn': voc_gcn,
#   'voc_drew': voc_drew,
#   'pept_gcn': pept_gcn,
#   'pept_drew': pept_drew,
#   'pcqm_gcn': pcqm_gcn,
#   'pcqm_drew': pcqm_drew,
#   'vocsuperpixels_drew': vocsuperpixels_drew,
# }

# num_fc_dict = {
#   'voc_gcn': voc_gcn,
