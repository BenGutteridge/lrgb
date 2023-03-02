import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
from torch_geometric.graphgym.config import cfg
import os
from torch_geometric.data import Data

def get_task_id():
  if cfg.dataset.name.startswith('peptides'):
    return 'pept'
  elif cfg.dataset.format == 'PyG-VOCSuperpixels':
    return 'voc'
  elif cfg.dataset.format == 'PyG-COCOSuperpixels':
    return 'coco'
  elif cfg.dataset.name == 'PCQM4Mv2Contact-shuffle':
    return 'pcqm'
  else:
    raise NotImplementedError

default_heads = {
  'pept': 'graph',
  'voc': 'inductive_node',
}

def set_jumping_knowledge():
  if cfg.jk_mode == 'none':
    return # uses default head
  task = get_task_id()
  try: cfg.gnn.head = '%s_jk_%s' % (cfg.jk_mode, default_heads[task])
  except: assert False, 'Error: JK head for %s not yet defined.' % task


def custom_set_out_dir(cfg, cfg_fname, name_tag, default=False):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = get_run_name(cfg_fname, default)
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def get_run_name(cfg_fname, default):
  dataset_name = ('-' + cfg.dataset.name) if cfg.dataset.name!='none' else ''
  if default:
    return os.path.splitext(os.path.basename(cfg_fname))[0]
  elif cfg.model.type == 'gnn':
    model = cfg.gnn.stage_type if cfg.beta==1 else 'beta=%d' % cfg.beta
  elif 'custom_gnn' in cfg.model.type:
    model = cfg.gnn.layer_type
  else:
    model = cfg.model.type
  if '+' in cfg.dataset.node_encoder_name: # note if PE used
    model += '_%s' % cfg.dataset.node_encoder_name.split('+')[-1]
  if cfg.nu != 1:
    nu = '%02d' % cfg.nu if cfg.nu != -1 else 'inf'
    model += '_nu=%s' % nu
  if cfg.k_max < cfg.gnn.layers_mp:
    model += '_kmax=%02d' % cfg.k_max
  if cfg.rho > 0:
    model += '_rho=%02d' % cfg.rho
    if cfg.rho_max < cfg.gnn.layers_mp:
      model += '_rho_max=%02d' % cfg.rho_max
  if cfg.jk_mode != 'none':
    model += '_JK=%s' % cfg.jk_mode
  if cfg.agg_weights.use:
    model += '_weights'
  if cfg.agg_weights.convex_combo:
    model += '_CC'
  if cfg.spn.K != 0:
    model += '_K=%02d' % cfg.spn.K
  # run_name = "%s%s_%s_bs=%04d_d=%03d_L=%02d" % (cfg.dataset.format, dataset_name, model, cfg.train.batch_size, cfg.gnn.dim_inner, cfg.gnn.layers_mp) # with BS
  run_name = "%s%s_%s_d=%03d_L=%02d" % (cfg.dataset.format, dataset_name, model, cfg.gnn.dim_inner, cfg.gnn.layers_mp) # without BS
  cut = ['ides', 'ural', 'tional', 'PyG-', 'OGB-']
  for c in cut:
    run_name = run_name.replace(c, '')
  return run_name


def get_edge_labels(dataset):
  """takes in PyG dataset object and spits out some edge labels"""
  e = dataset.data.edge_attr
  if 'peptides' in dataset.root:
    print('Getting edge labels for peptides dataset...')
    # [bond_type, is_stereo, is_conj]
    edge_labels = e[:,0]*1 + e[:,2]*10 + e[:,1]*100  # columns
  elif 'QM9' in dataset.root:
    print('Getting edge labels for QM9 dataset')
    # one-hot vectors for bond type
    edge_labels = torch.argmax(e, dim=1)
  else:
    raise NotImplementedError("Dataset '%s' not supported" % dataset.folder)
  return edge_labels


# K <= BETA ADJACENCIES

def add_k_leq_beta_adj(dataset, beta=1):
  """beta = 1 defaults to vanilla GCN"""
  all_indices = []
  graph_edge_cutoffs = dataset.slices['edge_index']
  for i in tqdm(range(len(graph_edge_cutoffs)-1)): # iterating over each graph in the dataset
    graph_edge_index = dataset.data.edge_index[:, graph_edge_cutoffs[i]:graph_edge_cutoffs[i+1]]
    if graph_edge_index.shape[-1] == 0:
      print('Warning: graph with no edges. i: %d, graph_edge_cutoffs[i]: %d, graph_edge_cutoffs[i+1]: %d' % (i, graph_edge_cutoffs[i], graph_edge_cutoffs[i+1]))
      print('Empty graph skipped.')
      k_hop_edges = [torch.empty((2,0), dtype=torch.long)]
    else:
      k_hop_edges = get_k_leq_beta_adj(graph_edge_index, beta)
    all_indices.append(k_hop_edges)
  ei_slices = torch.tensor([0] + [indices.shape[-1] for indices in all_indices]).cumsum(dim=0)
  all_indices = torch.cat(all_indices, dim=1)
  dataset.data.edge_index = all_indices
  # slices
  dataset.slices['edge_index'] = ei_slices
  dataset.slices['edge_attr'] = ei_slices
  try:
    dataset.data.__delattr__('edge_attr')
    print('Unused edge attrs deleted')
  except:
    print('No edge attrs to delete')
  assert dataset.data.edge_attr is None
  return dataset
  

def get_k_leq_beta_adj(edge_index, beta):
  """Return k <= beta - hop adjacency matrix"""
  try:
    tmp = to_dense_adj(edge_index).float()
  except:
    print('Offending tensor:\nedge_index:\n', edge_index, '\nedge_index.shape:', edge_index.shape)
  adj = tmp.to_sparse().float()
  for k in range(2, beta+1):
    tmp += torch.bmm(adj, tmp)
  for i in range(tmp.shape[-1]):
    tmp[0, i, i] = 0 # remove self-connections
  tmp = (tmp>0).float() # remove edge multiples
  edge_idx, _ = dense_to_sparse(tmp)
  return edge_idx


### K-HOP ADJACENCIES

def add_k_hop_edges(dataset, K, edge_labels=None):
    """Add k-hop edges and labels, etc to PyG Dataset object"""

    graph_edge_cutoffs = dataset.slices['edge_index']
    # for each graph get the khops separately
    all_labels, all_indices, all_edge_type_labels = [], [], []
    print('Generating k-hop adjacencies...')
    for i in tqdm(range(len(graph_edge_cutoffs)-1)): # iterating over each graph in the dataset
        graph_edge_index = dataset.data.edge_index[:, graph_edge_cutoffs[i]:graph_edge_cutoffs[i+1]]
        if graph_edge_index.shape[-1] == 0:
          print('Warning: graph with no edges. i: %d, graph_edge_cutoffs[i]: %d, graph_edge_cutoffs[i+1]: %d' % (i, graph_edge_cutoffs[i], graph_edge_cutoffs[i+1]))
          print('Empty graph skipped.')
          k_hop_edges = [torch.empty((2,0), dtype=torch.long)]
        else:
          k_hop_edges, _ = get_k_hop_adjacencies(graph_edge_index, K)
          assert torch.mean((k_hop_edges[0] == graph_edge_index).float())==1.0 # check that the 1-hop edges are the same
          
        # get k-hop edge cutoffs for labels
        cutoffs = torch.tensor([v.shape[-1] for v in k_hop_edges])
        k_hop_edges = torch.cat(k_hop_edges, dim=1) # list -> Tensor
            
        # make edge labels for k-hops
        k_hop_labels = []
        for j in range(len(cutoffs)):
            k = j + 1
            k_hop_labels.append(k * torch.ones(cutoffs[j]))
        k_hop_labels = torch.cat(k_hop_labels)

        if edge_labels is not None:
          graph_edge_labels = edge_labels[graph_edge_cutoffs[i]:graph_edge_cutoffs[i+1]]
          graph_edge_labels = [graph_edge_labels]
          for j in range(1,len(cutoffs)):
              graph_edge_labels.append(-1 * torch.ones(cutoffs[j])) # all >1-hop edges are labeled -1
          graph_edge_labels = torch.cat(graph_edge_labels)
          all_edge_type_labels.append(graph_edge_labels)
        
        # lists of edges and labels over entire dataset of graphs
        all_labels.append(k_hop_labels)
        all_indices.append(k_hop_edges)

    # stack all the edges and labels
    ei_slices = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor([v.shape[-1] for v in all_indices]), dim=0)])
    all_labels = torch.cat(all_labels)
    all_indices = torch.cat(all_indices, dim=1)

    # edit dataset directly
    dataset.data.edge_index = all_indices
    # dataset.data.edge_attr = all_labels
    dataset.slices['edge_index'] = ei_slices
    dataset.slices['edge_attr'] = ei_slices

    # stack on the edge type labels as well, if needed
    if edge_labels is not None: # COMBINE
      all_edge_type_labels = torch.cat(all_edge_type_labels)
      dataset.data.edge_attr = torch.stack([all_labels, all_edge_type_labels]).T
    else:
      dataset.data.edge_attr = all_labels

    print('Checking correct conversion...')
    count  = 0
    for i in tqdm(range(len(dataset))):
      if not torch.equal(dataset.get(i).edge_attr, dataset.data.edge_attr[ei_slices[i]:ei_slices[i+1]]):
        # print('Graph %d not changed in dataset._data_list; setting manually' % i)
        count += 1
        dataset._data_list[i] = Data(x=dataset.get(i).x,
                                  edge_index=dataset.data.edge_index[:, ei_slices[i]:ei_slices[i+1]],
                                  edge_attr=dataset.data.edge_attr[ei_slices[i]:ei_slices[i+1]],
                                  y=dataset.get(i).y)
      assert torch.equal(dataset.get(i).edge_attr, dataset.data.edge_attr[ei_slices[i]:ei_slices[i+1]]) # check that the conversion worked
    if count > 0: print('Warning: %d/%d graphs not changed in dataset._data_list; have been set manually' % (count, len(dataset)))

    return dataset


# Copy pasted funcs from ben_utils -- TODO: fix this so we don't have copies of code about
def get_k_hop_adjacencies(edge_index, max_k, stack_edge_indices=False):
  """Return list of matrices/edge indices for 1,..,k-hop adjacency matrices
  n.b. binary matrix
  n.b. pretty inefficient"""
  try:
    tmp = to_dense_adj(edge_index).float()
  except:
    print('Offending tensor:\nedge_index:\n', edge_index, '\nedge_index.shape:', edge_index.shape)
  adj = tmp.to_sparse().float()
  idxs, matrices = [edge_index], [tmp]
  cutoffs, n_edges_per_k = [0], edge_index.shape[-1]
  for k in range(2, max_k+1):
    tmp = torch.bmm(adj, tmp)
    for i in range(tmp.shape[-1]):
      tmp[0, i, i] = 0 # remove self-connections
    tmp = (tmp>0).float() # remove edge multiples
    for m in matrices:
      tmp -= m
    tmp = (tmp>0).float() # remove -ves, cancelled edges
    idx, _ = dense_to_sparse(tmp) # outputs int64, which we want
    matrices.append(tmp)
    idxs.append(idx)
    cutoffs.append(n_edges_per_k)
    n_edges_per_k += idx.shape[-1]
    if torch.sum(tmp) == 0:
      break # adj matrix is empty
  cutoffs.append(n_edges_per_k)
  if stack_edge_indices:
    idxs = torch.cat(idxs, dim=-1)
  # matrices = torch.stack(matrices, dim=1)
  return idxs, get_khop_labels(cutoffs)

def get_khop_labels(cutoffs):
# generates k-hop edge labels from cutoff tensor - used when all k-hop indices are put in Data.edge_index
    num_per_k = [cutoffs[i+1]-cutoffs[i] for i in range(len(cutoffs)-1)]
    edge_khop_labels = []
    for k in range(1, len(cutoffs)):
        edge_khop_labels.append(k*torch.ones(num_per_k[k-1]))
    return torch.cat(edge_khop_labels).reshape((-1, 1))

###