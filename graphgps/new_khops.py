import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
from torch_geometric.graphgym.config import cfg
from os.path import join, exists
from torch_geometric.data import Data

def add_k_hop_edges(dataset, K, format, name):
  print('Stage type %s, model type %s, using %d-hops' % (cfg.gnn.stage_type, cfg.model.type, K))
  # get k-hop edge amended dataset - either load or make it
  cluster_filedir = '/data/beng' # data location for aimscdt cluster
  local_filedir = 'graphgps/loader/k_hop_indices' # data location for verges/mac, local
  if not exists(cluster_filedir):
      print('Not on aimscdt cluster, using local data storage.')
      filedir = local_filedir
  else:
      print('On aimscdt cluster, using cluster data storage.')
      filedir = join(cluster_filedir, 'k_hop_indices')

  # check if files exist already
  if not exists(join(filedir, "%s-%s_k=%02d.pt" % (format, name, K))):
    print('Edge index files not found for %s-%s_k=%02d; making them now...' % (format, name, K))
    get_k_hop_edges(dataset, K, filedir, format, name) # if they don't, make them

  # load files
  all_graphs = [torch.load(join(filedir, "%s-%s_k=%02d.pt" % (format, name, k))) for k in range(1,K+1)] # [K,N,2,d]
  all_hops = [list(n) for n in zip(*all_graphs)] # Transposing. n is graph; all_hops indexed by graph. [N,K,2,d]
  labels = []   # get k-hop labels
  for n in all_hops:
    for k, khop in enumerate(n,1):
      labels += [1*k]*khop.shape[-1]
  labels = torch.tensor(labels, dtype=torch.long)
  all_hops = [torch.cat(n, dim=1) for n in all_hops] # [K,2,d]
  count, ei_slices = 0 , [0]
  for d in all_hops:
    count += d.shape[-1]
    ei_slices.append(count)
  ei_slices = torch.tensor(ei_slices)
  all_hops = torch.cat(all_hops, dim=1) # [2,d]
  # set to dataset
  dataset.data.edge_index = all_hops
  dataset.data.edge_attr = labels
  dataset.slices['edge_index'] = dataset.slices['edge_attr'] = ei_slices

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


def get_k_hop_edges(dataset, K, filedir, format, name):
  """take regular dataset, save k-hop edges"""
  # we're saving a list of k-hop edge indices
  edge_indices = dataset.data.edge_index
  slices = dataset.slices['edge_index']
  idxs = [[] for _ in range(K)]
  for i in tqdm(range(len(slices)-1)):
  # for edge_index in [edge_indices]:
    edge_index = edge_indices[:, slices[i]:slices[i+1]]
    idxs[0].append(edge_index) # 1-hop
    try:
      tmp = to_dense_adj(edge_index).float()
    except:
      print('Offending tensor:\nedge_index:\n', edge_index, '\nedge_index.shape:', edge_index.shape)
      adj = None # if it fails, set adj to None to force an errorx
    adj = tmp.to_sparse().float()
    matrices = [tmp]
    for k in range(2, K+1):
      tmp = torch.bmm(adj, tmp)
      for j in range(tmp.shape[-1]):
        tmp[0, j, j] = 0 # remove self-connections
      tmp = (tmp>0).float() # remove edge multiples
      for m in matrices:
        tmp -= m
      tmp = (tmp>0).float() # remove -ves, cancelled edges
      idx, _ = dense_to_sparse(tmp) # outputs int64, which we want
      matrices.append(tmp)
      idxs[k-1].append(idx)
  for k, ei_k in enumerate(idxs, 1):
    filepath = join(filedir, "%s-%s_k=%02d.pt" % (format, name, k))
    if not exists(filepath):
      print('Saving edge indices for k=%d to %s...' % (k, filepath))
      torch.save(ei_k, filepath)