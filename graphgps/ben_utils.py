import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

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

def add_k_hop_edges(dataset, K):
    """Add k-hop edges and labels, etc to PyG Dataset object"""

    graph_edge_cutoffs = dataset.slices['edge_index']
    # for each graph get the khops separately
    all_labels, all_indices = [], []
    print('Generating k-hop adjacencies...')
    for i in tqdm(range(len(graph_edge_cutoffs)-1)): # iterating over each graph in the dataset
        graph_edge_index = dataset.data.edge_index[:, graph_edge_cutoffs[i]:graph_edge_cutoffs[i+1]]
        k_hop_edges, _ = get_k_hop_adjacencies(graph_edge_index, K)
        assert torch.mean((k_hop_edges[0] == graph_edge_index).float())==1.0 # check that the 1-hop edges are the same
        
        # get k-hop edge cutoffs for labels
        cutoffs = torch.tensor([v.shape[-1] for v in k_hop_edges])
        k_hop_edges = torch.cat(k_hop_edges, dim=1) # list -> Tensor
            
        # make edge labels for k-hops
        k_hop_labels = []
        for i in range(len(cutoffs)):
            k = i + 1
            k_hop_labels.append(k * torch.ones(cutoffs[i]))
        k_hop_labels = torch.cat(k_hop_labels)
        
        # lists of edges and labels over entire dataset of graphs
        all_labels.append(k_hop_labels)
        all_indices.append(k_hop_edges)

    # stack all the edges and labels
    ei_slices = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor([v.shape[-1] for v in all_indices]), dim=0)])
    all_labels = torch.cat(all_labels)
    all_indices = torch.cat(all_indices, dim=1)

    # edit dataset directly
    dataset.data.edge_index = all_indices
    dataset.data.edge_attr = all_labels
    dataset.slices['edge_index'] = ei_slices
    dataset.slices['edge_attr'] = ei_slices

    return dataset