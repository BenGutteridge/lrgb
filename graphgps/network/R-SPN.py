import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from typing import Callable, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, List

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset

from torch_sparse import SparseTensor, matmul

class RSPN(nn.Module):
    """
    R-SPN with dense MLP setup

    num parameters scales as d**2 * L(K+|E|) for hidden dim d, L layers, num edges types E
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        assert not cfg.rbar_v2, "rbar_v2 not implemented yet"
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        else:
            self.pre_mp = nn.Identity()

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."
        assert cfg.spn.K > 0, "cfg.spn.K must be set to > 0"
        assert cfg.spn.K <= cfg.max_graph_diameter, "cfg.spn.K must be <= cfg.max_graph_diameter"

        K = cfg.spn.K   # number of SP-hops (fixed for every layer)
        d = dim_in
        gin_nn_post = nn.Sequential(
            # pyg_nn.Linear(d, d), 
            # nn.ReLU(),
            # pyg_nn.Linear(d, d),
            )
        alpha = torch.nn.Parameter(torch.randn(K))
        mlp_s = nn.Sequential(pyg_nn.Linear(d, d), nn.ReLU()) # self-connection
        mlp_k = nn.ModuleList([nn.Sequential(pyg_nn.Linear(d, d), nn.ReLU()) for k in range(2, K+1)]) # k-hop connections
        mlp_e = nn.ModuleList([nn.Sequential(pyg_nn.Linear(d, d), nn.ReLU()) for _ in range(len(cfg.edge_types))]) # 1-hop edge-type connections
        all_modules = dict(gin_nn_post=gin_nn_post, # making an attr of the module so it shows in model summary (hopefully)
                            alpha=alpha,
                            mlp_s=mlp_s,
                            mlp_k=mlp_k,
                            mlp_e=mlp_e)

        self.gnn_layer = RSPNConvLayer(all_modules)   

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        batch = self.encoder(batch)
        batch = self.pre_mp(batch)
        for _ in range(cfg.gnn.layers_mp):
            x_res = batch.x
            batch = self.gnn_layer(batch)
            # taken out of GINEConvLayer
            batch.x = F.relu(batch.x)
            batch.x = F.dropout(batch.x, p=cfg.gnn.dropout, training=self.training)
            if cfg.gnn.residual:
                batch.x = x_res + batch.x  # residual connection
        batch = self.post_mp(batch)
        return batch


register_network('R-SPN', RSPN)

class RSPNConvLayer(nn.Module):
    """
    Just a nn.Module wrapper for the MessagePassing SPNConv
    """
    def __init__(self, all_modules):
        super().__init__()
        self.model = RSPNConv(all_modules)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, 
                             batch.edge_attr, # for k-hop labels
                             )
        return batch


class RSPNConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, modules: nn.ModuleDict,
                 eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(RSPNConv, self).__init__(**kwargs)
        self.nn_post = modules['gin_nn_post']
        self.alpha = modules['alpha'] # for the k-hop aggregations weights
        self.mlp_s = modules['mlp_s'] # for the self-connection ((1+eps) weighted)
        self.mlp_k = modules['mlp_k'] # for the k-hop aggregations (list)
        self.mlp_e = modules['mlp_e'] # for the edge-type aggregations (list)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn_post)
        self.eps.data.fill_(self.initial_eps)

    def forward(self,
                    x: List[Tensor],
                    edge_indices: List[Adj], # TODO: what is adj? a 2xN tensor?
                    edge_attr: OptTensor = None,
                    size: Size = None) -> Tensor:

        # if isinstance(x, Tensor): # TODO: does every element of xs need to be an OptPairTensor?
        #     x: OptPairTensor = (x, x)
        # # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)

        A = lambda k : edge_indices[:, edge_attr[:,0]==k]
        A_edge = lambda e : edge_indices[:, edge_attr[:,1]==int(e)] # using -1 to distinguish k>1 hop edges
        mlp = lambda k : self.mlp_k[k-2] # k=2 is the first element in the list
        mlp_e = lambda i : self.mlp_e[i]
        alpha_weights = F.softmax(self.alpha, dim=0) # convex combination
        alpha = lambda k : alpha_weights[k-1] # k=1 is the first element in the list

        # weighted self connection
        out = (1 + self.eps) * self.mlp_s(x)

        # TODO add alpha weights
        # k=1
        for i, e in enumerate(cfg.edge_types):
            out += alpha(1) * mlp_e(i)(self.propagate(A_edge(e), x=x, size=size))
        # k>1
        for k in range(2, cfg.spn.K):
            out += alpha(k) * mlp(k)(self.propagate(A(k), x=x, size=size))

        return self.nn_post(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    # def __repr__(self):
    #     return '{}(nn={})'.format(self.__class__.__name__, self.nn)