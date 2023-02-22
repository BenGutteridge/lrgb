import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn.models import JumpingKnowledge
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_head
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
import torch_geometric.graphgym.models.pooling  # noqa, register module


class GNNGraphHeadJK(nn.Module):
    """
    Used by peptides func,

    GNN prediction head for graph prediction tasks.
    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.

    NB. this is modified with a version of Jumping Knowledge: 
    - each layer output before a k-neighbourhood is dropped by rho/dropout is saved
    - for the M graph node embeddings, we perform a max pooling over all M for each node feature *element*
    - then proceed as usual: graph level pooling, final MLP, etc
    - OR concat

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(self, dim_in, dim_out):
        super(GNNGraphHeadJK, self).__init__()
        if 'rho' in cfg.gnn.head: assert ((cfg.rho > 0) & (cfg.rho+cfg.k_max < cfg.gnn.layers_mp)), 'Error, invalid arguments for JK head %s: L=%d and rho=%d' % (cfg.gnn.head, cfg.gnn.layers_mp, cfg.rho)
        print('Using jumping knowledge, JK aggregation mode: %s' % cfg.gnn.head)
        if 'cat' in cfg.gnn.head:
            n_agg_terms = cfg.gnn.layers_mp
            if 'rho' in cfg.gnn.head: n_agg_terms -= (cfg.rho + cfg.k_max - 1)
            self.jumping_knowledge = nn.Sequential(
                JumpingKnowledge('cat'), # stack node feats then linear layer to compress again
                nn.Linear(dim_in * n_agg_terms, dim_in)
            )
        elif 'max' in cfg.gnn.head:
            self.jumping_knowledge = JumpingKnowledge('max')
        else:
            raise NotImplementedError
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        batch, h = batch # unroll tuple - takes the current batch output and each layer's node feats, h
        if 'rho' in cfg.gnn.head:
            # assumes rho+1 neighbourhoods aggregated (3 outermost + k=1) and appends from before first dropped neighbourhood
            h = h[cfg.rho+cfg.k_max:] + [batch.x]
        else: 
            h = h[1:] + [batch.x]
        h_jk = self.jumping_knowledge(h) # don't include pre-MP; include final output
        graph_emb = self.pooling_fun(h_jk, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label

register_head('max_jk_graph', GNNGraphHeadJK)
register_head('cat_jk_graph', GNNGraphHeadJK)
register_head('rho_max_jk_graph', GNNGraphHeadJK) # for only including some layers, based on rho
register_head('rho_cat_jk_graph', GNNGraphHeadJK)


class GNNInductiveNodeHeadJK(nn.Module):
    """
    Used by VOC,

    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNInductiveNodeHeadJK, self).__init__()
        if 'rho' in cfg.gnn.head: assert ((cfg.rho >= 0) & (cfg.rho+cfg.k_max < cfg.gnn.layers_mp)), 'Error, invalid arguments for JK head %s: L=%d and rho=%d' % (cfg.gnn.head, cfg.gnn.layers_mp, cfg.rho)
        layers_post_mp = cfg.gnn.layers_post_mp
        print('Using jumping knowledge, JK aggregation mode: %s' % cfg.gnn.head)
        if 'cat_jk' in cfg.gnn.head:
            n_agg_terms = cfg.gnn.layers_mp
            if 'rho' in cfg.gnn.head: n_agg_terms -= (cfg.rho + cfg.k_max - 1)
            self.jumping_knowledge = nn.Sequential(
                JumpingKnowledge('cat'), # stack node feats then linear layer to compress again
                nn.Linear(dim_in * n_agg_terms, dim_in)
            )
            # layers_post_mp -= 1 # optional: replace one of the MLP layers with this linear one
        elif 'max_jk' in cfg.gnn.head:
            self.jumping_knowledge = JumpingKnowledge('max')
        else:
            raise NotImplementedError('Head: ', cfg.gnn.head)
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        batch, h = batch # unroll tuple - takes the current batch output and each layer's node feats, h
        if 'rho' in cfg.gnn.head:
            # assumes rho+1 neighbourhoods aggregated (3 outermost + k=1) and appends from before first dropped neighbourhood
            h = h[cfg.rho+cfg.k_max:] + [batch.x]
        else: 
            h = h[1:] + [batch.x]
        h = self.jumping_knowledge(h)
        batch.x = self.layer_post_mp(h)
        pred, label = self._apply_index(batch)
        return pred, label


register_head('max_jk_inductive_node', GNNInductiveNodeHeadJK)
register_head('cat_jk_inductive_node', GNNInductiveNodeHeadJK)
register_head('rho_max_jk_inductive_node', GNNInductiveNodeHeadJK)
register_head('rho_cat_jk_inductive_node', GNNInductiveNodeHeadJK)