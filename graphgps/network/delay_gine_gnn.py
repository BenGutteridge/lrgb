import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.delay_gine_conv_layer import DelayGINEConvLayer
from graphgps.custom_conv.gin_agg import GINEAggregation

class DelayGINEGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.

    This is a copy of the CustomGNN class in graphgps/network/custom_gnn.py
    that I am adapting for new delay / r* GIN(E) layers
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        self.model_type = cfg.gnn.layer_type

        """
        each layer t needs:
        - an aggregation over k hop edges for k in [1, t+1] -- GINEAggregation
        - each layer does consist of an aggregation plus the extra stuff, but we have separate aggregations
        - so we need all our k-hop aggs and then a final Conv layer to put them all together for an arbitrary number
        """
        self.k_hop_agg = GINEAggregation(dim_in)

        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(DelayGINEConvLayer(dim_in,
                                             dim_in,
                                             dropout=cfg.gnn.dropout,
                                             residual=cfg.gnn.residual))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


register_network('delay_gine_gnn', DelayGINEGNN)
