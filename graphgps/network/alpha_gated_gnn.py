import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.gcnii_conv_layer import GCN2ConvLayer
from graphgps.layer.mlp_layer import MLPLayer
from graphgps.layer.drew_gatedgcn_layer import DRewGatedGCNLayer

class AlphaGatedGNN(torch.nn.Module):
    """
    Fully connected up to kmax, dense alpha Gated
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
            self.encoder = torch.nn.Sequential([self.encoder, self.pre_mp])

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = DRewGatedGCNLayer
        self.model_type = cfg.gnn.layer_type
        layers = []
        self.K = min(cfg.k_max, cfg.gnn.layers_mp)
        cfg.nu = -1 # can't be using any delay -- but want to keep using default nu=1, so it doesn't show up in run name
        assert cfg.nu == -1
        for t in range(cfg.gnn.layers_mp):
            layers.append(conv_model(self.K-1, dim_in, dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual))
        self.gnn_layers = torch.nn.ModuleList(layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)


    def forward(self, batch):
        batch = self.encoder(batch)             # Encoder (+ Pre-MP)
        xs = [None] * len(self.gnn_layers)
        xs[self.K-1] = batch.x
        for t in range(len(self.gnn_layers)):   # Message Passing
            # xs.append(batch.x)
            batch = self.gnn_layers[t](self.K-1, xs, batch) 
        batch = self.post_mp(batch)             # (Post-MP +) Head
        return batch


register_network('alpha_gated_gnn', AlphaGatedGNN)
