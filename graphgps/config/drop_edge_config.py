from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_dropedge(cfg):
    r'''
    Arguments for reducing parameter count by dropping edges, i.e. dropping terms from the multi-hop aggregation.

    k_max: the 'core' k-neighbourhoods that are aggregated for (nu)DRew
    rho: sets the width of the 'wavefront' k-hop that is aggregated beyond the 'core' k_max k-neighbourhoods
    rho_max: sets the outer limit that the 'wavefront' can reach -- caps the number of k-hop graph edge indices we need
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.k_max = int(1e6)    # default (stand-in for) inf, no outer limit
    cfg.rho = 0             # default no extra outer-most layers
    cfg.rho_max = int(1e6)  # If rho > 0, this gives a max k-neighbourhood to go up to; i.e. the wavefront stops at that point. Default: inf


register_config('dropedge', set_cfg_dropedge)
