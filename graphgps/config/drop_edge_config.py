from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_dropedge(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.rho = 0             # default no extra outer-most layers
    cfg.rho_max = int(1e6)  # If rho > 0, this gives a max k-neighbourhood to go up to; i.e. the wavefront stops at that point. Default: inf
    cfg.k_max = int(1e6)    # default (stand-in for) inf, no outer limit

    """
    DRew:               k_max = inf, rho = 0
    rhoDRew (v1):       k_max = 0 (where we don't keep k=1 neighbourhood)
    rhoDRew (v2):       k_max = 1 
    'core' DRew:        rho = 0 (max_k is outer limit, no aggregations beyound k_max hops)

    N.B. I much prefer the name rho for  what is currently k_max and pi /omega/rho_outer for what is currently rho
    """

register_config('dropedge', set_cfg_dropedge)
