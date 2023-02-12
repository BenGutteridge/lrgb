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

    # rho determines the max number of k-hop neighbourhoods per layer
    cfg.rho = -1 # default -1, which means no limit, ie vanilla DRew

register_config('rho', set_cfg_dropedge)
