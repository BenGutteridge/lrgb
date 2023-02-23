from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_nu(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument
    cfg.nu = 1
    cfg.nu_v2 = False # determines whether to use (k-1)//nu as delay (True), or max(k, 0) (False)
    cfg.use_agg_weights = False # determines whether to use convex combination of weights for k-hop aggregation or not


register_config('nu', set_cfg_nu)
