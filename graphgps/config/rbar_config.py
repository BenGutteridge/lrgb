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
    
    cfg.agg_weights = CN()
    cfg.agg_weights.use = False # determines whether to use learned weights for k-hop aggregations
    cfg.agg_weights.convex_combo = False # determines whether to use convex combination of weights -- weights are equal if 'use'=False


register_config('nu', set_cfg_nu)
