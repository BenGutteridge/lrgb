from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_fixed_params(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''
    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.fixed_params = CN()
    cfg.fixed_params.N = 0 # a default, ignored if not >0
    cfg.fixed_params.model_task = 'none' # a default, ignored if not >0 ['voc_gcn', 'pept_drew', etc]

register_config('fixed_params', set_cfg_fixed_params)