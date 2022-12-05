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

    cfg.fixed_params = False
    cfg.fixed_mp_params_num = 450_000

register_config('fixed_params', set_cfg_fixed_params)