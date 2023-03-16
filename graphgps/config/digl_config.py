from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_digl(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument group
    cfg.digl = CN()

    # then argument can be specified within the group
    cfg.digl.alpha = 0.15


register_config('digl', set_cfg_digl)
