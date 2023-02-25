from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_dirichlet(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument
    cfg.dirichlet = CN()
    cfg.dirichlet.use = False

register_config('dirichlet', set_cfg_dirichlet)
