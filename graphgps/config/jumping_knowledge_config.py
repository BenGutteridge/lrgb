from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_jumping_knowledge(cfg):
    r'''
    Whether to use jumping knowledge or not, and if so, what method of aggregation (max, cat),
    and whether to do pure JK or only agreggate after rho layers
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument
    cfg.jk_mode = 'none' # ['none', 'rho_max', 'rho_cat', 'cat', 'max']

register_config('jumping_knowledge', set_cfg_jumping_knowledge)
