from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_alt_postlayer(cfg):
    r'''
    For use with delay_gnn_BN, to set whether you want BN/activation for each 
    GCN conv used in aggregation, or only on the final summation
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.alt_postlayer = CN()

    # example argument
    cfg.alt_postlayer.bn = False

    # example argument group
    cfg.alt_postlayer.act = False


register_config('alt_postlayer', set_cfg_alt_postlayer)
