import datetime
import os
import torch
import logging

import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optimizer import create_optimizer, \
    create_scheduler, OptimizerConfig, SchedulerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
from graphgps.ben_utils import custom_set_out_dir
from param_calcs import set_d_fixed_params

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return SchedulerConfig(scheduler=cfg.optim.scheduler,
                           steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
                           max_epoch=cfg.optim.max_epoch)


# def custom_set_out_dir(cfg, cfg_fname, name_tag):
#     """Set custom main output directory path to cfg.
#     Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

#     Args:
#         cfg (CfgNode): Configuration node
#         cfg_fname (string): Filename for the yaml format configuration file
#         name_tag (string): Additional name tag to identify this execution of the
#             configuration file, specified in :obj:`cfg.name_tag`
#     """
#     run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
#     run_name += f"-{name_tag}" if name_tag else ""
#     cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)

def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices

model = 'DelayGCN'
# model = 'GCN'
# model='SAN'
# model='alphaGCN'

# # VOC superpixels
# argpath = "/Users/beng/Documents/lrgb/configs/%s/vocsuperpixels-%s.yaml" % (model, model)
# Coco-superpixels
# argpath = '/Users/beng/Documents/lrgb/configs/%s/cocosuperpixels-%s.yaml' % (model, model)
# Peptides-func
argpath = '/Users/beng/Documents/lrgb/configs/%s/peptides-func-%s.yaml' % (model, model)
# Peptides-struct
# argpath = "/Users/beng/Documents/lrgb/configs/%s/peptides-struct-%s.yaml" % (model, model)
# # PCQM-Contact
# argpath = "configs/%s/pcqm-contact-%s.yaml" % (model, model)

# argpath = 'configs/Transformer/peptides-func-Transformer+LapPE.yaml'

# argpath = 'configs/alphaGCN/peptides-func-alphaGCN_L=21_d=032.yaml'

# argpath = 'configs/alphaGCN/QM9-alphaGCN_L=13.yaml'
# argpath = 'configs/DelayGCN/500k_stretched/peptides-func-DelayGCN_L=07_d=130.yaml'
# argpath = 'configs/rbar-GCN/QM9-rGCN.yaml'
# argpath = 'configs/rbar-GIN/QM9-r*GIN.yaml'
# argpath = 'configs/GINE/QM9-GINE.yaml'

# argpath = 'configs/GatedGCN/peptides-struct-GatedGCN+RWSE.yaml'

# argpath = 'configs/DelayGCN/peptides-func-DelayGCN+RWSE.yaml'
# argpath = 'configs/rbar-GCN/peptides-struct-DelayGCN+LapPE.yaml'
# argpath = 'configs/rbar-GCN/peptides-struct-DelayGCN.yaml'

# argpath='configs/GCN/pcqm-contact-GCN.yaml'

# argpath = 'configs/GatedGCN/vocsuperpixels-GatedGCN.yaml'

# argpath = 'configs/DRewGatedGCN/peptides-func-DRewGatedGCN.yaml'
# argpath = 'configs/GatedGCN/peptides-func-GatedGCN.yaml'
# argpath = 'configs/DRewGatedGCN/voc-DRewGatedGCN.yaml'

# argpath = 'configs/SAN/vocsuperpixels-SAN.yaml'
# argpath = 'configs/DRewGatedGCN/vocsuperpixels-DRewGatedGCN.yaml'

repeat = 1
import argparse
def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    extra_args = [
        'out_dir results',
        'dataset.dir datasets',
        'train.mode my_custom',
        # 'optim.max_epoch 2',
        # 'model.type drew_gated_gnn',
        # 'model.type alpha_gated_gnn',
        # 'model.type my_custom_gnn',
        # 'gnn.stage_type my_stack',
        'gnn.stage_type delay_gnn_alt_postlayer',
        # 'gnn.stage_type alpha_gnn',
        # 'gnn.stage_type delay_share_gnn',
        'gnn.layer_type my_gcnconv',
        # 'gnn.layer_type drewgatedgcnconv',        
        # 'gnn.layer_type share_drewgatedgcnconv',
        # 'nu 2',
        # 'gnn.dim_inner 4',
        'gnn.layers_mp 8',
        # 'dataset.edge_encoder False',
        # 'use_edge_labels True',
        # 'train.batch_size 128',
        # 'posenc_RWSE.kernel.times_func range(1,17)', # 16 steps for RWPE
        # 'posenc_LapPE.dim_pe 8', # 8 steps for RWSE
        # 'spn.K 10',
        # 'posenc_RWSE.dim_pe 8',
        # 'seed 5',
        # 'train.auto_resume True',
        # 'gnn.l2norm False',
        # 'gnn.batchnorm False',

        # 'fixed_params.N 0',
        'fixed_params.N 500_000',
        # 'agg_weights.use True',
        # 'agg_weights.convex_combo True'
        # 'rho 3',
        # 'rho_max 10',
        # 'k_max 5',

        # 'jk_mode cat', # none, [rho_][max, cat] 
        # 'train.ckpt_period 1',
        'alt_postlayer.bn True',
        'alt_postlayer.act True',
        ]

    # argpath='results/pept-func_delay_gnn_kmax=03_rho=03_rho_max=10_d=080_L=14/config.yaml'

# TODO: SORT
#  {'pept': 'graph', 
#  'voc': 'inductive_node'}
# {
#     'none': cfg.gnn.head,
#     'max': 'jk_max',
#     'cat': 'jk_cat_graph',
#     'rho_max': 'rho_jk_max_graph'
#     'rho_cat': 'rho_jk_cat_graph'
#     }

# it's just '%s_%s' % (cfg.jk_head, )

    # debug_args = [
    #     'gnn.layer_type my_gcnconv',
    #     'nu -1',
    #     'gnn.layers_mp 2', # og 8
    #     'optim.max_epoch 300',
    #     'train.mode my_custom'
    #     # 'fixed_params.N 500_000',

    # 'out_dir results/new_rho',
    # 'gnn.layers_mp 5',
    # 'nu 1',
    # 'rho 5',
    # ]
    # if debug_args: extra_args += debug_args

    extra_args = ' '.join(extra_args)
    return parser.parse_args("--cfg {} --repeat {} {}".format(argpath, repeat, extra_args).split())


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    cfg.run_dir = cfg.out_dir # prevents error when loading config.yaml file generated from run
    load_cfg(cfg, args)
    set_d_fixed_params(cfg) # for setting d with fixed param budget
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.train.finetune:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        if cfg.train.finetune:
            model = init_model_from_pretrained(model, cfg.train.finetune,
                                               cfg.train.freeze_pretrained)
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: {}'.format(cfg.params))
        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, '{}_done'.format(args.cfg_file))
    logging.info(f"[*] All done: {datetime.datetime.now()}")
