#! /bin/bash
#SBATCH --job-name=QnGINCClap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH --partition=short
# must be on htc, only one w/ GPUs
#SBATCH --clusters=htc
# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --account=engs-oxnsg

cd $DATA/repos/lrgb
# cd $DATA/repos/clean_lrgb/lrgb

module purge
module load Anaconda3
module load CUDAcore/11.1.1
source activate $DATA/longrange
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"


# L=$1
# nu=$2
# pe=$3
# task=$4

# file="configs/GCN/peptides-${task}-GCN+${pe}.yaml"
# file="configs/GCN/peptides-${task}-ResGCN+${pe}.yaml"
# file="configs/rbar-GCN/peptides-${task}-DelayGCN+${pe}.yaml"

# file='configs/GCN/vocsuperpixels-GCN.yaml'
# file='configs/DelayGCN/vocsuperpixels-DelayGCN.yaml'
# file='configs/DelayGCN/vocsuperpixels-DelayGCN+LapPE.yaml'

# file='configs/GCN/pcqm-contact-GCN+none.yaml'
# file='configs/GCN/pcqm-contact-GCN+RWSE.yaml'
# file='configs/DelayGCN/pcqm-contact-DelayGCN+none.yaml'
# file='configs/DelayGCN/pcqm-contact-DelayGCN+RWSE.yaml'
file='configs/DelayGCN/pcqm-contact-DelayGCN+LapPE.yaml'

# file='configs/DelayGCN/cocosuperpixels-DelayGCN.yaml'
# file='configs/DelayGCN/cocosuperpixels-DelayGCN+LapPE.yaml'

# file='configs/GCN/pcqm-contact-GCN.yaml'
# file='configs/SAN/pcqm-contact-SAN.yaml'
# file='configs/GatedGCN/pcqm-contact-GatedGCN.yaml'

# # DRewGated, VOC 
# file='configs/GatedGCN/vocsuperpixels-GatedGCN.yaml'
# file='configs/GatedGCN/vocsuperpixels-GatedGCN+LapPE.yaml'
# file='configs/DRewGatedGCN/vocsuperpixels-DRewGatedGCN.yaml'
# file='configs/DRewGatedGCN/vocsuperpixels-DRewGatedGCN+LapPE.yaml'

# Just for runing pure SAN
# python main.py --cfg configs/SAN/vocsuperpixels-SAN.yaml --repeat 3 tensorboard_each_run True dataset.dir ../../lrgb/datasets wandb.use False train.ckpt_period 5 device cuda train.auto_resume True out_dir results/retry

# file="configs/DRewGatedGCN/peptides-${task}-DRewGatedGCN+${pe}.yaml"

# layer=gcnconv
layer=my_gcnconv
# layer=share_drewgatedgcnconv
# layer=drewgatedgcnconv
# layer=gatedgcnconv_noedge

seed=$SLURM_ARRAY_TASK_ID
dir=datasets
out_dir="results/seed=$seed"
# rho=$SLURM_ARRAY_TASK_ID
rho=0
rho_max=1000000
jk=none
k_max=1000000 # default 1e6
ckpt_period=10
edge_encoder=False
epochs=90
use_CC=True
digl_alpha=0.15

# avg_deg=$SLURM_ARRAY_TASK_ID
# tf="digl=$avg_deg"
tf=none

# gnn=drew_gated_gnn
# gnn=alpha_gated_gnn
# gnn=my_custom_gnn
gnn=drew_gin
# gnn=gnn
# gnn=custom_gnn
bn=True

nu=-1
L=10

python main.py --cfg "$file" --repeat 1 out_dir $out_dir digl.alpha $digl_alpha gnn.batchnorm $bn dataset.transform $tf seed $seed agg_weights.convex_combo $use_CC dataset.edge_encoder $edge_encoder model.type $gnn k_max $k_max jk_mode $jk fixed_params.N 500_000 rho $rho rho_max $rho_max train.auto_resume True train.ckpt_period $ckpt_period device cuda dataset.dir "$dir" nu $nu gnn.layers_mp $L optim.max_epoch $epochs tensorboard_each_run True train.mode custom

# FOR NO BN
# python main.py --cfg "$file" --repeat 3 gnn.layer_type $layer gnn.batchnorm False gnn.l2norm False out_dir $out_dir device cuda dataset.dir "$dir" nu $nu gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $dim tensorboard_each_run True train.mode my_custom

# FOR STANDARD BASELINES
# python main.py --cfg "$file" --repeat 3 out_dir $out_dir device cuda dataset.dir "$dir" optim.max_epoch 300 tensorboard_each_run True

### For ASC

# cfg=configs/rbar-GCN/peptides-struct-DelayGCN+none.yaml
# A=True
# S=delay_share_gnn
# C=True
# out_dir=C
# nu=-1
# L=5
# python main.py --cfg $cfg --repeat 3 gnn.stage_type $S agg_weights.use $A agg_weights.convex_combo $C fixed_params.N 500_000 gnn.layer_type my_gcnconv out_dir "results/$out_dir" device cuda dataset.dir datasets nu $nu gnn.layers_mp $L optim.max_epoch 300 tensorboard_each_run True train.mode my_custom train.auto_resume True train.ckpt_period 10