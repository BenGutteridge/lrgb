#! /bin/bash
#SBATCH --job-name=PCCC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH --partition=big
# must be on htc, only one w/ GPUs
# set number of GPUs
#SBATCH --gres=gpu:4

cd ..
module load cuda/10.2
module load python/anaconda3
source $condaDotFile
conda activate lrgb2
nvcc --version
python3.9 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

pe=none
task=struct
# file="configs/GCN/peptides-${task}-GCN+${pe}.yaml"
# file="configs/GCN/peptides-${task}-ResGCN+${pe}.yaml"
# file="configs/rbar-GCN/peptides-${task}-DelayGCN+${pe}.yaml"

# file='configs/GCN/vocsuperpixels-GCN.yaml'
# file='configs/DelayGCN/vocsuperpixels-DelayGCN.yaml'
# file='configs/DelayGCN/vocsuperpixels-DelayGCN+LapPE.yaml'

# file='configs/GCN/pcqm-contact-GCN+none.yaml'
# file='configs/GCN/pcqm-contact-GCN+RWSE.yaml'
file='configs/DelayGCN/pcqm-contact-DelayGCN+none.yaml'
# file='configs/DelayGCN/pcqm-contact-DelayGCN+RWSE.yaml'
# file='configs/DelayGCN/pcqm-contact-DelayGCN+LapPE.yaml'

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

# layer=gcnconv
layer=my_gcnconv
# layer=share_drewgatedgcnconv
# layer=drewgatedgcnconv
# layer=gatedgcnconv

seed=$SLURM_ARRAY_TASK_ID
dir=datasets
out_dir=results/pc_cc/seed=$seed
L=20
nu=-1
# rho=$SLURM_ARRAY_TASK_ID
rho=0
rho_max=1000000
jk=none
k_max=1000000 # default 1e6
ckpt_period=10
edge_encoder=False
epochs=80
use_CC=True

slic=10

# gnn=drew_gated_gnn
# gnn=alpha_gated_gnn
# gnn=my_custom_gnn
gnn=gnn
# gnn=custom_gnn

python3.9 main.py --cfg "$file" --repeat 1 seed $seed agg_weights.convex_combo $use_CC dataset.slic_compactness $slic dataset.edge_encoder $edge_encoder model.type $gnn k_max $k_max jk_mode $jk fixed_params.N 500_000 rho $rho rho_max $rho_max train.auto_resume True train.ckpt_period $ckpt_period gnn.layer_type $layer out_dir $out_dir device cuda dataset.dir "$dir" nu $nu gnn.layers_mp $L optim.max_epoch $epochs tensorboard_each_run True train.mode my_custom

# FOR NO BN
# python3.9 main.py --cfg "$file" --repeat 3 gnn.layer_type $layer gnn.batchnorm False gnn.l2norm False out_dir $out_dir device cuda dataset.dir "$dir" nu $nu gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $dim tensorboard_each_run True train.mode my_custom

# FOR STANDARD BASELINES
# python3.9 main.py --cfg "$file" --repeat 3 out_dir $out_dir device cuda dataset.dir "$dir" optim.max_epoch 300 tensorboard_each_run True