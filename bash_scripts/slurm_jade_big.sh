#! /bin/bash
#SBATCH --job-name=pcnuinf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH --partition=big
# must be on htc, only one w/ GPUs
# set number of GPUs
#SBATCH --gres=gpu:1

cd ..
module load cuda/10.2
module load python/anaconda3
source $condaDotFile
conda activate lrgb2
nvcc --version
python3.9 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# pe=none
# task='func'
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

# layer=gcnconv
layer=my_gcnconv

dir=datasets
out_dir="results"
L=$SLURM_ARRAY_TASK_ID
# L=23
nu=-1
# rho=$SLURM_ARRAY_TASK_ID
rho=0
jk=none
k_max=1000000 # default 1e6
ckpt_period=10


python3.9 main.py --cfg "$file" --repeat 3 k_max $k_max jk_mode $jk fixed_params.N 500_000 rho $rho train.auto_resume True train.ckpt_period $ckpt_period gnn.layer_type $layer out_dir $out_dir device cuda dataset.dir "$dir" nu $nu gnn.layers_mp $L optim.max_epoch 300 tensorboard_each_run True train.mode my_custom

# FOR NO BN
# python3.9 main.py --cfg "$file" --repeat 3 gnn.layer_type $layer gnn.batchnorm False gnn.l2norm False out_dir $out_dir device cuda dataset.dir "$dir" nu $nu gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $dim tensorboard_each_run True train.mode my_custom

# FOR STANDARD BASELINES
# python3.9 main.py --cfg "$file" --repeat 3 out_dir $out_dir device cuda dataset.dir "$dir" optim.max_epoch 300 tensorboard_each_run True