#! /bin/bash
#SBATCH --job-name=NOBNr1,inf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=48:00:00
#SBATCH --partition=small
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
pe=none
task='func'
# file="configs/GCN/peptides-${task}-GCN+${pe}.yaml"
# file="configs/GCN/peptides-${task}-ResGCN+${pe}.yaml"
file="configs/rbar-GCN/peptides-${task}-DelayGCN+${pe}.yaml"

dir=datasets
out_dir="results/no_batchnorm"
dim=64
L=$SLURM_ARRAY_TASK_ID

rbar=1
python3.9 main.py --cfg "$file" --repeat 3 gnn.batchnorm False gnn.l2norm False out_dir $out_dir device cuda dataset.dir "$dir" rbar $rbar gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $dim tensorboard_each_run False train.mode my_custom
rbar=-1
python3.9 main.py --cfg "$file" --repeat 3 gnn.batchnorm False gnn.l2norm False out_dir $out_dir device cuda dataset.dir "$dir" rbar $rbar gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $dim tensorboard_each_run False train.mode my_custom
