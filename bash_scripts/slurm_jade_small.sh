#! /bin/bash
#SBATCH --job-name=F.d=64r1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=30:00:00
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
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pe=none
task='func'
# task='struct'
rbar=1
# file="configs/GCN/peptides-${task}-GCN+${pe}.yaml"
# file="configs/GCN/peptides-${task}-ResGCN+${pe}.yaml"
file="configs/rbar-GCN/peptides-${task}-DelayGCN+${pe}.yaml"

dir=datasets
d=64
L=$SLURM_ARRAY_TASK_ID
rbar=1
# rbar=$(($SLURM_ARRAY_TASK_ID/2))
echo "r*=$rbar"
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" rbar $rbar gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $d tensorboard_each_run False train.mode my_custom