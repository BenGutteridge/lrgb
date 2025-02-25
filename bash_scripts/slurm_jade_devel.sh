#! /bin/bash
#SBATCH --job-name==func_d=64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:10:00
#SBATCH --partition=devel
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
nu=1
file="configs/GCN/peptides-${task}-GCN+${pe}.yaml"
file="configs/GCN/peptides-${task}-ResGCN+${pe}.yaml"
file="configs/rbar-GCN/peptides-${task}-DelayGCN+${pe}.yaml"

dir=datasets
d=64
L=$SLURM_ARRAY_TASK_ID
nu=1
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" nu $nu gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $d tensorboard_each_run False train.mode my_custom