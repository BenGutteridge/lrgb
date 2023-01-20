#! /bin/bash
#SBATCH --job-name=S.d=64r1
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
pe=none
# task='func'
task='struct'
# file="configs/GCN/peptides-${task}-GCN+${pe}.yaml"
# file="configs/GCN/peptides-${task}-ResGCN+${pe}.yaml"
file="configs/rbar-GCN/peptides-${task}-DelayGCN+${pe}.yaml"

dir=datasets
# d=64
dims=(  175 130 105 85 72 64 55 50 45 42)
layers=(5   7   9   11 13 15 17 19 21 23)
rbar=-1
rbars=(2 3 4 5 6 7 8 9 10 11)
# rbar=$(($SLURM_ARRAY_TASK_ID/2))
echo "r*=$rbar"

# # fixed d
# python3.9 main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" rbar $rbar gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $d tensorboard_each_run False train.mode my_custom
# # fixed params
python3.9 main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" rbar $rbar gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner ${dims[$SLURM_ARRAY_TASK_ID]} tensorboard_each_run False train.mode my_custom
# rbar=L/2
python3.9 main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" ${rbars[$SLURM_ARRAY_TASK_ID]}rbar $rbar gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner ${dims[$SLURM_ARRAY_TASK_ID]} tensorboard_each_run False train.mode my_custom