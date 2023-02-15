#! /bin/bash
cd ..
pe=$1
file="configs/rbar-GCN/peptides-func-DelayGCN+${pe}.yaml"
dir="datasets"
# dir="/data/beng/datasets"

SLURM_ARRAY_TASK_ID=$2

layers=(5   7   9   11 13 15 17 19 21 23)
dims=(  175 130 105 85 72 64 55 50 45 42)
# nus=( 2   3   4   5  6  7  8  9 10 11)
nu=$3

# python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" nu ${nus[$SLURM_ARRAY_TASK_ID]} gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner ${dims[$SLURM_ARRAY_TASK_ID]} tensorboard_each_run False train.mode my_custom
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" nu $nu gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner ${dims[$SLURM_ARRAY_TASK_ID]} tensorboard_each_run False train.mode my_custom