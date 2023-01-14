#! /bin/bash
t=$1
echo "Waiting to start"
sleep ${t}h
echo "Ready to run!"
cd ..
pe=LapPE
file="configs/rbar-GCN/peptides-struct-DelayGCN+${pe}.yaml"
# dir="datasets"
dir="/data/beng/datasets"

SLURM_ARRAY_TASK_ID=$2

layers=(5 6 7 8 9 10 11 13 15 17 19 21 23)
dims=(175 150 130 114 105 92 85 72 64 55 50 42)
rbars=(-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1)

python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" rbar ${rbars[$SLURM_ARRAY_TASK_ID]} gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner ${dims[$SLURM_ARRAY_TASK_ID]} tensorboard_each_run False train.mode my_custom