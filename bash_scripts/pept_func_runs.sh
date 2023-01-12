#! /bin/bash
echo "Waiting to start"
sleep 3h
echo "Ready to run!"
cd ..
pe=none
file="configs/rbar-GCN/pept-func-DelayGCN+${pe}.yaml"
dir="/data/beng/datasets"

SLURM_ARRAY_TASK_ID=$1

layers=(7 9 11 13 15 17)
dims=(130 105 85 72 64 55)
rbars=(1 1 1 1 1 1)

python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" rbar ${rbars[$SLURM_ARRAY_TASK_ID]} gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner ${dims[$SLURM_ARRAY_TASK_ID]} tensorboard_each_run True train.mode my_custom