#! /bin/bash
t=$1
echo "Waiting to start"
sleep ${t}h
echo "Ready to run!"
cd ..
pe=none
file="configs/GCN/peptides-struct-GCN+${pe}.yaml"
# dir="datasets"
dir="/data/beng/datasets"
train_mode=my_custom


SLURM_ARRAY_TASK_ID=$2

layers=(5 7 9 11 13 15 17 19 21 23)
dims=(300 260 230 205 190 175 170 160 150 142)

stage_type=custom_stack
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner ${dims[$SLURM_ARRAY_TASK_ID]} tensorboard_each_run False train.mode my_custom gnn.stage_type $stage_type
stage_type=stack_residual
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner ${dims[$SLURM_ARRAY_TASK_ID]} tensorboard_each_run False train.mode my_custom gnn.stage_type $stage_type
