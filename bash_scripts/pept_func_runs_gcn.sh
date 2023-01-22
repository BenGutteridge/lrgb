#! /bin/bash
cd ..
pe=$1
file="configs/GCN/peptides-func-GCN+${pe}.yaml"
# dir="datasets"
dir="/data/beng/datasets"
train_mode=my_custom
stage_type=my_stack
# stage_type=stack_residual

SLURM_ARRAY_TASK_ID=$2

layers=(5 7 9 11 13 15 17 19 21 23)
dims=(300 260 230 205 190 175 170 160 150 142)

python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner ${dims[$SLURM_ARRAY_TASK_ID]} tensorboard_each_run False train.mode my_custom gnn.stage_type $stage_type