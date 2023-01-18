#!/bin/bash
cd ..

pe=$1
file="configs/rbar-GCN/peptides-struct-DelayGCN+${pe}.yaml"
rbar=$2
L=$3
d=$4
dir=$5

python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" rbar "$rbar" gnn.layers_mp "$L" optim.max_epoch 300 gnn.dim_inner "$d" tensorboard_each_run True