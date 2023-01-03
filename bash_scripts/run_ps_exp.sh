#!/bin/bash
cd ..

pe=$1
file="configs/rbar-GCN/peptides-struct-DelayGCN+${pe}.yaml"
rbar=$2
L=$3

python main.py --cfg "$file" device cuda dataset.dir data/beng/datasets rbar "$rbar" gnn.layers_mp "$L" optim.max_epoch 300 gnn.dim_inner 64