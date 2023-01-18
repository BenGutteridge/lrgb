#!/bin/bash
cd ..
pe=none
task=struct
# task=struct
file="configs/GCN/peptides-${task}-GCN+${pe}.yaml"
# file="configs/GCN/peptides-${task}-ResGCN+${pe}.yaml"
# file="configs/rbar-GCN/peptides-${task}-DelayGCN+${pe}.yaml"

# dir=datasets
dir="/data/beng/datasets"
d=64
L=$1
rbar=1
# echo "r*=$rbar"
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" rbar $rbar gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $d tensorboard_each_run False train.mode my_custom