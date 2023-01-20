#!/bin/bash
cd ..
pe=none
task=func
# task=struct
# file="configs/GCN/peptides-${task}-GCN+${pe}.yaml"
# file="configs/GCN/peptides-${task}-ResGCN+${pe}.yaml"
# file="configs/rbar-GCN/peptides-${task}-DelayGCN+${pe}.yaml"

residual=$1
files=(
  "configs/GCN/peptides-${task}-GCN+${pe}.yaml"
  "configs/GCN/peptides-${task}-ResGCN+${pe}.yaml"
)

# dir=datasets
dir="/data/beng/datasets"
d=64
L=$2

# rbar=$((L/2))
# echo "r*=$rbar"

python main.py --cfg "${files[$residual]}" --repeat 3 gnn.batchnorm False gnn.l2norm False out_dir "results/no_batchnorm" device cuda dataset.dir "$dir" gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $d tensorboard_each_run False train.mode my_custom