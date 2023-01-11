#!/bin/bash
cd ..
file="configs/GCN/peptides-struct-GCN+LapPE.yaml"
dir="datasets"
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir $dir gnn.layers_mp 5 optim.max_epoch 300 gnn.dim_inner 300 tensorboard_each_run True gnn.stage_type stack_residual train.mode my_custom
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir $dir gnn.layers_mp 9 optim.max_epoch 300 gnn.dim_inner 230 tensorboard_each_run True gnn.stage_type stack_residual train.mode my_custom
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir $dir gnn.layers_mp 13 optim.max_epoch 300 gnn.dim_inner 190 tensorboard_each_run True gnn.stage_type stack_residual train.mode my_custom
