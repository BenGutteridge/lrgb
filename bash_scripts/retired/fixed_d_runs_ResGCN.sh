
#!/bin/bash
cd ..
file="configs/GCN/peptides-struct-GCN+LapPE.yaml"
dir="/data/beng/datasets"

# fixed d ResGCN
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir $dir gnn.layers_mp 5 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run True gnn.stage_type stack_residual train.mode my_custom
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir $dir gnn.layers_mp 9 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run True gnn.stage_type stack_residual train.mode my_custom
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir $dir gnn.layers_mp 13 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run True gnn.stage_type stack_residual train.mode my_custom
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir $dir gnn.layers_mp 7 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run True gnn.stage_type stack_residual train.mode my_custom
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir $dir gnn.layers_mp 11 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run True gnn.stage_type stack_residual train.mode my_custom
python main.py --cfg "$file" --repeat 3 device cuda dataset.dir $dir gnn.layers_mp 15 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run True gnn.stage_type stack_residual train.mode my_custom