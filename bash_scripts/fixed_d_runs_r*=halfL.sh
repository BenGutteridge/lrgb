
#!/bin/bash
cd ..
dir="/data/beng/datasets"
# fixed d r*=L/2
file="configs/rbar-GCN/peptides-struct-DelayGCN+LapPE.yaml"
python main.py --cfg "$file" --repeat 3 rbar 2 gnn.layers_mp 5 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run True train.mode my_custom device cuda dataset.dir $dir
python main.py --cfg "$file" --repeat 3 rbar 4 gnn.layers_mp 9 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run True train.mode my_custom device cuda dataset.dir $dir
python main.py --cfg "$file" --repeat 3 rbar 6 gnn.layers_mp 13 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run True train.mode my_custom device cuda dataset.dir $dir
python main.py --cfg "$file" --repeat 3 rbar 8 gnn.layers_mp 17 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run True train.mode my_custom device cuda dataset.dir $dir

