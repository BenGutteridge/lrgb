
#!/bin/bash
cd ..
dir="/data/beng/datasets"
# dir='datasets'
# fixed d r*=1
file="configs/rbar-GCN/peptides-struct-DelayGCN+LapPE.yaml"

python main.py --cfg "$file" --repeat 3 rbar 1 gnn.layers_mp 11 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run False train.mode my_custom device cuda dataset.dir $dir
python main.py --cfg "$file" --repeat 3 rbar -1 gnn.layers_mp 11 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run False train.mode my_custom device cuda dataset.dir $dir
python main.py --cfg "$file" --repeat 3 rbar 5 gnn.layers_mp 11 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run False train.mode my_custom device cuda dataset.dir $dir