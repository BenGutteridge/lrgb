
#!/bin/bash
cd ..
dir="/data/beng/datasets"
# dir='datasets'
# fixed d r*=1
pe=none
file="configs/rbar-GCN/peptides-func-DelayGCN+$pe.yaml"

python main.py --cfg "$file" --repeat 3 rbar 1 gnn.layers_mp 23 optim.max_epoch 300 gnn.dim_inner 96 tensorboard_each_run False train.mode my_custom device cuda dataset.dir $dir