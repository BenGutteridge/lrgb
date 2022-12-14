#!/bin/bash
cd ..

model_type="R*-SPN"

rbar=$2

DIR_NAME="22.12.13_QM9_${model_type}_r*=$rbar"
# python bash_scripts/progress_bar.py
runs=(
configs/rbar-GIN/QM9-r*GIN.yaml
)
batch_sizes=(
  128
  # 1024
)
hidden_dims=(
  128
)
num_layers=$1

echo "model type = ${model_type}_L=$num_layers, rbar=$rbar"

for L in "${num_layers[@]}" ; do
  for d in "${hidden_dims[@]}" ; do
    for bs in "${batch_sizes[@]}" ; do
      DIR="results/$DIR_NAME"
      mkdir -p DIR
      for run in "${runs[@]}" ; do
        # python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets train.batch_size 16
        python main.py --cfg "$run" --repeat 1 use_edge_labels True rbar "$rbar" model.type "${model_type}" device cuda dataset.dir /datasets out_dir "$DIR" optim.max_epoch 200 train.batch_size "$bs" gnn.dim_inner "$d" gnn.layers_mp "$L"
        # python bash_scripts/progress_bar.py "$run"
      done
    done
  done
done