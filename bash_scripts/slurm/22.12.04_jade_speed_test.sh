#!/bin/bash
cd ../..
DIR_NAME="jade_speed_test"
python bash_scripts/progress_bar.py
runs=(
configs/rbar-GCN/rbar=inf/peptides-func-DelayGCN_L=05_d=175_rbar=inf.yaml
)
batch_sizes=(
  128
)
hidden_dims=(
  64
)
num_layers=(
  6
)

for L in "${num_layers[@]}" ; do
  for d in "${hidden_dims[@]}" ; do
    for bs in "${batch_sizes[@]}" ; do
      DIR="results/$DIR_NAME\_bs=$bs\_d=$d\_L=$L"
      mkdir -p DIR
      for run in "${runs[@]}" ; do
        echo "RUN: $DIR"
        # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
        python3 main.py --cfg "$run" --repeat 1 device cuda dataset.dir datasets out_dir "$DIR" optim.max_epoch 300 train.batch_size "$bs" gnn.dim_inner "$d" gnn.layers_mp "$L"
        python3 bash_scripts/progress_bar.py "$run"
      done
    done
  done
done