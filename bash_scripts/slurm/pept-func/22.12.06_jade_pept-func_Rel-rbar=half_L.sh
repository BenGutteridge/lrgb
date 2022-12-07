#!/bin/bash
cd ../../..
DIR_NAME="22.12.06_jade_pept-func_rbar=half_L"
runs=(
configs/rbar-GCN/pept-func-rGCN.yaml
)
batch_sizes=(
  128
  # 256
)
hidden_dims=(
  64
)

L="$1"

rbar="$((L/2))"
echo "rbar: $rbar"

for d in "${hidden_dims[@]}" ; do
  for bs in "${batch_sizes[@]}" ; do
    DIR="results/$DIR_NAME\_bs=$bs\_d=$d\_L=$L"
    mkdir -p DIR
    for run in "${runs[@]}" ; do
      echo "RUN: $DIR"
      # python main.py --cfg "$run" --repeat 3 device cuda use_edge_labels True dataset.dir /data/beng/datasets train.batch_size 16
      python3 main.py --cfg "$run" --repeat 3 rbar "$rbar" device cuda use_edge_labels True gnn.stage_type rel_delay_gnn dataset.dir datasets out_dir "$DIR" optim.max_epoch 300 train.batch_size "$bs" gnn.dim_inner "$d" gnn.layers_mp "$1"
      mv "$DIR" "results/completed/"
    done
  done
done
