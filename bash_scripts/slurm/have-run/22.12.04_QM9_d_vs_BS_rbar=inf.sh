#!/bin/bash
cd ../..
DIR_NAME="22.12.04_QM9_bs_vs_d_rbar=inf"
python bash_scripts/progress_bar.py
runs=(
# something to compare batch size for alphaGCN and other stuff on QM9 -- what is a reasonable batch size
configs/rbar-GCN/rbar=01/QM9-rGCN_L=05_r=inf.yaml
)
batch_sizes=(
  128
  256
  512
  1024
)
hidden_dims=(
  32
  64
  128
)
num_layers=(
  6
  10
  14
)

for L in "${num_layers[@]}" ; do
  for d in "${hidden_dims[@]}" ; do
    for bs in "${batch_sizes[@]}" ; do
      DIR="results/$DIR_NAME\_bs=$bs"
      mkdir -p DIR
      for run in "${runs[@]}" ; do
        # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
        python3 main.py --cfg "$run" --repeat 1 device cuda dataset.dir datasets out_dir "$DIR" optim.max_epoch 300 train.batch_size "$bs" gnn.dim_inner "$d" gnn.layers_mp "$L"
        python3 bash_scripts/progress_bar.py "$run"
      done
    done
  done
done