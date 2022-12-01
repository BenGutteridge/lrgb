#!/bin/bash
cd ..
DIR_NAME="22.11.30_compare_QM9_alphaGCN_batch_size_d=064"
python bash_scripts/progress_bar.py
runs=(
# something to compare batch size for alphaGCN and other stuff on QM9 -- what is a reasonable batch size
configs/rbar-GCN/rbar=01/QM9-rGCN_L=05_r=01.yaml
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

for d in "${hidden_dims[@]}" ; do
  for bs in "${batch_sizes[@]}" ; do
    for L in "${num_layers[@]}" ; do
      DIR="results/$DIR_NAME\_bs=$bs\_d=$d\_L=$L"
      mkdir -p DIR
      for run in "${runs[@]}" ; do
        rbar=$((num_layers/2))
        echo "rbar = $rbar"
        # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
        python main.py --cfg "$run" --repeat 1 rbar "$rbar" device cuda dataset.dir /data/beng/datasets out_dir "$DIR" optim.max_epoch 300 train.batch_size "$bs" gnn.dim_inner "$d" gnn.layers_mp "$L"
        python bash_scripts/progress_bar.py "$run"
      done
    done
  done
done