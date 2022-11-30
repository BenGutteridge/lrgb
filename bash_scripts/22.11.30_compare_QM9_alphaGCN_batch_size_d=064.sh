#!/bin/bash
cd ..
DIR_NAME="22.11.30_compare_QM9_alphaGCN_batch_size_d=064"
python bash_scripts/progress_bar.py
runs=(
# something to compare batch size for alphaGCN and other stuff on QM9 -- what is a reasonable batch size
configs/alphaGCN/QM9-alphaGCN_L=05.yaml
configs/alphaGCN/QM9-alphaGCN_L=09.yaml
configs/alphaGCN/QM9-alphaGCN_L=15.yaml
)
batch_sizes=(
  128
  256
  512
  # 1024
)

for bs in "${batch_sizes[@]}" ; do
  mkdir -p "results/$DIR_NAME\_=$bs"
  DIR="results/$DIR_NAME\_bs=$bs"
  for run in "${runs[@]}" ; do
    # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
    python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$DIR" optim.max_epoch 300 train.batch_size "$bs" gnn.dim_inner 64
    python bash_scripts/progress_bar.py "$run"
  done
done