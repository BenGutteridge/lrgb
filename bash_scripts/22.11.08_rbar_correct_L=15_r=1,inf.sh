#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
configs/rbar-GCN/peptides-func-DelayGCN_L=15_d=060_rbar=inf.yaml
configs/rbar-GCN/peptides-func-DelayGCN_L=15_d=060_rbar=01.yaml
configs/rbar-GCN/peptides-func-DelayGCN_L=13_d=070_rbar=inf.yaml
configs/rbar-GCN/peptides-func-DelayGCN_L=13_d=070_rbar=01.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda optim.max_epoch 300 dataset.dir datasets
  python bash_scripts/progress_bar.py "$run"
done