#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
  configs/rbar-GCN/ZINC-rGCN_L=07_d=130_r=inf.yaml
  configs/rbar-GCN/ZINC-rGCN_L=05_d=175_r=inf.yaml
  configs/rbar-GCN/ZINC-rGCN_L=02_d=400_r=inf.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done
