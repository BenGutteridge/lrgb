#!/bin/bash
cd ..
BATCH="qm9_stragglers_r=inf_B=1024"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # stretched QM9 for everything
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=19_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=21_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=23_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=25_r=inf.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300 train.batch_size 1024
  python bash_scripts/progress_bar.py "$run"
done