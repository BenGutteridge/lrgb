#!/bin/bash
cd ..
BATCH="qm9_5_r=inf"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # stretched QM9 for everything
  # configs/rbar-GCN/rbar=inf/QM9-rGCN_L=05_r=inf.yaml
  # configs/rbar-GCN/rbar=inf/QM9-rGCN_L=07_r=inf.yaml
  # configs/rbar-GCN/rbar=inf/QM9-rGCN_L=09_r=inf.yaml
  # configs/rbar-GCN/rbar=inf/QM9-rGCN_L=11_r=inf.yaml
  # configs/rbar-GCN/rbar=inf/QM9-rGCN_L=13_r=inf.yaml
  # configs/rbar-GCN/rbar=inf/QM9-rGCN_L=15_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=17_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=19_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=21_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=23_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=25_r=inf.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done