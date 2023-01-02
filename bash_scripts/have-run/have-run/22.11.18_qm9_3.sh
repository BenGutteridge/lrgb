#!/bin/bash
cd ..
BATCH="qm9_3_r=1"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # alphaGCN 5 layers multi runs to see if its consistent
  # stretched QM9 for everything
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=05_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=07_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=09_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=11_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=13_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=15_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=17_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=19_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=21_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=23_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=25_r=01.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done