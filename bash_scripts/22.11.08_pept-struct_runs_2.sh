#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
# r*=1 5:15
# r*=inf 5:15
# r*=? 5:15
configs/rbar-GCN/peptides-struct-DelayGCN_L=07_rbar=01.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=07_rbar=03.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=07_rbar=inf.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=09_rbar=01.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=09_rbar=03.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=09_rbar=inf.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=11_rbar=01.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=11_rbar=03.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=11_rbar=inf.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=13_rbar=01.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=13_rbar=03.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=13_rbar=inf.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=15_rbar=01.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=15_rbar=03.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=15_rbar=inf.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done