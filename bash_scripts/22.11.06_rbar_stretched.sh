#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
# correct rbar pept-func
configs/DelayGCN_rbar/peptides-func-DelayGCN_L=05_d=175_rbar=inf.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_L=07_d=130_rbar=inf.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_L=09_d=100_rbar=inf.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_L=11_d=085_rbar=inf.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_L=05_d=175_rbar=01.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_L=07_d=130_rbar=01.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_L=09_d=100_rbar=01.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_L=11_d=085_rbar=01.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 2 device cuda dataset.dir /data/beng/datasets optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done