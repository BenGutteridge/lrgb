#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
configs/DelayGCN_rbar/peptides-func-DelayGCN_rbar_L=07_d=130.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_rbar_L=09_d=100.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_rbar_L=11_d=085.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_rbar_L=13_d=070.yaml
configs/DelayGCN_rbar/peptides-func-DelayGCN_rbar_L=15_d=060.yaml)

for run in "${runs[@]}" ; do
  python main.py --cfg "$run" device cuda dataset.dir /data/beng/datasets False optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done