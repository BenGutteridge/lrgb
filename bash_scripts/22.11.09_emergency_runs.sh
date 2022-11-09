#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
configs/rbar-GCN/peptides-struct-DelayGCN_L=11_rbar=inf.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=13_rbar=inf.yaml
configs/alphaGCN/peptides-struct-alphaGCN_L=09_d=075.yaml
configs/alphaGCN/peptides-struct-alphaGCN_L=11_d=060.yaml

configs/rbar-GCN/peptides-func-DelayGCN_L=09_d=100_rbar=06.yaml
configs/rbar-GCN/peptides-func-DelayGCN_L=09_d=100_rbar=06.yaml
configs/rbar-GCN/peptides-func-DelayGCN_L=13_d=070_rbar=06.yaml


)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done