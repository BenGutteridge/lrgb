#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
configs/alphaGCN/peptides-func-alphaGCN_L=17_d=042.yaml
configs/alphaGCN/peptides-func-alphaGCN_L=19_d=038.yaml
configs/alphaGCN/peptides-func-alphaGCN_L=21_d=034.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=17_rbar=01.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=17_rbar=10.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=17_rbar=15.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=17_rbar=inf.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=19_rbar=01.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=19_rbar=10.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=19_rbar=15.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=19_rbar=inf.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=21_rbar=01.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=21_rbar=10.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=21_rbar=15.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=21_rbar=20.yaml
configs/rbar-GCN/peptides-struct-DelayGCN_L=21_rbar=inf.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done