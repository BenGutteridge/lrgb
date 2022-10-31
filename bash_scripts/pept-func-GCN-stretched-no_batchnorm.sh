#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
configs/GCN/500k_stretched/peptides-func-GCN_L=07_d=250.yaml
configs/GCN/500k_stretched/peptides-func-GCN_L=09_d=225.yaml
configs/GCN/500k_stretched/peptides-func-GCN_L=11_d=200.yaml
configs/GCN/500k_stretched/peptides-func-GCN_L=13_d=190.yaml
configs/GCN/500k_stretched/peptides-func-GCN_L=15_d=175.yaml
)

for run in "${runs[@]}" ; do
  python main.py --cfg "$run" device cuda dataset.dir /data/beng/datasets gnn.batchnorm False optim.max_epoch 200
  python bash_scripts/progress_bar.py "$run"
done