#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=( 
configs/DeLiteGCN/500k_stretched/peptides-func-DeLiteGCN_L=11_d=200.yaml
configs/DeLiteGCN/500k_stretched/peptides-func-DeLiteGCN_L=13_d=190.yaml
configs/DeLiteGCN/500k_stretched/peptides-func-DeLiteGCN_L=15_d=175.yaml
)

for run in "${runs[@]}" ; do
  python main.py --cfg "$run" device cuda dataset.dir /data/beng/datasets
  python bash_scripts/progress_bar.py "$run"
done
