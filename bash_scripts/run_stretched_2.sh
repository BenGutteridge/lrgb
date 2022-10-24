#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=( 
configs/DelayGCN/500k_stretched/peptides-func-DelayGCN_L=07_d=130.yaml
configs/DelayGCN/500k_stretched/peptides-func-DelayGCN_L=09_d=100.yaml
configs/DelayGCN/500k_stretched/peptides-func-DelayGCN_L=11_d=085.yaml
configs/DelayGCN/500k_stretched/peptides-func-DelayGCN_L=13_d=070.yaml
configs/DelayGCN/500k_stretched/peptides-func-DelayGCN_L=15_d=060.yaml
configs/DeLiteGCN/500k_stretched/peptides-func-DeLiteGCN_L=07_d=250.yaml
configs/DeLiteGCN/500k_stretched/peptides-func-DeLiteGCN_L=09_d=225.yaml
configs/DeLiteGCN/500k_stretched/peptides-func-DeLiteGCN_L=11_d=200.yaml
configs/DeLiteGCN/500k_stretched/peptides-func-DeLiteGCN_L=13_d=190.yaml
configs/DeLiteGCN/500k_stretched/peptides-func-DeLiteGCN_L=15_d=175.yaml
)

for run in "${runs[@]}" ; do
  python main.py --cfg "$run" device cuda dataset.dir /data/beng/datasets
  python bash_scripts/progress_bar.py "$run"
done
