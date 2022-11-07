#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
# correct rbar pept-func
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=07_d=250_beta=2.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=07_d=250_beta=3.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=07_d=250_beta=5.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=09_d=225_beta=2.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=09_d=225_beta=3.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=09_d=225_beta=5.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=11_d=200_beta=2.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=11_d=200_beta=3.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=11_d=200_beta=5.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=13_d=190_beta=2.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=13_d=190_beta=3.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=13_d=190_beta=5.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=15_d=175_beta=2.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=15_d=175_beta=3.yaml
configs/betaGCN/500k_stretched/peptides-func-betaGCN_L=15_d=175_beta=5.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done