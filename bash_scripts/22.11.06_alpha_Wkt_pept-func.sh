#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
  configs/KGCN/500k_stretched/peptides-func-KGCN_L=07_d=130_RUN_AGAIN.yaml
  configs/alpha_kGCN/peptides-func-alpha_kt_GCN_L=05_d=175.yaml
  configs/alpha_kGCN/peptides-func-alpha_kt_GCN_L=07_d=130.yaml
  configs/alpha_kGCN/peptides-func-alpha_kt_GCN_L=09_d=100.yaml
  configs/alpha_kGCN/peptides-func-alpha_kt_GCN_L=11_d=085.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets
  python bash_scripts/progress_bar.py "$run"
done