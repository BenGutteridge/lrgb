#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
  configs/alpha_kGCN/peptides-func-alpha_kGCN.yaml
  configs/KGCN/peptides-func-KGCN.yaml
  configs/DelayGCN/peptides-func-DelayGCN.yaml
  configs/GCN/peptides-func-GCN.yaml
  # configs/SAN/peptides-func-SAN.yaml
  # configs/SAN/peptides-func-SAN+RWSE.yaml
)
for run in "${runs[@]}" ; do
  python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets
  python bash_scripts/progress_bar.py "$run"
done
