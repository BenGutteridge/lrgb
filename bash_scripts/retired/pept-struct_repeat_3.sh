#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
  configs/alpha_kGCN/peptides-struct-alpha_kGCN.yaml
  configs/KGCN/peptides-struct-KGCN.yaml
  configs/DelayGCN/peptides-struct-DelayGCN.yaml
  configs/GCN/peptides-struct-GCN.yaml
  # configs/SAN/peptides-func-SAN.yaml
  # configs/SAN/peptides-func-SAN+RWSE.yaml
  # configs/SAN/peptides-struct-SAN.yaml
  # configs/SAN/peptides-struct-SAN+RWSE.yaml
)
for run in "${runs[@]}" ; do
  python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets
  python bash_scripts/progress_bar.py "$run"
done
