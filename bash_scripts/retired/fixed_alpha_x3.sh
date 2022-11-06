#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
  configs/alpha_kGCN/peptides-func-alpha_kGCN.yaml
  configs/alpha_kGCN/peptides-struct-alpha_kGCN.yaml
)
for run in "${runs[@]}" ; do
  python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets fixed_alpha True
  python bash_scripts/progress_bar.py "$run"
done
