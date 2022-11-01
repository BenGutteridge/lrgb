#!/bin/bash
cd ..
python bash_scripts/progress_bar.py

files=(
  configs/DelayGCN_rbar/peptides-func-DelayGCN_L=05_d=175_rbar=
  configs/DelayGCN_rbar/peptides-func-DelayGCN_L=07_d=130_rbar=
  configs/DelayGCN_rbar/peptides-func-DelayGCN_L=09_d=100_rbar=
  configs/DelayGCN_rbar/peptides-func-DelayGCN_L=11_d=085_rbar=
  configs/DelayGCN_rbar/peptides-func-DelayGCN_L=13_d=070_rbar=
  configs/DelayGCN_rbar/peptides-func-DelayGCN_L=15_d=060_rbar=
)

rbars=(
  2
  3
  4
  5
  6
)

for rbar in "${rbars[@]}"; do
  for file in "${files[@]}"; do
    python main.py --cfg "$file"0"$rbar".yaml device cuda dataset.dir /data/beng/datasets rbar "$rbar"
    python bash_scripts/progress_bar.py "$file"0"$rbar".yaml
  done
done