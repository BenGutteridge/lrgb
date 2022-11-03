#!/bin/bash
cd ..
python bash_scripts/progress_bar.py

files=(
  configs/DelayGCN_rbar/peptides-struct-DelayGCN_rbar=
  configs/DelayGCN_rbar/peptides-struct-DelayGCN_rbar=
  configs/DelayGCN_rbar/peptides-struct-DelayGCN_rbar=
  configs/DelayGCN_rbar/peptides-struct-DelayGCN_rbar=
  configs/DelayGCN_rbar/peptides-struct-DelayGCN_rbar=
  configs/DelayGCN_rbar/peptides-struct-DelayGCN_rbar=
)

rbars=(
  1
  2
  3
  4
  5
)

for rbar in "${rbars[@]}"; do
  for file in "${files[@]}"; do
    python main.py --cfg "$file"0"$rbar".yaml device cuda dataset.dir /data/beng/datasets rbar "$rbar"
    python bash_scripts/progress_bar.py "$file"0"$rbar".yaml
  done
done