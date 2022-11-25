#!/bin/bash
cd ..
BATCH="debugging_rel"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # configs/debug_cfgs/peptides-func-DelayGCN_L=05_d=064_rbar=inf.yaml
  configs/debug_cfgs/peptides-func-RelDelayGCN_L=05_d=064_rbar=inf.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir datasets out_dir "results/$BATCH"
  python bash_scripts/progress_bar.py "$run"
done