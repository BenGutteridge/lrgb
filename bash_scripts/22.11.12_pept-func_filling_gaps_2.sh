#!/bin/bash
cd ..
BATCH="22.11.12_pept-func_filling_gaps"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # r*=1: 17,19,21
  # r*=inf: 17,19,21
  configs/rbar-GCN/peptides-func-DelayGCN_L=17_rbar=inf.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=17_rbar=01.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=19_rbar=01.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=19_rbar=inf.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=21_rbar=01.yaml # <- RUN ALREADY - STOP IF YOU CAN
  configs/rbar-GCN/peptides-func-DelayGCN_L=21_rbar=inf.yaml
  )
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done