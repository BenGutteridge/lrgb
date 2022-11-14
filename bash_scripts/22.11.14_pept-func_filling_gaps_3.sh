#!/bin/bash
cd ..
BATCH="22.11.12_pept-func_filling_gaps_3"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # r*2,3,4: 5,7,9,11,13,17,19
  configs/rbar-GCN/peptides-func-DelayGCN_L=05_d=175_rbar=03.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=07_d=130_rbar=03.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=09_d=100_rbar=03.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=11_d=085_rbar=03.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=13_d=070_rbar=03.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=17_rbar=03.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=19_rbar=03.yaml

  configs/rbar-GCN/peptides-func-DelayGCN_L=05_d=175_rbar=02.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=07_d=130_rbar=02.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=09_d=100_rbar=02.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=11_d=085_rbar=02.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=13_d=070_rbar=02.yaml

  configs/rbar-GCN/peptides-func-DelayGCN_L=05_d=175_rbar=04.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=07_d=130_rbar=04.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=09_d=100_rbar=04.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=11_d=085_rbar=04.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=13_d=070_rbar=04.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=17_rbar=04.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=19_rbar=04.yaml

  configs/rbar-GCN/peptides-func-DelayGCN_L=21_rbar=03.yaml
  configs/rbar-GCN/peptides-func-DelayGCN_L=21_rbar=04.yaml
  # r*=1: 17,19,21
  # r*=inf: 17,19,21
  )
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done