#!/bin/bash
cd ..
BATCH="22.11.12_pept-struct_filling_gaps_2"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # beta2 L=13,15,17,19
  configs/betaGCN/beta02/peptides-struct-betaGCN_L=13_d=190_beta=2.yaml
  configs/betaGCN/beta02/peptides-struct-betaGCN_L=15_d=175_beta=2.yaml
  configs/betaGCN/beta02/peptides-struct-betaGCN_L=17_d=165_beta=2.yaml
  configs/betaGCN/beta02/peptides-struct-betaGCN_L=19_d=155_beta=2.yaml
  # r*=3 11,13,15,17,19
  configs/rbar-GCN/rbar=03/peptides-struct-DelayGCN_L=11_rbar=03.yaml
  configs/rbar-GCN/rbar=03/peptides-struct-DelayGCN_L=13_rbar=03.yaml
  configs/rbar-GCN/rbar=03/peptides-struct-DelayGCN_L=15_rbar=03.yaml
  configs/rbar-GCN/rbar=03/peptides-struct-DelayGCN_L=17_rbar=03.yaml
  configs/rbar-GCN/rbar=03/peptides-struct-DelayGCN_L=19_rbar=03.yaml
  # r*=10 15,17
  configs/rbar-GCN/rbar=10/peptides-func-DelayGCN_L=15_d=060_rbar=10.yaml
  configs/rbar-GCN/rbar=10/peptides-func-DelayGCN_L=17_d=055_rbar=10.yaml
  )
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done