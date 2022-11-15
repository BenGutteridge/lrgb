#!/bin/bash
cd ..
BATCH="22.11.14_pept-struct_25L"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  configs/rbar-GCN/peptides-struct-DelayGCN_L=25_rbar=24.yaml
  )
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir datasets out_dir "results/$BATCH" optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done