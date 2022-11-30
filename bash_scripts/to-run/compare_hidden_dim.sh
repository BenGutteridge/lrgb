#!/bin/bash
cd ..
BATCH=""
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
# something to compare hidden dim for QM9 and pept, all models -- what is a reasonable thing to fix it on?
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 rbar 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300 train.batch_size 1024 gnn.dim_inner 64
  python bash_scripts/progress_bar.py "$run"
done