#!/bin/bash
cd ..
BATCH="qm9_stragglers_r=inf_B=1024"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # regular peptides func, 5 layers, hidden_dim=64, rbar=inf
  # normal DelayGCN using delay_gnn stage
  # using rel_delay_gnn but edited to (hopefully) be identical

)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300 train.batch_size 1024
  python bash_scripts/progress_bar.py "$run"
done