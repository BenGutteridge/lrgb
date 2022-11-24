#!/bin/bash
cd ..
BATCH="qm9_r=inf_edges"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=05_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=07_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=09_r=inf.yaml
  configs/rbar-GCN/rbar=inf/QM9-rGCN_L=11_r=inf.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300 use_edge_labels True gnn.stage_type rel_delay_gnn train.batch_size 1024
  python bash_scripts/progress_bar.py "$run"
done