#!/bin/bash
cd ..
BATCH="qm9_1_edges"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # alphaGCN 5 layers multi runs to see if its consistent
  # stretched QM9 for everything
  # configs/GCN/QM9-GCN_L=02_d=500.yaml
  configs/GCN/QM9-GCN_L=05.yaml
  configs/GCN/QM9-GCN_L=07.yaml
  configs/GCN/QM9-GCN_L=09.yaml
  configs/GCN/QM9-GCN_L=11.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300 use_edge_labels True
  python bash_scripts/progress_bar.py "$run"
done