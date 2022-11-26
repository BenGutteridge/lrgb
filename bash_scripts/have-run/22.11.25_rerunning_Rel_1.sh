#!/bin/bash
cd ..
BATCH="rel_fixed_AGAIN_qm9"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=05_d=175_rbar=01.yaml
  # configs/rbar-GCN/rbar=inf/peptides-func-DelayGCN_L=05_d=175_rbar=inf.yaml
  # configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=07_d=130_rbar=01.yaml
  # configs/rbar-GCN/rbar=inf/peptides-func-DelayGCN_L=07_d=130_rbar=inf.yaml
  # configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=09_d=100_rbar=01.yaml
  # configs/rbar-GCN/rbar=inf/peptides-func-DelayGCN_L=09_d=100_rbar=inf.yaml
  # configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=11_d=085_rbar=01.yaml
  # configs/rbar-GCN/rbar=inf/peptides-func-DelayGCN_L=11_d=085_rbar=inf.yaml

  configs/rbar-GCN/rbar=01/QM9-rGCN_L=05_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=07_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=09_r=01.yaml
  configs/rbar-GCN/rbar=01/QM9-rGCN_L=11_r=01.yaml

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