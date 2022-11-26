#!/bin/bash
cd ..
BATCH="pept-func-rel_rGCN_large_L_r*=1"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=13_d=070_rbar=01.yaml
configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=15_d=060_rbar=01.yaml
configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=17_rbar=01.yaml
configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=19_rbar=01.yaml
configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=21_rbar=01.yaml
configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=23_rbar=01.yaml
configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=25_rbar=01.yaml
configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=27_rbar=01.yaml
configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=29_rbar=01.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir datasets out_dir "results/$BATCH" optim.max_epoch 300 use_edge_labels True gnn.stage_type rel_delay_gnn
  python bash_scripts/progress_bar.py "$run"
done