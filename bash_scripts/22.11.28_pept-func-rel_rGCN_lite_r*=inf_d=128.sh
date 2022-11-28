#!/bin/bash
cd ..
BATCH="cdt1_pept-func-rel_rGCN_lite_r*=inf_d=128"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # don't be fooled by filenames, they've all had rbar set to inf manually
# configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=05_d=175_rbar=01.yaml
# configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=07_d=130_rbar=01.yaml
# configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=09_d=100_rbar=01.yaml
# configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=11_d=085_rbar=01.yaml
# configs/rbar-GCN/rbar=01/peptides-func-DelayGCN_L=13_d=070_rbar=01.yaml
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
  python main.py --cfg "$run" --repeat 1 rbar -1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300 use_edge_labels True gnn.stage_type rel_delay_gnn_lite gnn.dim_inner 128 train.batch_size 64
  python bash_scripts/progress_bar.py "$run"
done