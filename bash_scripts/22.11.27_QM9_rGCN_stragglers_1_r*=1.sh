#!/bin/bash
cd ..
BATCH="QM9_non-rel_stragglers_d=128"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
# QM9 r*=1, L=13-23
# QM9 r*=inf, L=13-23 (have several of these but before fixing the empty S_k issue)

# alphaGCN, 13-23 (same empty S_k issue)
# configs/alphaGCN/QM9-alphaGCN_L=02_d=335.yaml
# configs/alphaGCN/QM9-alphaGCN_L=05.yaml
# configs/alphaGCN/QM9-alphaGCN_L=07.yaml
# configs/alphaGCN/QM9-alphaGCN_L=09.yaml
# configs/alphaGCN/QM9-alphaGCN_L=11.yaml
# configs/alphaGCN/QM9-alphaGCN_L=13.yaml
# configs/alphaGCN/QM9-alphaGCN_L=15.yaml
# configs/alphaGCN/QM9-alphaGCN_L=17.yaml

configs/rbar-GCN/rbar=01/QM9-rGCN_L=13_r=01.yaml
configs/rbar-GCN/rbar=01/QM9-rGCN_L=15_r=01.yaml
configs/rbar-GCN/rbar=01/QM9-rGCN_L=17_r=01.yaml
configs/rbar-GCN/rbar=01/QM9-rGCN_L=19_r=01.yaml
configs/rbar-GCN/rbar=01/QM9-rGCN_L=21_r=01.yaml
configs/rbar-GCN/rbar=01/QM9-rGCN_L=23_r=01.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 rbar 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300 train.batch_size 1024 gnn.dim_inner 128
  python bash_scripts/progress_bar.py "$run"
done