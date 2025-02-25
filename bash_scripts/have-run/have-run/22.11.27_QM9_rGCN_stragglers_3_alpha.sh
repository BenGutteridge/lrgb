#!/bin/bash
cd ..
BATCH="QM9_non-rel_stragglers_d=128"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
# alphaGCN, 13-23 (some previously had empty S_k issue)
# configs/alphaGCN/QM9-alphaGCN_L=02_d=335.yaml
configs/alphaGCN/QM9-alphaGCN_L=05.yaml
configs/alphaGCN/QM9-alphaGCN_L=07.yaml
configs/alphaGCN/QM9-alphaGCN_L=09.yaml
configs/alphaGCN/QM9-alphaGCN_L=11.yaml
configs/alphaGCN/QM9-alphaGCN_L=13.yaml
configs/alphaGCN/QM9-alphaGCN_L=15.yaml
configs/alphaGCN/QM9-alphaGCN_L=17.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300 train.batch_size 1024 gnn.dim_inner 128
  python bash_scripts/progress_bar.py "$run"
done