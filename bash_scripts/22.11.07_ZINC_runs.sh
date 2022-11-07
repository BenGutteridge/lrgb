#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
# gcn
configs/GCN/ZINC-GCN_L=02_d=500.yaml
configs/GCN/ZINC-GCN_L=05_d=300.yaml
configs/GCN/ZINC-GCN_L=07_d=250.yaml
# alpha
configs/alphaGCN/ZINC-alphaGCN_L=02_d=350.yaml
configs/alphaGCN/ZINC-alphaGCN_L=05_d=140.yaml
configs/alphaGCN/ZINC-alphaGCN_L=07_d=100.yaml
# beta
configs/betaGCN/ZINC-betaGCN_L=02_d=500_beta=2.yaml
configs/betaGCN/ZINC-betaGCN_L=05_d=300_beta=2.yaml
configs/betaGCN/ZINC-betaGCN_L=05_d=300_beta=5.yaml
configs/betaGCN/ZINC-betaGCN_L=07_d=250_beta=2.yaml
configs/betaGCN/ZINC-betaGCN_L=07_d=250_beta=7.yaml
# rbar 1 - inf, L=2-9
configs/rbar-GCN/ZINC-rGCN_L=02_d=400_r=01.yaml
configs/rbar-GCN/ZINC-rGCN_L=02_d=400_r=inf.yaml
configs/rbar-GCN/ZINC-rGCN_L=05_d=175_r=01.yaml
configs/rbar-GCN/ZINC-rGCN_L=05_d=175_r=02.yaml
configs/rbar-GCN/ZINC-rGCN_L=05_d=175_r=04.yaml
configs/rbar-GCN/ZINC-rGCN_L=05_d=175_r=inf.yaml
configs/rbar-GCN/ZINC-rGCN_L=07_d=130_r=01.yaml
configs/rbar-GCN/ZINC-rGCN_L=07_d=130_r=02.yaml
configs/rbar-GCN/ZINC-rGCN_L=07_d=130_r=04.yaml
configs/rbar-GCN/ZINC-rGCN_L=07_d=130_r=06.yaml
configs/rbar-GCN/ZINC-rGCN_L=07_d=130_r=inf.yaml
configs/rbar-GCN/ZINC-rGCN_L=09_d=100_r=01.yaml
configs/rbar-GCN/ZINC-rGCN_L=09_d=100_r=02.yaml
configs/rbar-GCN/ZINC-rGCN_L=09_d=100_r=06.yaml
configs/rbar-GCN/ZINC-rGCN_L=09_d=100_r=inf.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done
