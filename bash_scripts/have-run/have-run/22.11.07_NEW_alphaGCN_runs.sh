#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
# correct rbar pept-func

configs/alphaGCN/peptides-func-alphaGCN_L=05_d=140.yaml
# configs/alphaGCN/peptides-func-alphaGCN_L=07_d=100.yaml
# configs/alphaGCN/peptides-func-alphaGCN_L=09_d=080.yaml
# configs/alphaGCN/peptides-func-alphaGCN_L=11_d=065.yaml
# configs/alphaGCN/peptides-func-alphaGCN_L=13_d=055.yaml
configs/alphaGCN/peptides-func-alphaGCN_L=15_d=050.yaml
configs/alphaGCN/peptides-func-alphaGCN_L=05_d=140_alpha=11.yaml
configs/alphaGCN/peptides-func-alphaGCN_L=07_d=100_alpha=11.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done