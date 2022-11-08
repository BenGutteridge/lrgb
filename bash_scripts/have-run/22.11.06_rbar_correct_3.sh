#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
# correct rbar pept-func
# configs/rbar-GCN/peptides-func-DelayGCN_L=15_d=060_rbar=02.yaml
# configs/rbar-GCN/peptides-func-DelayGCN_L=15_d=060_rbar=03.yaml
# configs/rbar-GCN/peptides-func-DelayGCN_L=15_d=060_rbar=04.yaml
configs/rbar-GCN/peptides-func-DelayGCN_L=15_d=060_rbar=05.yaml
configs/rbar-GCN/peptides-func-DelayGCN_L=15_d=060_rbar=06.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 2 device cuda optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done