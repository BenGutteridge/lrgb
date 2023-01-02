#!/bin/bash
cd ..
BATCH="22.11.15_pept-struct_alphas"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # alpha L 13,15,17,19
  configs/alphaGCN/peptides-struct-alphaGCN_L=13_d=050.yaml
  configs/alphaGCN/peptides-struct-alphaGCN_L=15_d=045.yaml
  configs/alphaGCN/peptides-struct-alphaGCN_L=17_d=040.yaml
  configs/alphaGCN/peptides-struct-alphaGCN_L=19_d=035.yaml
  configs/alphaGCN/peptides-struct-alphaGCN_L=21_d=032.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir /data/beng/datasets out_dir "results/$BATCH" optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done