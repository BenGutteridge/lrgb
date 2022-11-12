#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
# alpha L=5-15
configs/alphaGCN/peptides-struct-alphaGCN_L=05_d=135.yaml
configs/alphaGCN/peptides-struct-alphaGCN_L=07_d=095.yaml
configs/alphaGCN/peptides-struct-alphaGCN_L=09_d=075.yaml
configs/alphaGCN/peptides-struct-alphaGCN_L=11_d=060.yaml
# configs/alphaGCN/peptides-struct-alphaGCN_L=13_d=050.yaml
# configs/alphaGCN/peptides-struct-alphaGCN_L=15_d=045.yaml

# beta2 L=5-15
# betaL L=5:15
configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=05_d=300_beta=2.yaml
configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=05_d=300_beta=5.yaml
configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=07_d=250_beta=2.yaml
configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=07_d=250_beta=7.yaml
configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=09_d=225_beta=2.yaml
configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=09_d=225_beta=9.yaml
configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=11_d=200_beta=2.yaml
configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=11_d=200_beta=11.yaml
# configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=13_d=190_beta=2.yaml
# configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=13_d=190_beta=13.yaml
# configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=15_d=175_beta=2.yaml
# configs/betaGCN/500k_stretched/peptides-struct-betaGCN_L=15_d=175_beta=15.yaml

# r*=1 5:15
# r*=inf 5:15
# r*=? 5:15
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda optim.max_epoch 300
  python bash_scripts/progress_bar.py "$run"
done