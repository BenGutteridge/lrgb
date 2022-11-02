#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
configs/betaGCN/peptides-func-betaGCN_beta=02.yaml
configs/betaGCN/peptides-func-betaGCN_beta=03.yaml
configs/betaGCN/peptides-func-betaGCN_beta=04.yaml
configs/betaGCN/peptides-func-betaGCN_beta=05.yaml
configs/betaGCN/peptides-struct-betaGCN_beta=02.yaml
configs/betaGCN/peptides-struct-betaGCN_beta=03.yaml
configs/betaGCN/peptides-struct-betaGCN_beta=04.yaml
configs/betaGCN/peptides-struct-betaGCN_beta=05.yaml
)

for run in "${runs[@]}" ; do
  python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets
  python bash_scripts/progress_bar.py "$run"
done