#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
  configs/alpha_kGCN/peptides-func-alpha_kGCN.yaml
  configs/alpha_kGCN/peptides-struct-alpha_kGCN.yaml
  configs/alpha_kGCN/pcqm-contact-alpha_kGCN.yaml
  configs/alpha_kGCN/vocsuperpixels-alpha_kGCN.yaml
)
for run in "${runs[@]}" ; do
  python main.py --cfg "$run" device cuda dataset.dir /data/beng/datasets
  python bash_scripts/progress_bar.py "$run"
done
