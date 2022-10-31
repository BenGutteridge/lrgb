#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
  configs/SAN/peptides-func-SAN.yaml
  configs/SAN/peptides-func-SAN+RWSE.yaml
)
for run in "${runs[@]}" ; do
  python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python bash_scripts/progress_bar.py "$run"
done
