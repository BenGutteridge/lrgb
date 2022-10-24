#!/bin/bash
cd ..
python progress_bar.py
models=( 
  GCN
  DelayGCN
)
datasets=(
  peptides-func
)
for model in "${models[@]}" ; do
  for dataset in "${datasets[@]}" ; do
    python main.py --cfg "configs/$model/$dataset-$model.yaml" device cuda dataset.dir /data/beng/datasets
    # python bash_scripts/progress_bar.py "$model-$dataset"
  done
done