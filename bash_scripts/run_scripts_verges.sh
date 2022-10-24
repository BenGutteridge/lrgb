#!/bin/bash
python progress_bar.py
cd ..
models=( 
  GCN
  DelayGCN
)
datasets=(
  peptides-func
)
for model in "${models[@]}" ; do
  for dataset in "${datasets[@]}" ; do
    python main.py --cfg "configs/$model/$dataset-$model.yaml" device cuda optim.max_epoch 1
    # python bash_scripts/progress_bar.py "$model-$dataset"
  done
done