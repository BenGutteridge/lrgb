#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
models=( 
  GatedGCN
  GCNII
  GINE
  Transformer
  SAN
)
datasets=(
  peptides-struct
  pcqm-contact
  vocsuperpixels
)
extras=(
  RWSE
  LapPE
)
for model in "${models[@]}" ; do
  for dataset in "${datasets[@]}" ; do
    for extra in "${extras[@]}" ; do
      python main.py --cfg "configs/$model/$dataset-$model+$extra.yaml" device cuda dataset.dir /data/beng/datasets
      python bash_scripts/progress_bar.py "$model-$dataset"
    done
  done
done