#!/bin/bash
cd ..

files=(
  configs/rbar-GCN/peptides-struct-DelayGCN_RWSE.yaml
  configs/rbar-GCN/peptides-struct-DelayGCN_LapPE.yaml
)

rbar=$1

L=$2

for file in "${files[@]}"
do
  python main.py --cfg "$file" device cuda dataset.dir data/beng/datasets rbar "$rbar" gnn.layers_mp "$L"
done