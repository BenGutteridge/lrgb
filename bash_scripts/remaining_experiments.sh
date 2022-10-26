#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=(
  # configs/GCNII/peptides-struct-GCNII.yaml
  # configs/GINE/peptides-struct-GINE.yaml
  # configs/GatedGCN/peptides-struct-GatedGCN.yaml
  # configs/MLP/peptides-struct-MLP.yaml
  # configs/SAN/peptides-struct-SAN+RWSE.yaml
  # configs/SAN/peptides-struct-SAN.yaml
  # configs/expts_l2/GCNII/peptides-struct-GCNII.yaml
  # configs/expts_l2/GINE/peptides-struct-GINE.yaml
  # configs/expts_l2/GatedGCN/peptides-struct-GatedGCN.yaml
  # configs/DeLiteGCN/pcqm-contact-DeLiteGCN.yaml
  # configs/GCNII/pcqm-contact-GCNII.yaml
  # configs/GINE/pcqm-contact-GINE.yaml
  # configs/GatedGCN/pcqm-contact-GatedGCN.yaml
  # configs/SAN/pcqm-contact-SAN.yaml
  # configs/SAN/pcqm-contact-SAN+RWSE.yaml
  # configs/expts_l2/GCNII/pcqm-contact-GCNII.yaml
  # configs/expts_l2/GINE/pcqm-contact-GINE.yaml
  # configs/expts_l2/GatedGCN/pcqm-contact-GatedGCN.yaml
  configs/DeLiteGCN/vocsuperpixels-DeLiteGCN.yaml
  configs/DelayGCN/vocsuperpixels-DelayGCN.yaml
  configs/GCN/vocsuperpixels-GCN.yaml
  configs/GCNII/vocsuperpixels-GCNII.yaml
  configs/GINE/vocsuperpixels-GINE.yaml
  configs/GatedGCN/vocsuperpixels-GatedGCN.yaml
  configs/KGCN/vocsuperpixels-KGCN.yaml
  configs/SAN/vocsuperpixels-SAN.yaml
  configs/SAN/vocsuperpixels-SAN+RWSE.yaml
  configs/Transformer/vocsuperpixels-Transformer+LapPE.yaml
  configs/expts_l2/GCN/vocsuperpixels-GCN.yaml
  configs/expts_l2/GCNII/vocsuperpixels-GCNII.yaml
  configs/expts_l2/GINE/vocsuperpixels-GINE.yaml
  configs/expts_l2/GatedGCN/vocsuperpixels-GatedGCN.yaml
)

for run in "${runs[@]}" ; do
  python main.py --cfg "$run" device cuda dataset.dir /data/beng/datasets
  python bash_scripts/progress_bar.py "$run"
done
