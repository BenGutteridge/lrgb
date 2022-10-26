#!/bin/bash
cd ..
python bash_scripts/progress_bar.py
runs=( 
peptides-struct-GCNII
peptides-struct-GINE
peptides-struct-GatedGCN
peptides-struct-KGCN
peptides-struct-MLP
peptides-struct-SAN+RWSE
peptides-struct-SAN
peptides-struct-GCNII
peptides-struct-GINE
peptides-struct-GatedGCN
pcqm-contact-DeLiteGCN
pcqm-contact-GCNII
pcqm-contact-GINE
pcqm-contact-GatedGCN
pcqm-contact-KGCN
pcqm-contact-SAN
pcqm-contact-SAN+RWSE
pcqm-contact-GCNII
pcqm-contact-GINE
pcqm-contact-GatedGCN
vocsuperpixels-DeLiteGCN
vocsuperpixels-GCN
vocsuperpixels-GCNII
vocsuperpixels-GINE
vocsuperpixels-GatedGCN
vocsuperpixels-SAN
vocsuperpixels-SAN+RWSE
vocsuperpixels-Transformer+LapPE
vocsuperpixels-GCN
vocsuperpixels-GCNII
vocsuperpixels-GINE
vocsuperpixels-GatedGCN
)

for run in "${runs[@]}" ; do
  python main.py --cfg "$run" device cuda dataset.dir /data/beng/datasets
  python bash_scripts/progress_bar.py "$run"
done
