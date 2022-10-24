#!/bin/bash
cd ..
python main.py --cfg configs/GT/peptides-func-Transformer+LapPE.yaml device cuda dataset.dir /data/beng/datasets
# python main.py --cfg configs/GatedGCN/peptides-func-GatedGCN.yaml device cuda dataset.dir /data/beng/datasets
python main.py --cfg configs/GCNII/peptides-func-GCNII.yaml device cuda dataset.dir /data/beng/datasets
python main.py --cfg configs/MLP/peptides-func-MLP.yaml device cuda dataset.dir /data/beng/datasets
python main.py --cfg configs/SAN/peptides-func-SAN.yaml device cuda dataset.dir /data/beng/datasets
python main.py --cfg configs/SAN/peptides-func-SAN+RWSE.yaml device cuda dataset.dir /data/beng/datasets