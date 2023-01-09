#!/bin/bash
cd ..
python main.py --cfg configs/GatedGCN/peptides-struct-GatedGCN+RWSE.yaml --repeat 3 optim.max_epoch 300
python main.py --cfg configs/GatedGCN/peptides-struct-GatedGCN+LapPE.yaml --repeat 3 optim.max_epoch 300





