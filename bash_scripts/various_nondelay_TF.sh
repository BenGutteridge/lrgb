#!/bin/bash
cd ..
python main.py --cfg configs/Transformer/peptides-struct-Transformer+LapPE.yaml --repeat 3 optim.max_epoch 300 device cuda
python main.py --cfg configs/SAN/peptides-struct-SAN+LapPE.yaml --repeat 3 optim.max_epoch 300 device cuda





