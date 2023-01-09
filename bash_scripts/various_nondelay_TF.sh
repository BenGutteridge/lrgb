#!/bin/bash
python main.py --cfg configs/Transformer/peptides-struct-Transformer+LapPE.yaml --repeat 3 optim.max_epoch 300
python main.py --cfg configs/SAN/peptides-struct-SAN+LapPE.yaml --repeat 3 optim.max_epoch 300





