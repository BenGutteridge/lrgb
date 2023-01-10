#!/bin/bash
cd ..
python main.py --cfg configs/Transformer/peptides-struct-Transformer+LapPE.yaml --repeat 3 optim.max_epoch 500 device cuda dataset.dir datasets
python main.py --cfg configs/SAN/peptides-struct-SAN+LapPE.yaml --repeat 3 optim.max_epoch 500 device cuda dataset.dir datasets





