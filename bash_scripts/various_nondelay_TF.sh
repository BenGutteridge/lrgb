#!/bin/bash
cd ..
python main.py --cfg configs/Transformer/peptides-struct-Transformer+LapPE.yaml --repeat 3 optim.max_epoch 300 device cuda dataset.dir /data/beng/datasets train.batch_size 64
python main.py --cfg configs/SAN/peptides-struct-SAN+LapPE.yaml --repeat 3 optim.max_epoch 300 device cuda dataset.dir /data/beng/datasets





