#! /bin/bash
conda -y install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda -y install pyg=2.0.2 -c pyg -c conda-forge
conda -y install pandas scikit-learn
conda -y install openbabel fsspec rdkit -c conda-forge
conda -y install -c dglteam dgl-cuda11.3
pip install performer-pytorch
pip install torchmetrics==0.7.2
pip install ogb
pip install wandb
pip install tensorboard
pip install tensorboardX
conda clean --all
