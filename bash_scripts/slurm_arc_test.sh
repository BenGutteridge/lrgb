#! /bin/bash
#SBATCH --job-name=smallrgb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=short
# must be on htc, only one w/ GPUs
#SBATCH --clusters=htc
# set number of GPUs
#SBATCH --gres=gpu:1
cd $DATA/repos/lrgb/bash_scripts
module load Anaconda3
module load CUDA/11.3
source activate $DATA/lrgb
python -c "import torch; print(torch.__version__)"
# bash run_ps_exp.sh LapPE -1 9
python main.py --cfg configs/rbar-GCN/peptides-struct-DelayGCN+LapPE.yaml device cuda dataset.dir datasets rbar -1 gnn.layers_mp 3 optim.max_epoch 5 gnn.dim_inner 16