#! /bin/bash
#SBATCH --job-name=SANslic30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH --partition=short
# must be on htc, only one w/ GPUs
#SBATCH --clusters=htc
# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --account=engs-oxnsg

# cd $DATA/repos/lrgb
cd $DATA/repos/clean_lrgb/lrgb

module purge
module load Anaconda3
module load CUDAcore/11.1.1
source activate $DATA/longrange
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

seed=0
slic=30
epochs=300
# Just for runing pure SAN
python main.py --cfg configs/SAN/vocsuperpixels-SAN.yaml --repeat 3 seed $seed optim.max_epoch $epochs dataset.slic_compactness $slic tensorboard_each_run True dataset.dir ../../lrgb/datasets wandb.use False train.ckpt_period 5 device cuda train.auto_resume True out_dir results/retrySANslic30