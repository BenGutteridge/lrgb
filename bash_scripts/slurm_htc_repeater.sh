#! /bin/bash
#SBATCH --job-name=repeatS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH --partition=short
# must be on htc, only one w/ GPUs
#SBATCH --clusters=htc
# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --account=engs-oxnsg

cd $DATA/repos/lrgb

module purge
module load Anaconda3
module load CUDAcore/11.1.1
source activate $DATA/longrange
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"


# file=bash_scripts/to_repeat/PCQM4Mv2Contact-shuffle_drew_gin_LapPE_nu=inf_CC_bn_bs=0256_d=085_L=10/config.yaml

file=bash_scripts/to_repeat/pept-struct_drew_gin_nu=inf_bn_bs=0128_d=059_L=15/config.yaml

# file=bash_scripts/to_repeat/pept-struct_drew_gin_LapPE_nu=inf_bn_bs=0128_d=152_L=05/config.yaml

out_dir=results/repeats

seed=$SLURM_ARRAY_TASK_ID

ckpt_period=10

epochs=300

python main.py --cfg "$file" --repeat 1 seed $seed train.auto_resume True train.ckpt_period $ckpt_period out_dir $out_dir device cuda optim.max_epoch $epochs tensorboard_each_run True train.mode custom