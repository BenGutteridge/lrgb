#! /bin/bash
#SBATCH --job-name=PCrepro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=48:00:00
#SBATCH --partition=medium
# must be on htc, only one w/ GPUs
#SBATCH --clusters=htc
# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --account=engs-oxnsg

cd $DATA/repos/lrgb
# cd $DATA/repos/clean_lrgb/lrgb

module purge
module load Anaconda3
module load CUDAcore/11.1.1
source activate $DATA/longrange
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"


run_dir=$1
cfg = "$run_dir/config.yaml"
dataset_dir=datasets
ckpt_period=10

python main.py --cfg $cfg --repeat 3 train.auto_resume True train.ckpt_period $ckpt_period out_dir $out_dir dataset.dir "$dataset_dir" train.mode my_custom