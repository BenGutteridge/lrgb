#! /bin/bash
#SBATCH --job-name=test_lrgb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --partition=devel
# must be on htc, only one w/ GPUs
#SBATCH --clusters=htc
# set number of GPUs
#SBATCH --gres=gpu:1
cd $DATA/repos/lrgb/bash_scripts
module load Anaconda3
module load CUDA/11.3
source activate $DATA/lrgb
python -c "import torch; print(torch.__version__)"
bash run_ps_exp.sh LapPE -1 9