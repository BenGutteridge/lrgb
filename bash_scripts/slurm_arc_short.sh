#! /bin/bash
#SBATCH --job-name=more_r*_struct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH --partition=short
# must be on htc, only one w/ GPUs
#SBATCH --clusters=htc
# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --account=engs-oxnsg
cd $DATA/repos/lrgb/bash_scripts
pe=$1
rbar=$2
module load Anaconda3
module load CUDA/11.3
source activate $DATA/lrgb
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

layers=(7 11 15 19)
dims=(130 85 64 50)
rbars=(3 5 7 9)
bash run_struct_pe_exp.sh $pe ${rbars[$SLURM_ARRAY_TASK_ID]} ${layers[$SLURM_ARRAY_TASK_ID]} ${dims[$SLURM_ARRAY_TASK_ID]} datasets
