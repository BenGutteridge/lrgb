#! /bin/bash
#SBATCH --job-name=func_r*=1
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
module load Anaconda3
module load CUDA/11.3
source activate $DATA/lrgb2
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

pe=none
file="configs/rbar-GCN/pept-func-DelayGCN+${pe}.yaml"

dir=datasets

layers=(7 9 11 13 15 17)
dims=(130 105 85 72 64 55)
rbars=(1 1 1 1 1 1)

python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" rbar ${rbars[$SLURM_ARRAY_TASK_ID]} gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner ${dims[$SLURM_ARRAY_TASK_ID]} tensorboard_each_run True train.mode my_custom