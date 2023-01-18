#! /bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:10:00
#SBATCH --partition=devel
# must be on htc, only one w/ GPUs
# set number of GPUs
#SBATCH --gres=gpu:1

cd lrgb
module load cuda/10.2
module load python/anaconda3
source $condaDotFile
conda activate lrgb2
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pe=none
file="configs/rbar-GCN/pept-func-DelayGCN+${pe}.yaml"
dir=datasets
d=64
layers=(7 9 11 13 15 17)
rbar=1

python main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" rbar $rbar gnn.layers_mp ${layers[$SLURM_ARRAY_TASK_ID]} optim.max_epoch 300 gnn.dim_inner $d tensorboard_each_run False train.mode my_custom