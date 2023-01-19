#! /bin/bash
#SBATCH --job-name=Sd=64rL/2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH --partition=big
# must be on htc, only one w/ GPUs
# set number of GPUs
#SBATCH --gres=gpu:1

cd ..
module load cuda/10.2
module load python/anaconda3
source $condaDotFile
conda activate lrgb2
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pe=none
# task='func'
task='struct'
# file="configs/GCN/peptides-${task}-GCN+${pe}.yaml"
# file="configs/GCN/peptides-${task}-ResGCN+${pe}.yaml"
file="configs/rbar-GCN/peptides-${task}-DelayGCN+${pe}.yaml"

dir=datasets
d=64
L=$SLURM_ARRAY_TASK_ID
# rbar=-1
rbar=$(($SLURM_ARRAY_TASK_ID/2))
echo "r*=$rbar"
python3.9 main.py --cfg "$file" --repeat 3 device cuda dataset.dir "$dir" rbar $rbar gnn.layers_mp $L optim.max_epoch 300 gnn.dim_inner $d tensorboard_each_run False train.mode my_custom

# python3.9 main.py --cfg file="configs/rbar-GCN/peptides-struct-DelayGCN+none.yaml" --repeat 3 device cuda dataset.dir datasets rbar 1 gnn.layers_mp 17 optim.max_epoch 300 gnn.dim_inner 64 tensorboard_each_run False train.mode my_custom