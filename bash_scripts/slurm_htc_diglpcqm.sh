#! /bin/bash
#SBATCH --job-name=Qdig
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


file='configs/GCN/pcqm-contact-GCN.yaml'

# seed=$SLURM_ARRAY_TASK_ID
seed=0
dir=datasets
out_dir=results/diglpcqm
ckpt_period=10
edge_encoder=False
epochs=100

digl_alpha=0.20
avg_deg=$SLURM_ARRAY_TASK_ID
# avg_deg=15
tf="digl=$avg_deg"

python main.py --cfg "$file" --repeat 1 digl.alpha $digl_alpha dataset.transform $tf seed $seed dataset.edge_encoder $edge_encoder fixed_params.N 500_000 train.auto_resume True train.ckpt_period $ckpt_period out_dir $out_dir device cuda dataset.dir "$dir" optim.max_epoch $epochs tensorboard_each_run True train.mode custom