#! /bin/bash
#SBATCH --job-name=QMixP3
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
# cd $DATA/repos/clean_lrgb/lrgb

module purge
module load Anaconda3
module load CUDAcore/11.1.1
source activate $DATA/longrange
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"


file='configs/GCN/pcqm-contact-GCN.yaml'
# file='configs/GCN/pcqm-contact-GCN+LapPE.yaml'

seed=$SLURM_ARRAY_TASK_ID
# seed=0
dir=datasets
out_dir=results/mixhop_pcqm
ckpt_period=10
edge_encoder=False
epochs=90
use_CC=True

gnn=mixhop_gcn

bn=True

P=3

# L=$SLURM_ARRAY_TASK_ID

python main.py --cfg "$file" --repeat 1 mixhop_args.max_P $P seed $seed agg_weights.convex_combo $use_CC dataset.edge_encoder $edge_encoder model.type $gnn fixed_params.N 500_000 train.auto_resume True train.ckpt_period $ckpt_period out_dir $out_dir device cuda dataset.dir "$dir" optim.max_epoch $epochs tensorboard_each_run True train.mode custom
