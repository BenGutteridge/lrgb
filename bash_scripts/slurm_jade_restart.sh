#! /bin/bash
#SBATCH --job-name=rePCL20
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
python3.9 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

run_dir=$1
cfg = "$run_dir/config.yaml"
dataset_dir=datasets
ckpt_period=10

python3.9 main.py --cfg $cfg --repeat 3 train.auto_resume True train.ckpt_period $ckpt_period out_dir $out_dir dataset.dir "$dataset_dir" train.mode my_custom