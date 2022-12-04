#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:10:00

# set name of job
#SBATCH --job-name=ben_job_1

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=gutterboyben@gmail.com

# run the application
module load tmux
module load cuda/10.2
module load python/anaconda3
conda activate lrgb2
echo "HAVE RUN"
python check_cuda.py