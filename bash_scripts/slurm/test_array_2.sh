#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=12:00:00

# set name of job
#SBATCH --job-name=jade_test

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=gutterboyben@gmail.com

# run the application
module load cuda/10.2
module load python/anaconda3
source $condaDotFile
conda activate lrgb2
conda info --
echo $CONDA_DEFAULT_ENV
# conda list
bash 22.12.04_jade_speed_test_2.sh "${SLURM_ARRAY_TASK_ID}"