#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=00:10:00

# set name of job
#SBATCH --job-name=test_something

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
# bash 22.12.05_jade_QM9_alpha.sh "${SLURM_ARRAY_TASK_ID}"
N="${SLURM_ARRAY_TASK_ID}"
echo $N & echo $((N+10))
      # bash 22.12.06_jade_QM9_rbar=03.sh "${SLURM_ARRAY_TASK_ID}" &
      # bash 22.12.06_jade_QM9_rbar=half_L.sh "${SLURM_ARRAY_TASK_ID}" &
      # bash 22.12.06_jade_QM9_rbar=inf.sh "${SLURM_ARRAY_TASK_ID}" &
      # bash 22.12.06_jade_QM9_rbar=L-2.sh "${SLURM_ARRAY_TASK_ID}"