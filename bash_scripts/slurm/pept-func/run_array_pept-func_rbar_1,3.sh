#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=72:00:00

# set name of job
#SBATCH --job-name=pf-r1,3

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=gutterboyben@gmail.com

# puts outfiles in outfiles folder
#SBATCH -o ./outfiles/output.%a.out # STDOUT

# run the application
module load cuda/10.2
module load python/anaconda3
source $condaDotFile
conda activate lrgb2
conda info --
echo $CONDA_DEFAULT_ENV
# conda list
# bash 22.12.05_jade_pept-func_alpha.sh "${SLURM_ARRAY_TASK_ID}"
L="${SLURM_ARRAY_TASK_ID}"
bash 22.12.06_jade_pept-func_rbar=01.sh "$L"
bash 22.12.06_jade_pept-func_rbar=03.sh "$L" &
      bash 22.12.06_jade_pept-func_rbar=half_L.sh "$L"
# bash 22.12.06_jade_pept-func_rbar=inf.sh "$L"
# bash 22.12.06_jade_pept-func_rbar=L-2.sh "$L"
# bash 22.12.06_jade_pept-func_alpha.sh "$L"