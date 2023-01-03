#!/bin/bash


bash run_ps_exp.sh LapPE -1 11
bash run_ps_exp.sh LapPE -1 5
bash run_ps_exp.sh LapPE 5 11
bash run_ps_exp.sh LapPE 5 5

bash run_ps_exp.sh RWSE -1 5
bash run_ps_exp.sh RWSE 5 5

bash run_ps_exp.sh none -1 11
bash run_ps_exp.sh none -1 5
bash run_ps_exp.sh none 1 15
bash run_ps_exp.sh none -1 15
bash run_ps_exp.sh none 5 15