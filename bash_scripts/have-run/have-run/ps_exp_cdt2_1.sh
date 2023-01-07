#!/bin/bash

# various r*, L=11, RWSE
bash run_ps_exp.sh none 1 11
bash run_ps_exp.sh RWSE -1 11
bash run_ps_exp.sh RWSE 5 11

# various r*, L=7, RWSE
bash run_ps_exp.sh RWSE 1 7
bash run_ps_exp.sh RWSE -1 7
bash run_ps_exp.sh RWSE 5 7

# various r*, L=15, RWSE
bash run_ps_exp.sh RWSE 1 15
bash run_ps_exp.sh RWSE -1 15
bash run_ps_exp.sh RWSE 5 15