#!/bin/bash

# various r*, L=11, LapPE
bash run_ps_exp.sh LapPE 1 11
bash run_ps_exp.sh LapPE -1 11
bash run_ps_exp.sh LapPE 5 11

# various r*, L=7, LapPE
bash run_ps_exp.sh LapPE 1 7
bash run_ps_exp.sh LapPE -1 7
bash run_ps_exp.sh LapPE 5 7

# various r*, L=15, LapPE
bash run_ps_exp.sh LapPE 1 15
bash run_ps_exp.sh LapPE -1 15
bash run_ps_exp.sh LapPE 5 15