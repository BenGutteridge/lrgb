#!/bin/bash

# r*=1 L=5,9,13,17 for fixed 500k params
bash run_struct_pe_exp.sh LapPE 1 7 130 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 1 11 85 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 1 15 64 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 1 19 50 data/beng/datasets
