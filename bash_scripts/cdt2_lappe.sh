#!/bin/bash

# r*=L/2 (rounded down), L=5,9,13,17 for fixed 500k params
bash run_struct_pe_exp.sh LapPE 3 7 130 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 5 11 85 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 7 15 64 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 9 19 50 data/beng/datasets
