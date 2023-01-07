#!/bin/bash

# r*=L/2 (rounded down), L=5,9,13,17 for fixed 500k params
bash run_struct_pe_exp.sh LapPE 2 5 175 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 4 9 105 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 6 13 72 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 8 17 55 data/beng/datasets
