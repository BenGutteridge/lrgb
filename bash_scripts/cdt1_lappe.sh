#!/bin/bash

# r*=1 L=5,9,13,17 for fixed 500k params
bash run_struct_pe_exp.sh LapPE 1 5 175 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 1 9 105 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 1 13 72 data/beng/datasets
bash run_struct_pe_exp.sh LapPE 1 17 55 data/beng/datasets
