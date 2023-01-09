#!/bin/bash

# r*=L/2 (rounded down), L=5,9,13,17 for fixed 500k params
bash run_struct_pe_exp.sh LapPE -1 5 175 datasets
bash run_struct_pe_exp.sh LapPE -1 9 105 datasets
bash run_struct_pe_exp.sh LapPE -1 13 72 datasets
bash run_struct_pe_exp.sh LapPE -1 17 55 dataset