#!/bin/bash

bash 22.12.13_QM9_R-SPN.sh 8 10 & 
  bash 22.12.13_QM9_R-SPN.sh 8 5 & 
  bash 22.12.13_QM9_R*-SPN.sh 8 1 & 
  bash 22.12.13_QM9_R*-SPN.sh 8 -1 &
  bash 22.12.13_QM9_R*-SPN.sh 8 4