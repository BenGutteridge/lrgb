#!/bin/bash

# Hardcode two arrays of integers with a fixed length of 4
array1=(1 2 3 4)
array2=(5 6 7 8)

# Take in an integer as argument $1
index=$1

# Run a bash file called run.sh with the indexed values in the two arrays as arguments
# ./run.sh ${array1[$index]} ${array2[$index]}
echo "${array1[$index]} and ${array2[$index]}"