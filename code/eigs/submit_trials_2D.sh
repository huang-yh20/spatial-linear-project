#!/bin/bash

file_num=8
param_num=6

for trial1 in $(seq 0 $(($file_num - 1)))
do
    for trial2 in $(seq 0 $(($param_num - 1)))
    do
        sbatch --export=TRIAL1=$trial1,TRIAL2=$trial2 slurm_script.sh
    done
done

