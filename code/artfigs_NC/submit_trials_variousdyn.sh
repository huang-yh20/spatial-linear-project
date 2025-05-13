#!/bin/bash

for trial2 in $(seq 0 4)
do
    for trial1 in $(seq 0 4)
    do
        sbatch --export=TRIAL1=$trial1,TRIAL2=$trial2 slurm_script.sh
    done
done





