#!/bin/bash

trial_num=21

for trial1 in $(seq 0 $(($trial_num - 1)))
do
    for trial2 in $(seq 0 $(($trial_num - 1)))
    do
        sbatch --export=TRIAL1=$trial1,TRIAL2=$trial2 slurm_script.sh
    done
done

