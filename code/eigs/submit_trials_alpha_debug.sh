#!/bin/bash

trial1=3
param_num=6


for trial2 in $(seq 0 $(($param_num - 1)))
do
    sbatch --export=TRIAL1=$trial1,TRIAL2=$trial2 slurm_script_2D.sh
done


