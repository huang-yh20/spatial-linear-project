#!/bin/bash

trial_num=5

for trial in $(seq 0 $(($trial_num - 1)))
do
    sbatch --export=TRIAL=$trial slurm_script_variousdyn.sh
done

