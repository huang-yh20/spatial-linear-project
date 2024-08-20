#!/bin/bash

trial_num=5

for trial1 in $(seq 0 $(($trial_num - 1)))
do
    sbatch --export=TRIAL=$trial slurm_script_diagram.sh
done

