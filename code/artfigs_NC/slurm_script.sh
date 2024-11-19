#!/bin/bash

#SBATCH -e diag.err
#SBATCH -o diag.out
#SBATCH -J spatial-linear

#SBATCH --partition=IGIpcluster
#SBATCH --nodes=1
#SBATCH --time=71:59:00
#SBATCH --exclusive
#SBATCH --output=output.log



###
cd ../..
trial1=$TRIAL1
trial2=$TRIAL2

python ./code/artfigs_NC/Fig2_phase_prorec.py $trial1 $trial2
