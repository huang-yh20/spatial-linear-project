#!/bin/bash

#SBATCH -e ./temp/%j.err
#SBATCH -o ./temp/%j.out
#SBATCH -J spatial-linear

#SBATCH --partition=IGIpcluster
#SBATCH --nodes=1
#SBATCH --time=71:59:00
#SBATCH --exclusive
#SBATCH --output=./temp/%j.log



###
cd ../..
trial=$TRIAL

python ./code/artfigs/various_dyn_prorec.py $trial
