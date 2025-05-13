#!/bin/bash

#SBATCH -e 1p.err
#SBATCH -o 1p.out
#SBATCH -J spatial-linear

#SBATCH --partition=IMLpcluster
#SBATCH --nodes=1
#SBATCH --time=71:59:00
#SBATCH --exclusive
#SBATCH --output=1p.log



###
cd ../..
trial=$TRIAL

python ./code/artfigs_NC/Fig3_eigs_temp.py $trial
