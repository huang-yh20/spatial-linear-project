#!/bin/bash

#SBATCH -e diagall.err
#SBATCH -o diagall.out
#SBATCH -J spatial-linear

#SBATCH --partition=IGIpcluster
#SBATCH --nodes=1
#SBATCH --time=71:59:00
#SBATCH --exclusive
#SBATCH --output=output.log



###
cd ../..
trial=$TRIAL

python ./code/phase/phase_diagram1p_all.py $trial
