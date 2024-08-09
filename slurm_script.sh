#!/bin/bash

#SBATCH -e test.err
#SBATCH -o test.out
#SBATCH -J spatial-linear

#SBATCH --partition=IGIpcluster
#SBATCH --nodes=1
#SBATCH --time=71:59:00
#SBATCH --exclusive




###
python_script="$1"
srun python "$python_script"
