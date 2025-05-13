#!/bin/bash

#SBATCH -e "%j.err"
#SBATCH -o test.out
#SBATCH -J spatial-linear

#SBATCH --partition=IMLpcluster
#SBATCH --nodes=1
#SBATCH --time=71:59:00
#SBATCH --exclusive




###
conda activate pt2
python_script="$1"
srun python "$python_script"
