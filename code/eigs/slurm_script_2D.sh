#!/bin/bash

#SBATCH -e temp\%j.err
#SBATCH -o temp\%j.out
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

python ./code/eigs/eigs2D_product.py $trial1 $trial2