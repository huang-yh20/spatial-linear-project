#!/bin/bash

find . -type f -name "dyn_gif*.py" | while read python_file; do
    sbatch slurm_script.sh "$python_file"
done

