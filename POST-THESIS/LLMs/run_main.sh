#!/bin/bash

#SBATCH --job-name=evalla
#SBATCH --account=hg31

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:2
#SBATCH --time=3:00:00

module load python/transformers/r1

python main.py