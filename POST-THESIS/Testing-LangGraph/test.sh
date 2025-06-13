#!/bin/bash

#SBATCH --job-name=testlang                 # Job name
#SBATCH --account=hg31                # Account to use
#SBATCH --partition=gpu               # GPU partition
#SBATCH --nodes=1                     # Number of nodes (use 1 node)
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --mem=32000                   # Memory per node (32GB)
#SBATCH --gres=gpu:v100d32q:1         # Request 2 GPUs on the same node
#SBATCH --time=5:00:00                # Maximum runtime

# Load the required module
module load python/3.13

python test.py