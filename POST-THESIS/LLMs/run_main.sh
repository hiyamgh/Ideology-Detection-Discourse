#!/usr/bin/env bash

#SBATCH --job-name=Disc
#SBATCH --account=hg31
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=5:00:00
#SBATCH --array=1-12%1

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

module load python/transformers/r1

python main.py $(head -n $SLURM_ARRAY_TASK_ID jobs_octopus.txt | tail -n 1)