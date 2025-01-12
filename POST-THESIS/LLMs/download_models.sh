#!/usr/bin/env bash
#SBATCH --job-name=dwl
#SBATCH --account=hg31
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00

module load python/transformers/r1

huggingface-cli login --token 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
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir /scratch/8379933-hg31/huggingface_models/meta-llama/Llama-3.2-1B
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir /scratch/8379933-hg31/huggingface_models/meta-llama/Llama-3.1-8B
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /scratch/8379933-hg31/huggingface_models/Qwen/Qwen2.5-7B-Instruct
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir /scratch/8379933-hg31/huggingface_models/meta-llama/Llama-3.1-8B-Instruct