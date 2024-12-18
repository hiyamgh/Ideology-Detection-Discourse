#!/bin/bash

#SBATCH --job-name=eval-mistral
#SBATCH --account=hg31

#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=3:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/transformers/r1

#Mitigate fragmentation
#export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_NODEID


srun --mpi=pmi2 python -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR --nproc_per_node=$SLURM_NTASKS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT testOctopus.py

