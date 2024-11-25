#!/bin/bash

#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sweep-timesteps
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/error.%N.%j.log 
#SBATCH --mem=16G 
#SBATCH --gres=gpu:a6000:1

CPT=$HOME/cls-cond-mdlm/db/arxiv-abs/2024.11.21/175929/db/checkpoints/2-53500.ckpt

python main.py \
  mode=sweep_timesteps \
  model=medium-cond \
  eval.checkpoint_path=$CPT \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  loader.eval_batch_size=1 \
  backbone=dit \
  sweep.num_samples=100
# backbone=hf_dit \
