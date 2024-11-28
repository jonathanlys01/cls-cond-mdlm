#!/bin/bash

#SBATCH --job-name=cls-cond-amazon-reviews
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/error.%N.%j.log 
#SBATCH --mem=16G 
#SBATCH --gres=gpu:a6000:4

python main.py \
  model=medium-cond-amazon \
  data=amazon-polarity \
  trainer.max_steps=60_000 \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  wandb.name=cls-cond-amazon-reviews \
  parameterization=subs \
  model.length=1_024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1_000