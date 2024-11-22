#!/bin/bash

#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cls-cond-arxiv
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/error.%N.%j.log 
#SBATCH --mem=16G 
#SBATCH --gres=gpu:a6000:4

python main.py \
  model=medium-cond \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=arxiv-abs \
  wandb.name=cls-cond-arxiv-abs \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000