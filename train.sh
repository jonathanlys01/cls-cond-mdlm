#!/bin/bash

#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=cls-cond-arxiv
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/error.%N.%j.log 
#SBATCH --mem=10G 
#SBATCH --gres=gpu:a6000:1

python main.py \
  trainer.max_steps=1000 \
  model=medium-cond \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=arxiv-abs \
  wandb.name=cls-cond-arxiv-a \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000