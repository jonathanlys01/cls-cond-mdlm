#!/bin/bash

#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sweep-cfg-fine
#SBATCH --output=slurm-logs/output.%N.%j.log
#SBATCH --error=slurm-logs/error.%N.%j.log 
#SBATCH --mem=16G 
#SBATCH --gres=gpu:a6000:1

# CPT=$HOME/cls-cond-mdlm/db/arxiv-abs/2024.11.21/175929/db/checkpoints/2-53500.ckpt
CPT=$HOME/cls-cond-mdlm/db/amazon-polarity/2024.11.26/161320/checkpoints/best.ckpt

python main.py \
  mode=sweep \
  sweep.target=cfg \
  model=medium-cond-amazon \
  data=amazon-polarity \
  eval.checkpoint_path=$CPT \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  loader.eval_batch_size=1 \
  backbone=dit \
  sampling.num_sample_batches=100 \
  sweep.num_samples=50 \
  sampling.steps=128