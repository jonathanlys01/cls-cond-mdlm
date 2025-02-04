#!/bin/bash

#SBATCH --account=vaz@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # --ntasks not PL compatible
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread


#SBATCH --job-name=epsilon-lm1b
#SBATCH --output=slurm-logs/epsilon-lm1b.out
#SBATCH --error=slurm-logs/epsilon-lm1b.err
#SBATCH --time=10:00:00
#SBATCH --qos=qos_gpu_a100-t3


if ! [ -x "$(command -v sbatch)" ]; then
  echo "sbatch is not installed. Running script locally."
  PRE=""
  BS=8
else
  echo "Job started at $(date)"
  export WANDB_MODE=offline
  PRE="srun"
  module purge
  module load arch/a100
  source $WORK/projects/cls-cond-mdlm/.venv/bin/activate
  BS=64
fi

$PRE python main.py \
  model=eps-small \
  data=epsilon-lm1b \
  parameterization=subs \
  eval.compute_generative_perplexity=True \
  sampling.steps=10_000 \
  loader.global_batch_size=$BS \
  loader.eval_batch_size=$BS \
  model.proba_method="bucket" \
  model.length=256 \
  trainer.max_epochs=2 \
  wandb.name=rope-true-mask-epsilon-lm1b
  # wandb=False

# SCRATCH=$(pwd)/db/data ./train_eps.sh 