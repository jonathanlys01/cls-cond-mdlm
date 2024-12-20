#!/bin/bash

#SBATCH --account=vaz@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 # --ntasks=2 not PL compatible
#SBATCH --gres=gpu:2
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
else
  echo "Job started at $(date)"
  export WANDB_MODE=offline
  PRE="srun"
  module purge
  module load arch/a100
  source $WORK/projects/cls-cond-mdlm/.venv/bin/activate
fi

$PRE python main.py \
  model=eps-small \
  data=epsilon-lm1b \
  trainer.max_steps=54_000 \
  parameterization=subs \
  model.length=256 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1_000 \
  loader.global_batch_size=32 \
  loader.eval_batch_size=1 \
  wandb.name=small-epsilon-lm1b \