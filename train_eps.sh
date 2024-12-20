#!/bin/bash

#SBATCH --account=vaz@v100
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 # --ntasks=2 not PL compatible
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread


#SBATCH --job-name=epsilon-lm1b
#SBATCH --output=slurm-logs/epsilon-lm1b.out
#SBATCH --error=slurm-logs/epsilon-lm1b.err
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu-dev


if ! [ -x "$(command -v sbatch)" ]; then
  echo "sbatch is not installed. Running script locally."
  PRE=""
else
  echo "Job started at $(date)"
  export WANDB_MODE=offline
  PRE="srun"
  module purge
  conda deactivate
  source $WORK/projects/cls-cond-mdlm/.venv/bin/activate
fi

$PRE python main.py \
  model=eps-small \
  data=epsilon-lm1b \
  trainer.max_steps=500_000 \
  parameterization=subs \
  model.length=256 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1_000 \
  loader.global_batch_size=32 \
  loader.eval_batch_size=1 \
  wandb.name=small-epsilon-lm1b-test \