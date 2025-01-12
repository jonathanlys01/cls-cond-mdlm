#!/bin/bash

#SBATCH --account=vaz@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread


#SBATCH --job-name=attention-bench
#SBATCH --output=slurm-logs/attention-bench.out
#SBATCH --error=slurm-logs/attention-bench.out
#SBATCH --time=1:00:00
#SBATCH --qos=qos_gpu_a100-dev


if ! [ -x "$(command -v sbatch)" ]; then
  echo "sbatch is not installed. Running script locally."
  PRE=""
else
  echo "Job started at $(date)"
  PRE="srun"
  module purge
  module load arch/a100
  source $WORK/projects/cls-cond-mdlm/.venv/bin/activate
fi

cd models
$PRE python position.py