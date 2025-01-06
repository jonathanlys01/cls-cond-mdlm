#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=prepost
#SBATCH --time=05:00:00
#SBATCH --account=vaz@v100
#SBATCH --job-name=lm1b_download
#SBATCH --output=slurm-logs/dummy_job.out
#SBATCH --error=slurm-logs/dummy_job.err

module purge
source $WORK/projects/cls-cond-mdlm/.venv/bin/activate

srun python epsilon/lm1b_dataset.py --cache_dir $SCRATCH/data
