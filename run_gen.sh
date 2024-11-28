#!/bin/bash

STEPS=$1

if [ -z "$STEPS" ]
then
  echo "Please provide the number of steps to sample"
  exit 1
fi

# CPT=$HOME/cls-cond-mdlm/db/arxiv-abs/2024.11.21/175929/db/checkpoints/2-53500.ckpt
CPT=$HOME/cls-cond-mdlm/db/amazon-polarity/2024.11.26/161320/checkpoints/best.ckpt

START=$(date +%s)

python main.py \
  mode=sample_eval \
  data=amazon-polarity \
  model=medium-cond-amazon \
  eval.checkpoint_path=$CPT \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=2 \
  backbone=dit \
  sampling.steps=$STEPS
# backbone=hf_dit \

END=$(date +%s)

echo "Elapsed time: $((END - START)) seconds"