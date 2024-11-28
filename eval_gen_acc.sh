#!/bin/bash

# last checkpoint
CPT=$HOME/cls-cond-mdlm/db/amazon-polarity/2024.11.26/161320/checkpoints/best.ckpt

START=$(date +%s)

python main.py \
  mode=gen_acc_eval \
  data=amazon-polarity \
  model=medium-cond-amazon \
  eval.checkpoint_path=$CPT \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=10 \
  backbone=dit \
  sampling.steps=128

END=$(date +%s)

echo "Elapsed time: $((END - START)) seconds"