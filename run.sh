#!/bin/bash

CPT=$HOME/cls-cond-mdlm/db/arxiv-abs/2024.11.21/175929/db/checkpoints/2-53500.ckpt

python main.py \
  mode=ppl_eval \
  model=medium-cond \
  eval.checkpoint_path=$CPT \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=10 \
  backbone=dit # hf_dit

  #mode=sample_eval \