#!/bin/bash

STEPS=$1

if [ -z "$STEPS" ]
then
  echo "Using default number of steps: 10"
  STEPS=10
fi

START=$(date +%s)

CST="mode=sample_eval \
  model=eps-tiny \
  loader.eval_batch_size=1_000 \
  sampling.num_sample_batches=50 \
  backbone=dit \
  sampling.steps=$STEPS \
  loader.num_workers=16 \
  eval.compute_generative_perplexity=False \
  hydra.run.dir=./db/a-grammar-eval-$STEPS"

# palidrome ########################################

# eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_palindrome/2025.03.18/232051/checkpoints/best.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_palindrome \
  data.max_eps_rate=0.20 \
  model.length=80

# no eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_palindrome/2025.03.19/005710/checkpoints/best.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_palindrome \
  data.max_eps_rate=0 \
  model.length=64

# alternating ########################################

# eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_alternating_ab/2025.03.19/030043/checkpoints/best.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_alternating_ab \
  data.max_eps_rate=0.20 \
  model.length=80

# no eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_alternating_ab/2025.03.19/031227/checkpoints/best.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_alternating_ab \
  data.max_eps_rate=0 \
  model.length=64

# balanced ab ########################################

# eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_balanced_ab/2025.03.19/032334/checkpoints/best.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_balanced_ab \
  data.max_eps_rate=0.20 \
  model.length=80

# no eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_balanced_ab/2025.03.19/033705/checkpoints/best.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_balanced_ab \
  data.max_eps_rate=0 \
  model.length=64

# balanced parenthesis ########################################

# eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_balanced_parentheses/2025.03.19/034935/checkpoints/best.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_balanced_parentheses \
  data.max_eps_rate=0.20 \
  model.length=80

# no eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_balanced_parentheses/2025.03.19/052824/checkpoints/best.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_balanced_parentheses \
  data.max_eps_rate=0 \
  model.length=64

# parity ########################################

# eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_parity/2025.03.19/022532/checkpoints/best.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_parity \
  data.max_eps_rate=0.20 \
  model.length=80

# no eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_parity/2025.03.19/024320/checkpoints/best.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_parity \
  data.max_eps_rate=0 \
  model.length=64

END=$(date +%s)

echo "Elapsed time: $((END - START)) seconds"