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
  loader.eval_batch_size=4_096 \
  sampling.num_sample_batches=100 \
  backbone=dit \
  sampling.steps=$STEPS \
  loader.num_workers=16 \
  eval.compute_generative_perplexity=False \
  hydra.run.dir=./db/grammar-eval-10"



# palidrome ########################################

# eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_palindrome/2025.03.12/085549/checkpoints/last.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_palindrome \
  data.max_eps_rate=0.3

# no eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_palindrome/2025.03.12/090827/checkpoints/last.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_palindrome \
  data.max_eps_rate=0.

# alternating ########################################

# eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_alternating_ab/2025.03.12/093507/checkpoints/last.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_alternating_ab \
  data.max_eps_rate=0.3

# no eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_alternating_ab/2025.03.12/094605/checkpoints/last.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_alternating_ab \
  data.max_eps_rate=0.

# balanced ab ########################################

# eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_balanced_ab/2025.03.12/092122/checkpoints/last.ckpt"  
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_balanced_ab \
  data.max_eps_rate=0.3

# no eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_balanced_ab/2025.03.12/093614/checkpoints/last.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_balanced_ab \
  data.max_eps_rate=0.

# balanced parenthesis ########################################

# eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_balanced_parentheses/2025.03.12/094830/checkpoints/last.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_balanced_parentheses \
  data.max_eps_rate=0.3

# no eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_balanced_parentheses/2025.03.12/100617/checkpoints/last.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_balanced_parentheses \
  data.max_eps_rate=0.

# parity ########################################

# eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_parity/2025.03.12/002939/checkpoints/last.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_parity \
  data.max_eps_rate=0.3

# no eps
CPT="/Brain/private/$USER/cls-cond-mdlm/db/grammar_parity/2025.03.12/004031/checkpoints/last.ckpt"
python main.py \
  $CST \
  eval.checkpoint_path=$CPT \
  data=grammar_parity \
  data.max_eps_rate=0.

END=$(date +%s)

echo "Elapsed time: $((END - START)) seconds"