# train epsilon-conditional model 

CPT=$HOME/cls-cond-mdlm/db/epsilon-text8/2024.12.16/214846/checkpoints/best.ckpt

python main.py \
  model=eps-small \
  data=epsilon-text8 \
  trainer.max_steps=500_000 \
  parameterization=subs \
  model.length=256 \
  eval.compute_generative_perplexity=True \
  sampling.steps=50_000 \
  mode=sample_eval \
  loader.global_batch_size=32 \
  eval.checkpoint_path=$CPT \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=2 \
  wandb=False
  #wandb.name=small-epsilon-text8 \