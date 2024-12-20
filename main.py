import os
import random
import time

import fsspec
import hydra
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import rich.syntax
import rich.tree
import torch
from tqdm import tqdm

import dataloader
import diffusion
import utils
from cls_cond.classification import SentimentClassifier
from epsilon.text_dataset import decode_without_epsilon


omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

# SLURM environment variables
omegaconf.OmegaConf.register_new_resolver("work", lambda: os.popen("echo $WORK").read().strip())
omegaconf.OmegaConf.register_new_resolver("scratch", lambda: os.popen("echo $SCRATCH").read().strip())
omegaconf.OmegaConf.register_new_resolver("dsdir", lambda: os.popen("echo $DSDIR").read().strip())


def _load_from_checkpoint(config, tokenizer):
    if "hf" in config.backbone:
        return diffusion.Diffusion(config, tokenizer=tokenizer).to("cuda")

    return diffusion.Diffusion.load_from_checkpoint(config.eval.checkpoint_path, tokenizer=tokenizer, config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open("{}/config_tree.txt".format(config.checkpointing.save_dir), "w") as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        print(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        print("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        print(f"First {k} tokens:", tokenizer.decode(first))
        print("ids:", first)
        print(f"Last {k} tokens:", tokenizer.decode(last))
        print("ids:", last)


def generate_samples(config, logger, tokenizer):
    """
    Generate text samples using a pre-trained model based on the provided configuration.
    Args:
      config (object): Configuration object containing sampling parameters and other settings.
      logger (object): Logger object for logging information.
      tokenizer (object): Tokenizer object for encoding and decoding text.
    Returns:
      tuple: A tuple containing:
        - text_samples (list): List of generated text samples.
        - all_texts (list): List of all generated text samples across batches.
        - all_labels (list): List of labels corresponding to the generated text samples.
    """

    authorized_labels = config.sampling.authorized_labels
    logger.info("Generating samples.")
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    model.gen_ppl_metric.reset()
    all_texts = []
    all_labels = []
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length, num_strides=num_strides, dt=1 / config.sampling.steps
            )
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            labels = [random.choice(authorized_labels) for _ in range(config.loader.eval_batch_size)]
            labels = torch.tensor(labels).to("cuda")
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps, labels=labels)
            if "eps" in config.model.name:
                text_samples = [decode_without_epsilon(samples[i].tolist()) for i in range(samples.size(0))]
            else:
                text_samples = model.tokenizer.batch_decode(samples)
            model.compute_generative_perplexity(text_samples)
        all_texts.extend(text_samples)
        all_labels.extend(labels.tolist())
    print("Text samples:", text_samples)
    if not config.sampling.semi_ar:
        print("Generative perplexity:", model.gen_ppl_metric.compute())
    return all_texts, all_labels


def _sweep_timesteps(config, logger, tokenizer):
    logger.info("Sweeping timesteps.")
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)

    ds_timestep = []
    ds_time = []
    ds_ppl = []

    assert config.sampling.semi_ar is False, "Semi-AR not supported for sweep."
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    log_min_timesteps = np.log2(config.sweep.min)
    log_max_timesteps = np.log2(config.sweep.max)

    pbar = tqdm(
        np.linspace(log_min_timesteps, log_max_timesteps, config.sweep.num_samples),
        total=config.sweep.num_samples,
        desc="Sweeping timesteps",
    )

    for log_timesteps in pbar:
        model.gen_ppl_metric.reset()

        timesteps = int(2**log_timesteps)
        total_time = 0

        for label in config.sampling.authorized_labels:
            start = time.time()

            labels = [label for _ in range(config.loader.eval_batch_size)]
            labels = torch.tensor(labels).to("cuda")

            samples = model.restore_model_and_sample(num_steps=timesteps, labels=labels)

            end = time.time()
            total_time += end - start
            text_samples = model.tokenizer.batch_decode(samples)
            model.compute_generative_perplexity(text_samples)

        ppl = model.gen_ppl_metric.compute().item()
        pbar.set_postfix({"timesteps": timesteps, "ppl": ppl, "time": total_time})

        ds_time.append(total_time)
        ds_timestep.append(timesteps)
        ds_ppl.append(ppl)

    sweep_timesteps = pd.DataFrame({"timesteps": ds_timestep, "time": ds_time, "ppl": ds_ppl})

    sweep_timesteps.to_csv(f"{config.checkpointing.save_dir}/sweep_timesteps.csv", index=False)


def _sweep_cfg(config, logger, tokenizer):
    logger.info("Sweeping Classifier-free Guidance.")
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)

    sentiment_classifier = SentimentClassifier().to("cuda")

    ds_cfg = []
    ds_time = []
    ds_ppl = []
    ds_acc = []

    assert config.sampling.semi_ar is False, "Semi-AR not supported for sweep."
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    log_min_cfg = np.log2(config.sweep.min)
    log_max_cfg = np.log2(config.sweep.max)

    pbar = tqdm(
        np.linspace(log_min_cfg, log_max_cfg, config.sweep.num_samples),
        total=config.sweep.num_samples,
        desc="Sweeping Classifier-free Guidance",
    )

    for log_cfg in pbar:
        cfg = 2**log_cfg
        # OmegaConf only supports primitive types
        # not numpy float64
        model.config.sampling.cfg_scale = float(cfg)
        model.gen_ppl_metric.reset()

        all_texts = []
        all_labels = []

        total_time = 0

        # inference
        for _ in range(config.sampling.num_sample_batches):
            start = time.time()
            labels = [random.choice(config.sampling.authorized_labels) for _ in range(config.loader.eval_batch_size)]
            labels = torch.tensor(labels).to("cuda")
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps, labels=labels)
            end = time.time()
            total_time += end - start
            text_samples = model.tokenizer.batch_decode(samples)
            model.compute_generative_perplexity(text_samples)
            all_texts.extend(text_samples)
            all_labels.extend(labels.tolist())

        # metrics
        ppl = model.gen_ppl_metric.compute().item()

        pred_labels = sentiment_classifier.predict(all_texts)
        acc = sentiment_classifier.compute_accuracy(pred_labels, all_labels)

        pbar.set_postfix({"cfg": cfg, "ppl": ppl, "time": total_time, "acc": acc})

        ds_cfg.append(cfg)
        ds_time.append(total_time)
        ds_ppl.append(ppl)
        ds_acc.append(acc)

    sweep_cfg = pd.DataFrame({"cfg": ds_cfg, "time": ds_time, "ppl": ds_ppl, "acc": ds_acc})

    sweep_cfg.to_csv(f"{config.checkpointing.save_dir}/sweep_cfg.csv", index=False)


def _gen_acc_eval(config, logger, tokenizer):
    logger.info("Evaluating generative accuracy.")
    gen_texts, gt_labels = generate_samples(config, logger, tokenizer)
    classifier = SentimentClassifier()
    pred_labels = classifier.predict(gen_texts)
    acc = classifier.compute_accuracy(pred_labels, gt_labels)
    logger.info(f"Generative accuracy: {acc}")
    return acc


def _ppl_eval(config, logger, tokenizer):
    logger.info("Starting Zero Shot Eval.")

    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    wandb_logger = None
    print(config.get("wandb", None))
    if config.get("wandb", None) is not None and config.get("wandb") is not False:
        wandb_logger = L.pytorch.loggers.WandbLogger(config=omegaconf.OmegaConf.to_object(config), **config.wandb)
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    _, valid_ds = dataloader.get_dataloaders(config, tokenizer, skip_train=True, valid_seed=config.seed)
    trainer.validate(model, valid_ds)


def _eval_infill(config, logger, tokenizer):
    logger.info("Evaluating infilling task.")
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    # eval_infill(config, logger, tokenizer, model)
    print(model)
    raise NotImplementedError("Infilling task not implemented.")


def _train(config, logger, tokenizer):
    logger.info("Starting Training.")
    wandb_logger = None
    if config.get("wandb", None) is not None and config.get("wandb") is not False:
        wandb_logger = L.pytorch.loggers.WandbLogger(config=omegaconf.OmegaConf.to_object(config), **config.wandb)

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    model = diffusion.Diffusion(config, tokenizer=valid_ds.tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.mode == "sample_eval":
        gen_texts, _ = generate_samples(config, logger, tokenizer)
        # write generated samples to file
        with fsspec.open("{}/generated_samples.txt".format(config.checkpointing.save_dir), "w") as fp:
            for text in gen_texts:
                fp.write(text + "\n")
    elif config.mode == "ppl_eval":
        _ppl_eval(config, logger, tokenizer)
    elif config.mode == "sweep":
        fn = {"timesteps": _sweep_timesteps, "cfg": _sweep_cfg}[config.sweep.target]
        fn(config, logger, tokenizer)

    elif config.mode == "gen_acc_eval":
        _gen_acc_eval(config, logger, tokenizer)

    elif config.mode == "infilling":
        _eval_infill(config, logger, tokenizer)

    else:
        _train(config, logger, tokenizer)


if __name__ == "__main__":
    main()
