import os

import hydra
import lightning
import pandas as pd
import torch
from tqdm import tqdm

import dataloader
import utils
from main import _load_from_checkpoint, _print_config


K = 5


def _shift_seq(seq: torch.Tensor, mask_token_id: int, epsilon_idx: int) -> torch.Tensor:
    B, L = seq.shape

    shifts = torch.randint(1, L, (B,))

    shifted_seq = torch.zeros_like(seq)

    for i in range(B):
        shift = shifts[i]
        shifted_seq[i, :shift] = seq[i, :shift]
        shifted_seq[i, shift + 1 :] = seq[i, shift:-1]
        shifted_seq[i, shift] = mask_token_id

    return {
        "input_ids": shifted_seq,
        "shifts": shifts,
        "gt": torch.ones_like(shifts) * epsilon_idx,
    }


def _mask_seq(seq: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    B, L = seq.shape

    corrupted_seq = seq.clone()
    corrupted_idces = torch.zeros(B, dtype=torch.long)

    for i in range(B):
        text_indices = torch.nonzero(seq[i] != mask_token_id, as_tuple=False).squeeze()
        if len(text_indices) > 0:  # always true (normally)
            corrupted_idces[i] = text_indices[torch.randint(0, len(text_indices), (1,))]
            corrupted_seq[i, corrupted_idces[i]] = mask_token_id

    return {
        "input_ids": corrupted_seq,
        "shifts": corrupted_idces,
        "gt": torch.tensor([seq[i, idx] for i, idx in enumerate(corrupted_idces)]),
    }


def corrupt_seq(seq: torch.Tensor, mask_token_id: int, strategy: str, epsilon_index: int) -> torch.Tensor:
    if strategy == "shift":
        return _shift_seq(seq, mask_token_id, epsilon_index)
    elif strategy == "mask":
        return _mask_seq(seq, mask_token_id)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def get_bucket(label, n_buckets=64) -> list[int]:
    return torch.floor(label * n_buckets).long().flatten().tolist()


####################################################################################################


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):  # noqa: C901, PLR0915
    lightning.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)

    logger.info("Eval infilling model")

    tokenizer = dataloader.get_tokenizer(config)

    cpt = _load_from_checkpoint(config, tokenizer)

    model = cpt.backbone

    noise = cpt.noise
    device = next(model.parameters()).device
    id_ = config.eval.checkpoint_path.split("/")[-1].split(".")[0]

    EPS = 1e-5

    vocab_size = tokenizer.vocab_size
    if tokenizer.mask_token_id is None:
        mask_index = vocab_size
        vocab_size += 1
    else:
        mask_index = tokenizer.mask_token_id

    epsilon_index = tokenizer.additional_special_tokens.index("[EPS]") + vocab_size

    device = next(model.parameters()).device

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)

    def _eval(dataloader, mode, strategy, limit=-1):
        logger.info(f"Starting evaluation on {mode} set with strategy {strategy}")

        limit = len(dataloader) if limit == -1 else limit
        limit *= config.loader.global_batch_size
        limit = min(limit, len(dataloader) * config.loader.global_batch_size)

        print(f"Limit: {limit}")

        pbar = tqdm(dataloader, desc=f"Validating ({mode})", total=len(dataloader))

        counter = {}
        accs = {}

        total_counter = 0

        for batch in pbar:
            # 'input_ids', 'label', 'attention_mask'
            clean_input = batch["input_ids"]
            labels = batch["label"].to(device)

            buckets = get_bucket(labels)

            corrupted_inputs = corrupt_seq(clean_input, mask_index, strategy, epsilon_index)

            corrupted_ids = corrupted_inputs["input_ids"].to(device)

            shifts = corrupted_inputs["shifts"]
            gt = corrupted_inputs["gt"].to(device)

            t = EPS * torch.ones(corrupted_ids.shape[0], 1)
            sigma_t, _ = noise(t)
            if sigma_t.ndim > 1:
                sigma_t = sigma_t.squeeze(-1)
            sigma_t = sigma_t.to(device)

            with torch.inference_mode():
                log_p_x0 = model(
                    corrupted_ids,
                    sigma_t,
                )  # labels=labels)  # B, L, V

            shift_pred = log_p_x0[torch.arange(corrupted_ids.shape[0]), shifts, :]  # B, V

            topk = shift_pred.topk(K, dim=-1).indices  # B, K

            for i, pred in enumerate(topk):
                bucket = buckets[i]

                if bucket not in accs:
                    accs[bucket] = [0] * K
                if bucket not in counter:
                    counter[bucket] = 0

                counter[bucket] += 1
                for j in range(K):
                    accs[bucket][j] += (pred[: j + 1] == gt[i]).any().item()

            total_counter += corrupted_ids.shape[0]

            if total_counter >= limit:
                break

        # res = [counter] + [acc / counter for acc in accs]

        res = {}

        for bucket, acc in accs.items():
            res[bucket] = [a / counter[bucket] for a in acc]

        pd.DataFrame(res).to_csv(f"{id_}_{mode}_{strategy}.csv")

    # Eval
    _eval(valid_ds, "val", "mask")
    _eval(valid_ds, "val", "shift")

    L = 5_000

    _eval(train_ds, "train", "mask", limit=L)
    _eval(train_ds, "train", "shift", limit=L)

    logger.info("Evaluation done")
    logger.info(f"Results saved in {os.getcwd()}")


if __name__ == "__main__":
    main()
