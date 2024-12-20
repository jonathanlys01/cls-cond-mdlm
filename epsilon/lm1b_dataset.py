import argparse
import functools
import itertools
import os
import re

import datasets
import numpy as np
import torch
from transformers import AutoTokenizer


ETA = 0.2
BLOCK_SIZE = 256  # OG was 128, double with epsilon tokens
MASK = [1] * BLOCK_SIZE

MAX_WORKERS = min(os.cpu_count() - 1, 16)

SPLITS = ["train", "test"]


def get_tokenizer():
    if os.environ.get("DSDIR"):
        jz_path = os.path.join(os.environ.get("DSDIR"), "tokenizer")
        print("Loading from local (JZ)")
        return AutoTokenizer.from_pretrained(jz_path)
    ret = AutoTokenizer.from_pretrained("gpt2")
    print("Adding", ret.add_special_tokens({"pad_token": "[PAD]", "additional_special_tokens": ["[EPS]"]}))
    return ret


tokenizer = get_tokenizer()


def untokenize(x):
    # remove tokenization artifacts
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x + " "


def _apply_detokenize(text):
    for i, t in enumerate(text):
        text[i] = untokenize(t)
    return text


def _group_texts(examples, block_size, bos, eos):
    # Concatenate all texts.
    concatenated_examples = list(itertools.chain(*examples["input_ids"]))
    total_length = len(concatenated_examples)

    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    total_length = (total_length // new_block_size) * new_block_size
    # Split by chunks of max_len.

    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        _values.append([bos] + concatenated_examples[i : i + new_block_size] + [eos])
        _attn_masks.append(torch.ones(block_size, dtype=torch.long))

    return {"input_ids": _values, "attention_mask": _attn_masks, "label": [0] * len(_values)}  # dummy label


group_texts = functools.partial(
    _group_texts,
    block_size=BLOCK_SIZE,
    bos=tokenizer.bos_token_id,
    eos=tokenizer.eos_token_id,
)


def _add_epsilon_tokens(
    x,
    spec_tokens={
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "eps": tokenizer.additional_special_tokens_ids[0],
    },
    n_range=(int(ETA * BLOCK_SIZE), BLOCK_SIZE - int(ETA * BLOCK_SIZE)),
    single=False,  # return single block (truncate the rest)
):
    n_min, n_max = n_range
    eos = spec_tokens["eos"]
    bos = spec_tokens["bos"]
    eps = spec_tokens["eps"]

    def _add_special_tokens(x):
        return [bos] + x[1:-1] + [eos]

    # add epsilon tokens to 1 list of input_ids (1d)
    n_eps = np.random.randint(n_min, n_max + 1)
    np_block = np.array(x)
    p = n_eps / len(np_block)

    # exclude first and last tokens (BOS and EOS)
    p1 = np.random.choice(np.arange(1, len(np_block) - 1), n_eps, replace=False)
    block1 = np.empty_like(np_block)
    block1[p1] = eps
    inv_p1 = np.setdiff1d(np.arange(len(np_block)), p1)

    for i, idx in enumerate(inv_p1):
        block1[idx] = np_block[i]

    if single:
        return [{"input_ids": _add_special_tokens(block1.tolist()), "label": p}]

    p2 = np.random.choice(np.arange(1, len(np_block) - 1), len(np_block) - n_eps, replace=False)
    block2 = np.empty_like(np_block)
    block2[p2] = eps
    inv_p2 = np.setdiff1d(np.arange(len(np_block)), p2)

    for i, idx in enumerate(inv_p2):
        block2[idx] = np_block[i]

    return [
        {"input_ids": _add_special_tokens(block1.tolist()), "label": p},
        {"input_ids": _add_special_tokens(block2.tolist()), "label": 1 - p},
    ]


def preproc(example):
    text = example["text"]
    text = _apply_detokenize(text)

    tokens = tokenizer(
        text,
        max_length=BLOCK_SIZE,
        truncation=True,
    )

    return {"input_ids": tokens["input_ids"]}


def _transform(examples):
    ids = []
    labels = []

    for x in examples["input_ids"]:
        blocks = _add_epsilon_tokens(x)
        for block in blocks:
            ids.append(block["input_ids"])
            labels.append(block["label"])

    return {
        "input_ids": torch.tensor(ids).long(),
        "label": torch.tensor(labels).float(),
        "attention_mask": torch.ones(len(ids), BLOCK_SIZE).long(),
    }


def download_lm1b(cache_dir):
    dataset = datasets.load_dataset("lm1b", cache_dir=cache_dir)
    preproc_dataset = dataset.map(
        preproc,
        desc="Preprocessing",
        batched=True,
        num_proc=MAX_WORKERS,
        load_from_cache_file=True,
    )

    preproc_dataset = preproc_dataset.select_columns("input_ids")

    chunked_dataset = preproc_dataset.map(
        group_texts,
        desc="Grouping texts",
        batched=True,
        num_proc=MAX_WORKERS,
    )

    chunked_dataset.save_to_disk(cache_dir)


def get_epsilon_lm1b(mode, cache_dir):
    if mode == "validation":
        mode = "test"

    assert mode in ["train", "test"]

    if not all(os.path.exists(os.path.join(cache_dir, split)) for split in SPLITS):
        print("Downloading LM1B dataset to", cache_dir)
        download_lm1b(cache_dir)

    dataset = datasets.load_from_disk(cache_dir)[mode]

    # lazy non-deterministic transformation -> good for training
    final_dataset = dataset.with_transform(_transform)

    return final_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download:
        download_lm1b(args.cache_dir)
        print("Downloaded LM1B dataset to", args.cache_dir)
    else:
        dataset = get_epsilon_lm1b(args.mode, args.cache_dir)

        print(dataset)

        for block in dataset:
            print(block)
            break
