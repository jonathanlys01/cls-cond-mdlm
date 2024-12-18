import argparse
import os
from functools import partial

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from cls_cond.token_utils import chunked_tokenize


DS_NAME = "fancyzhx/amazon_polarity"
CACHE_DIR = os.path.expanduser("~/cls-cond-mdlm/db/amazon-polarity")


CAT_COL = "label"
TITLE_COL = "title"
REVIEW_COL = "content"

BLOCK_SIZE = 1024

CAT_MAP = {
    0: "negative",
    1: "positive",
}


def _merge(example):
    # Remove leading/trailing whitespaces
    title = example[TITLE_COL].strip()
    review = example[REVIEW_COL].strip()

    # Add a period at the end if it's missing
    if title and title[-1] not in [".", "!", "?"]:
        title += "."
    if review and review[-1] not in [".", "!", "?"]:
        review += "."
    # Merge the title and review
    example["text"] = example[TITLE_COL] + " " + example[REVIEW_COL]
    return example


def preprocess(mode):
    """
    mode in ["train", "validation"]
    """
    assert mode in ["train", "validation"]

    if mode == "validation":
        dataset = load_dataset(DS_NAME, cache_dir=CACHE_DIR)["test"]
    else:  # train
        dataset = load_dataset(DS_NAME, cache_dir=CACHE_DIR)["train"]

    dataset = dataset.map(_merge, num_proc=8, remove_columns=[TITLE_COL, REVIEW_COL])

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    tokenized_ds = dataset.map(
        partial(chunked_tokenize, tokenizer=tokenizer, max_len=BLOCK_SIZE), num_proc=8, desc="Tokenizing text"
    ).to_pandas()

    print(dataset.cleanup_cache_files())
    del dataset

    # Mimic the original strategy with grouping by label

    BOS = tokenizer.bos_token_id
    EOS = tokenizer.eos_token_id

    final_input_ids = []
    final_labels = []

    new_block_size = BLOCK_SIZE - 2  # BOS and EOS tokens

    current = {label: [] for label in CAT_MAP}

    for i, row in tqdm(tokenized_ds.iterrows(), total=len(tokenized_ds), desc="Creating blocks"):
        text = row["input_ids"]
        label = row["label"]

        space = new_block_size - len(current[label])

        while len(text) > space:
            current[label].extend(text[:space])

            final_input_ids.append([BOS] + current[label] + [EOS])
            final_labels.append(label)

            current[label] = []
            text = text[space:]
            space = new_block_size
        current[label].extend(text)

    # Pad the last block
    for val, label in current.items():
        to_pad = new_block_size - len(val)
        current[label].extend([tokenizer.pad_token_id] * to_pad)
        final_input_ids.append([BOS] + current[label] + [EOS])
        final_labels.append(label)

    ds = pd.DataFrame({"input_ids": final_input_ids, "label": final_labels})

    output_path = os.path.join(CACHE_DIR, f"amazon-polarity-{mode}.parquet")
    ds.to_parquet(output_path)
    return os.system("du -sh " + output_path)


def get_amazon_polarity(mode):
    input_paths = {mode: os.path.join(CACHE_DIR, f"amazon-polarity-{mode}.parquet") for mode in ["train", "validation"]}

    all_exist = True
    for mode_ in ["train", "validation"]:
        if not os.path.exists(input_paths[mode]):
            print(f"{mode_} dataset not found.")
            all_exist = False
    if not all_exist:
        raise FileNotFoundError("Dataset not found. Please preprocess the dataset first.")

    dataset = load_dataset("parquet", data_files=input_paths)[mode]

    # only create the attn mask once
    attn_mask = [1 for _ in range(BLOCK_SIZE)]

    def _add_att_mask(x):
        x["attention_mask"] = attn_mask
        return x

    dataset = dataset.map(_add_att_mask, num_proc=min(os.cpu_count(), 8), desc="Adding attention mask")

    dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the Amazon polarity dataset for classification.")
    parser.add_argument("--force", action="store_true", help="Force preprocessing even if the dataset already exists.")
    args = parser.parse_args()

    template = "amazon-polarity-{mode}.parquet"
    for mode in ["train", "validation"]:
        if args.force or not os.path.exists(os.path.join(CACHE_DIR, template.format(mode=mode))):
            preprocess(mode)
        else:
            print(f"{mode} dataset already exists. Skipping.")

    train_ds = get_amazon_polarity("train")

    sample = train_ds[0]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(tokenizer.decode(sample["input_ids"]))
    print(CAT_MAP[sample["label"].item()])  # tensor to int
