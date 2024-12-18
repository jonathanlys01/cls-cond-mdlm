import argparse
import json
import os
from functools import partial

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import dataloader
from cls_cond.token_utils import chunked_tokenize


DS_NAME = "gfissore/arxiv-abstracts-2021"
CACHE_DIR = os.path.expanduser("~/cls-cond-mdlm/db/arxiv-abs")
CAT_DIR = os.path.expanduser("~/cls-cond-mdlm/cls_cond/categories.json")

CAT_COL = "categories"
ABS_COL = "abstract"
BLOCK_SIZE = 1024

# Default number of classes to keep
DEFAULT_TOPK = 20


def gen_categories():
    """
    Generate the categories from the dataset
    Save them to a file
    """
    dataset = load_dataset(DS_NAME, cache_dir=CACHE_DIR)["train"]
    categories = set()
    for row in dataset:
        cats = row[CAT_COL][0].split()
        for cat in cats:
            categories.add(cat)
    with open(CAT_DIR, "w") as f:
        json.dump(list(categories), f, indent=4)


def preprocess() -> int:
    """
    Preprocesses the arXiv dataset by performing the following steps:
    1. Loads the category mapping from the JSON file.
    2. Defines helper functions to get categories, map rows to categories, and detokenize text.
    3. Loads the dataset and selects relevant columns.
    4. Maps categories to the dataset and filters out rows without categories.
    5. Detokenizes the abstracts in the dataset.
    6. Initializes a tokenizer and configures padding and truncation settings.
    7. Tokenizes the abstracts in the dataset.
    8. Cleans up the dataset cache files.
    9. Groups tokenized text by label and creates blocks of input IDs.
    10. Pads the last block of each label.
    11. Converts the processed data into a pandas DataFrame.
    12. Saves the DataFrame to a Parquet file.
    13. Prints the disk usage of the saved Parquet file.
    """

    assert os.path.exists(CAT_DIR), "Categories file not found"
    with open(CAT_DIR, "r") as f:
        ALL_CAT = json.load(f)

    def map_to_category(row):
        def _get_category(text_seq):
            text = text_seq[0]
            categories = text.split()
            for cat in categories:
                if cat in ALL_CAT:
                    return cat
            return None

        row[CAT_COL] = _get_category(row[CAT_COL])
        return row

    def arxiv_detokenizer(x):
        x = dataloader.scientific_papers_detokenizer(x)
        # Remove newlines (randomly because of the parsing script)
        x = x.replace("\n", " ")
        return x

    # about 1.5 GB
    dataset = load_dataset(DS_NAME, cache_dir=CACHE_DIR)["train"]

    dataset = (
        dataset.select_columns([CAT_COL, ABS_COL])
        .map(map_to_category, num_proc=8, desc="Mapping categories")
        .filter(lambda x: x[CAT_COL] is not None, num_proc=8, desc="Filtering categories")
        .map(
            lambda x: {"label": x[CAT_COL], "text": arxiv_detokenizer(x[ABS_COL])},
            num_proc=8,
            remove_columns=[CAT_COL, ABS_COL],
            desc="Detokenizing abstracts",
        )
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    tokenized_ds = dataset.map(
        partial(chunked_tokenize, tokenizer=tokenizer, max_len=BLOCK_SIZE // 2), num_proc=8, desc="Tokenizing abstracts"
    ).to_pandas()

    dataset.cleanup_cache_files()
    del dataset

    # Mimic the original strategy with grouping by label

    BOS = tokenizer.bos_token_id
    EOS = tokenizer.eos_token_id

    final_input_ids = []
    final_labels = []

    new_block_size = BLOCK_SIZE - 2  # BOS and EOS tokens

    current = {label: [] for label in tokenized_ds["label"].unique()}

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

    output_path = os.path.join(CACHE_DIR, "arxiv-abs.parquet")
    ds.to_parquet(output_path)
    return os.system("du -sh " + output_path)


def get_arxiv_abs(mode, top_k=DEFAULT_TOPK) -> Dataset:
    """
    Load the preprocessed arXiv dataset
    Args:
        mode: "train" or "eval" (split is approx 95/5)
        top_k: number of classes to keep, ordered by frequency (-1 keeps all)
    Returns:
        dataset: the preprocessed arXiv dataset
    """

    input_path = os.path.join(CACHE_DIR, "arxiv-abs.parquet")
    if not os.path.exists(input_path):
        raise FileNotFoundError("Dataset not found. Please preprocess the dataset first.")

    dataset = load_dataset("parquet", data_files=input_path)["train"]
    len_ds = len(dataset)

    if mode == "train":
        dataset = dataset.select(range(int(len_ds * 0.95)))
    elif mode == "validation":
        dataset = dataset.select(range(int(len_ds * 0.95), len_ds))
    else:
        raise ValueError("Invalid mode. Please choose 'train' or 'validation'")

    assert os.path.exists(CAT_DIR), "Categories file not found"
    with open(CAT_DIR, "r") as f:
        ALL_CAT = json.load(f)

    top_cats = sorted(ALL_CAT, key=ALL_CAT.get, reverse=True)[:top_k]
    top_cats = {top_cats[i]: i for i in range(len(top_cats))}

    dataset = dataset.filter(lambda x: x["label"] in top_cats, num_proc=min(os.cpu_count(), 8))

    # create only one attention mask for the whole block
    attn_mask = [1 for _ in range(BLOCK_SIZE)]

    def _add_att_mask(x):
        x["attention_mask"] = attn_mask
        x["label"] = top_cats[x["label"]]

        return x

    dataset = dataset.map(_add_att_mask, num_proc=min(os.cpu_count(), 8), desc="Adding attention mask + label encoding")

    dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the arXiv dataset")
    parser.add_argument("--gen-categories", action="store_true", help="Generate categories")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the dataset")
    args = parser.parse_args()

    if args.gen_categories:
        gen_categories()
    if args.preprocess:
        preprocess()
