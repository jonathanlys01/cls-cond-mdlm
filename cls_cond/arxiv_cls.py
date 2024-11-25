import argparse
import os
from functools import partial

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import dataloader
from cls_cond.token_utils import chunked_tokenize


DS_NAME = "ccdv/arxiv-classification"
CACHE_DIR = os.path.expanduser("~/cls-cond-mdlm/db/arxiv-cls")


CAT_COL = "label"
ABS_COL = "text"
BLOCK_SIZE = 1024


CAT_MAP = {
    0: "math.AC",
    1: "cs.CV",
    2: "cs.AI",
    3: "cs.SY",
    4: "math.GR",
    5: "cs.CE",
    6: "cs.PL",
    7: "cs.IT",
    8: "cs.DS",
    9: "cs.NE",
    10: "math.ST",
}


def preprocess(mode) -> int:
    """
    Preprocess the dataset for training or evaluation.
    This function performs the following steps:
    1. Loads the dataset based on the specified mode (train, validation, or test).
    2. Applies a detokenizer to the text data.
    3. Tokenizes the text data using a specified tokenizer.
    4. Groups the tokenized data by label and creates blocks of a specified size.
    5. Pads the last block if necessary.
    6. Saves the preprocessed data to a parquet file.
    Args:
        mode (str): The mode of the dataset to preprocess. Can be "train", "validation", or "test".
    Returns:
        int: The size of the output file in human-readable format.
    """

    def arxiv_detokenizer(x):
        x = dataloader.scientific_papers_detokenizer(x)
        x = x.replace("\n", " ")
        return x

    if mode == "train":
        dataset = load_dataset(DS_NAME, cache_dir=CACHE_DIR)["train"]
    else:
        ds_1 = load_dataset(DS_NAME, cache_dir=CACHE_DIR)["validation"]
        ds_2 = load_dataset(DS_NAME, cache_dir=CACHE_DIR)["test"]
        dataset = concatenate_datasets([ds_1, ds_2])

    dataset = dataset.map(
        lambda x: {
            CAT_COL: x[CAT_COL],  # copy the category
            "text": arxiv_detokenizer(x[ABS_COL]),
        },
        num_proc=8,
        desc="Detokenizing text",
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    tokenized_ds = dataset.map(
        partial(chunked_tokenize,
                tokenizer=tokenizer,
                max_len=BLOCK_SIZE // 2
                ), num_proc=8, desc="Tokenizing text"
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
    for label in current:
        if current[label]:
            to_pad = new_block_size - len(current[label])
            current[label].extend([tokenizer.pad_token_id] * to_pad)
            final_input_ids.append([BOS] + current[label] + [EOS])
            final_labels.append(label)

    ds = pd.DataFrame({"input_ids": final_input_ids, "label": final_labels})

    output_path = os.path.join(CACHE_DIR, f"arxiv-cls-{mode}.parquet")
    ds.to_parquet(output_path)
    return os.system("du -sh " + output_path)


def get_arxiv_cls_categories(mode) -> dict:
    """
    Load the preprocessed arXiv dataset
    Args:
        mode: "train" or "validation"
    Returns:
        dataset: the preprocessed arXiv dataset
    """

    input_paths = {mode: os.path.join(CACHE_DIR, f"arxiv-cls-{mode}.parquet") for mode in ["train", "validation"]}

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
    parser = argparse.ArgumentParser(description="Preprocess the arXiv dataset for classification.")
    parser.add_argument("--force", action="store_true", help="Force preprocessing even if the dataset already exists.")
    args = parser.parse_args()

    template = "arxiv-cls-{mode}.parquet"
    for mode in ["train", "validation"]:
        if os.path.exists(os.path.join(CACHE_DIR, template.format(mode=mode))) and not args.force:
            print(f"Dataset already preprocessed for {mode}.")
        else:
            print(f"Preprocessing {mode} dataset.")
            preprocess(mode)

    print("Preprocessing complete.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("Train dataset")
    train_ds = get_arxiv_cls_categories("train")
    print(len(train_ds))
    ids = train_ds[0]["input_ids"]
    print(len(ids))
    print(tokenizer.decode(ids))

    print("Validation dataset")
    valid_ds = get_arxiv_cls_categories("validation")
    print(len(valid_ds))
    ids = valid_ds[0]["input_ids"]
    print(len(ids))
    print(tokenizer.decode(ids))
