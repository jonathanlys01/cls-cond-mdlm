import os
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import lightning as pl
import numpy as np
import torch
import transformers
from tqdm import tqdm

import utils


N_WORKERS = os.cpu_count() - 1

DEFAULT_BLOCK_SIZE = 256
ATTN_MASK = [1] * DEFAULT_BLOCK_SIZE

# Text 8 dataset


# Copied from dataloader.py (avoiding circular imports)
class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(  # noqa: PLR0913
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        self.characters = list("abcdefghijklmnopqrstuvwxyz ")
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[RESERVED]": 6,
            "[UNK]": 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def tokenize_block(
    stream: str,
    block_size=DEFAULT_BLOCK_SIZE,
    eps_token_id=Text8Tokenizer().pad_token_id,
):
    # Pre-generate blocks
    blocks = []
    current = []

    for token in tqdm(Text8Tokenizer().encode(stream), desc="Pre-generating blocks"):
        current.append(token)

        if len(current) == block_size:
            blocks.append({"text": current})
            current = []

    if current:
        blocks.append({"text": current + [eps_token_id] * (block_size - len(current))})

    return blocks


def get_eps_text8_dataset(cache_dir="data/text8"):
    """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.

    Returns:
      dataset: dataset.DatasetDict, with keys 'train', 'valid', 'test'.
    """
    url = "http://mattmahoney.net/dc/text8.zip"

    split_names = ["train", "validation", "test"]

    if not all(utils.fsspec_exists(os.path.join(cache_dir, split)) for split in split_names):
        raw_cache_dir = os.path.join(cache_dir, "raw_data")
        if not all(utils.fsspec_exists(os.path.join(raw_cache_dir, f"text8.{split}.txt")) for split in split_names):
            if not utils.fsspec_exists(os.path.join(raw_cache_dir, "text8.zip")):
                utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
                with (
                    urllib.request.urlopen(url) as in_stream,
                    open(os.path.join(raw_cache_dir, "text8.zip"), "wb") as out_file,
                ):
                    shutil.copyfileobj(in_stream, out_file)

            with fsspec.open(os.path.join(raw_cache_dir, "text8.zip"), "rb") as f:
                rawdata = zipfile.ZipFile(f).read("text8").decode("utf-8")

            splits = {
                "train": rawdata[:90000000],
                "validation": rawdata[90000000:95000000],
                "test": rawdata[95000000:],
            }

            for split, data in splits.items():
                _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
                with fsspec.open(_path, "w") as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
                with fsspec.open(_path, "r") as f:
                    splits[split] = f.read()

        dataset_dict = {}
        for k, v in splits.items():
            print(f"Processing {k} split")
            ds = tokenize_block(v)

            dataset_dict[k] = datasets.Dataset.from_dict(
                {
                    "input_ids": [block["text"] for block in ds],
                }
            )
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = datasets.load_from_disk(cache_dir)

    return dataset


def _add_epsilon_tokens(
    x,
    eps_token_id=Text8Tokenizer().pad_token_id,
    n_min=50,
    n_max=DEFAULT_BLOCK_SIZE - 50,
    single=False,  # return single block (truncate the rest)
):
    # add epsilon tokens to 1 list of input_ids (1d)
    n_eps = np.random.randint(n_min, n_max + 1)
    np_block = np.array(x)
    p = n_eps / len(np_block)

    p1 = np.random.choice(len(np_block), n_eps, replace=False)
    block1 = np.empty_like(np_block)
    block1[p1] = eps_token_id
    inv_p1 = np.setdiff1d(np.arange(len(np_block)), p1)

    for i, idx in enumerate(inv_p1):
        block1[idx] = np_block[i]

    if single:
        return [{"input_ids": block1.tolist(), "label": p}]

    p2 = np.random.choice(len(np_block), len(np_block) - n_eps, replace=False)
    block2 = np.empty_like(np_block)
    block2[p2] = eps_token_id
    inv_p2 = np.setdiff1d(np.arange(len(np_block)), p2)

    for i, idx in enumerate(inv_p2):
        block2[idx] = np_block[i]

    return [
        {"input_ids": block1.tolist(), "label": p},
        {"input_ids": block2.tolist(), "label": 1 - p},
    ]


def get_transform(
    n_min=50,
    n_max=DEFAULT_BLOCK_SIZE - 50,
    single=False,
):
    def _transform(examples):
        ids = []
        labels = []
        for x in examples["input_ids"]:
            for y in _add_epsilon_tokens(x, n_min=n_min, n_max=n_max, single=single):
                ids.append(y["input_ids"])
                labels.append(y["label"])

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(labels, dtype=torch.float),
            "attention_mask": torch.tensor([ATTN_MASK] * len(ids), dtype=torch.long),
        }

    return _transform


def load_text8_dataset(mode, cache_dir="data/text8", n_min=50, n_max=DEFAULT_BLOCK_SIZE - 50):
    assert mode in ["train", "validation", "test"]
    if not utils.fsspec_exists(cache_dir):
        cache_dir = "data/text8"
    dataset = get_eps_text8_dataset(cache_dir)

    def _add_dummy(x):
        x["attention_mask"] = 0  # dummy attention mask
        x["label"] = 0  # dummy label
        return x

    dataset = dataset.map(_add_dummy, num_proc=min(os.cpu_count(), 8), desc="Adding dummy fields")

    # create only once

    dataset = dataset.with_transform(get_transform(n_min=n_min, n_max=n_max, single=False))

    return dataset[mode]


def decode_without_epsilon(tokens, eps_token_id=Text8Tokenizer().pad_token_id):
    return Text8Tokenizer().decode([t for t in tokens if t != eps_token_id])


# Callback for dynamic rate of epsilon tokens (wrt step or epoch)


def epsilon_scheduler(epoch):
    # 0 to 0.2
    if epoch == 0:
        return get_transform(n_min=0, n_max=DEFAULT_BLOCK_SIZE // 5, single=True)
    # 0.2 to 0.7
    elif epoch == 1:
        return get_transform(n_min=DEFAULT_BLOCK_SIZE // 5, n_max=int(0.7 * DEFAULT_BLOCK_SIZE), single=True)
    else:
        # don't drop second block
        return get_transform(n_min=DEFAULT_BLOCK_SIZE // 5, n_max=4 * DEFAULT_BLOCK_SIZE // 5, single=False)


class EpsilonTransformScheduler(pl.Callback):
    def __init__(self, transform_scheduler):
        """
        Args:
            transform_scheduler (callable): Function to generate new transforms per epoch.
                                            Example: transform_scheduler(epoch) -> transform function.
        """
        self.transform_scheduler = transform_scheduler

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module):
        # Get the current epoch
        current_epoch = trainer.current_epoch

        # Generate a new transform
        new_transform = self.transform_scheduler(current_epoch)
        print(f"Epoch {current_epoch} ended. Applying new transform...")

        # Modify the dataset in-place for all train loaders
        for dataloader in trainer.fit_loop._data_loader_source.dataloader_iterable:
            if hasattr(dataloader, "dataset"):
                dataloader.dataset = dataloader.dataset.with_transform(new_transform)
                print("Transform successfully updated for dataset.")
            else:
                print("No dataset found in dataloader.")


############################################ LM1B Dataset ############################################


if __name__ == "__main__":
    dataset = load_text8_dataset("train")

    for i, row in enumerate(dataset):
        print(decode_without_epsilon(row["input_ids"]))

        break
