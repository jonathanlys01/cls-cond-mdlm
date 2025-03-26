"""
Grammar module for generating sequences that satisfy certain grammar rules
"""

import os
import random
from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache, partial
from multiprocessing import Pool

import pandas as pd
import torch
from datasets import Dataset
from pandas import DataFrame
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from utils import get_logger


logger = get_logger(__name__)


# Adjust the number of workers to the number of available cores
N_WORKERS = 8

CARDINAL_MAP = {
    "balanced_parentheses": {"train": 5_000_000, "validation": 10_000},
    "parity": {"train": 500_000, "validation": 10_000},
    "alternating_ab": {"train": 100_000, "validation": 10_000},
    "balanced_ab": {"train": 200_000, "validation": 10_000},
    "palindrome": {"train": 5_000_000, "validation": 10_000},
}


############################################ Tokenizer ############################################

OFFSET = 8  # offset for special tokens


class CharTokenizer(PreTrainedTokenizer):
    def __init__(  # noqa: PLR0913
        self,
        mapping: dict[str, int],
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[RESERVED]": 6,
            "[UNK]": 7,
            **mapping,
        }
        assert self._vocab_str_to_int["[EPS]"] == len(self._vocab_str_to_int) - 1, "EPS token must be last"
        del self._vocab_str_to_int["[EPS]"]  # remove EPS token (will be added later)

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

        self.add_special_tokens({"additional_special_tokens": ["[EPS]"]})

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab_str_to_int


############################################ Grammar ############################################


class Grammar(ABC):
    def __init__(self):
        self.mapping = {}
        self.rev_mapping = {}

        # most datasets rely on a symetric behavior of tokens and are thus parity sensitive
        self.parity_sensitive = True

        # metrics
        self.reset_metrics()

    def __post_init__(self):
        self.mapping = {**self.mapping, "[EPS]": len(self.mapping)}
        self.mapping = {k: v + OFFSET for k, v in self.mapping.items()}  # add offset for special tokens

        self.rev_mapping = {v: k for k, v in self.mapping.items()}

        self.tokenizer = CharTokenizer(self.mapping)

    def decode(self, sequence: list[int], ignore_eps: bool = False) -> str:
        if ignore_eps:
            sequence = [i for i in sequence if self.tokenizer.convert_ids_to_tokens(i) != "[EPS]"]
        return self.tokenizer.decode(sequence, skip_special_tokens=False)

    @abstractmethod
    def _generate_str(self, seq_len: int) -> str:
        """Generates a sequence of grammar symbols as a string"""
        pass

    @abstractmethod
    def generate(self, seq_len: int) -> list[int]:
        """Generates a sequence of integers that represents the grammar"""
        pass

    def remove_eps(self, sequence: list[int]) -> list[int]:
        return [i for i in sequence if self.tokenizer.convert_ids_to_tokens(i) != "[EPS]"]

    @abstractmethod
    def evaluate(self, sequence: list[int]) -> bool:
        """Evaluates whether the sequence satisfies the grammar"""
        pass

    @abstractmethod
    def add_eval(self, sequence: list[int]) -> None:
        pass

    def generate_metrics(self) -> Counter:
        return self.eval_counter

    def reset_metrics(self) -> None:
        print(f"Resetting metrics for {self.__class__.__name__}")
        self.eval_counter = Counter()
        self.eval_counter["HIT"] = 0
        self.eval_counter["MISS"] = 0


############################################ Grammar Implementations ############################################


class BalancedParentheses(Grammar):
    """
    Grammar for balanced parentheses
    """

    def __init__(self):
        super().__init__()
        self.mapping = {"(": 0, ")": 1}
        super().__post_init__()

    def generate_recursive(self, n: int) -> list[int]:
        if n == 0:
            return ""
        elif n == 1:
            return "()"
        else:
            parts = []
            remaining = n
            while remaining > 0:
                k = random.randint(1, remaining)
                parts.append("(" + self.generate_recursive(k - 1) + ")")
                remaining -= k
            random.shuffle(parts)
            return "".join(parts)

    def _generate_str(self, seq_len: int) -> str:
        return self.generate_recursive(seq_len // 2)

    def generate(self, seq_len) -> list[int]:
        seq = self._generate_str(seq_len)
        return [self.mapping[c] for c in seq]

    def evaluate(self, sequence: list[int]) -> bool:
        sequence = self.remove_eps(sequence)
        stack = []
        for token in sequence:
            if token == self.mapping["("]:
                stack.append(token)
            elif token == self.mapping[")"]:
                if len(stack) == 0:
                    return False
                stack.pop()
        return len(stack) == 0

    def add_eval(self, sequence: list[int]) -> None:
        pred = self.evaluate(sequence)
        self.eval_counter["HIT" if pred else "MISS"] += 1
        # format: {"HIT": 100, "MISS": 200}


class Parity(Grammar):
    """
    Grammar for even number of As"
    """

    def __init__(self):
        super().__init__()
        self.mapping = {"A": 0, "B": 1}
        super().__post_init__()

    def _generate_str(self, seq_len: int) -> str:
        int_seq = self.generate(seq_len)
        return "".join([self.rev_mapping[i] for i in int_seq])

    def generate(self, seq_len: int) -> list[int]:
        n_zeros = random.randint(0, seq_len // 2) * 2  # even number of zeros
        n_ones = seq_len - n_zeros

        seq = [self.mapping["A"]] * n_zeros + [self.mapping["B"]] * n_ones
        random.shuffle(seq)
        return seq

    def evaluate(self, sequence: list[int]) -> bool:
        sequence = self.remove_eps(sequence)
        return sequence.count(self.mapping["A"]) % 2 == 0

    def add_eval(self, sequence: list[int]) -> None:
        count = sequence.count(self.mapping["A"])
        self.eval_counter[count] += 1
        self.eval_counter["HIT" if count % 2 == 0 else "MISS"] += 1
        # format: {0: 100, 1: 200, 2: 300, "HIT": 400, "MISS": 500}


class AlternatingAB(Grammar):
    """
    Grammar for alternating AB"
    """

    def __init__(self):
        super().__init__()
        self.mapping = {"A": 0, "B": 1}
        self.parity_sensitive = False  # not parity sensitive
        super().__post_init__()

    def _generate_str(self, seq_len: int) -> str:
        seq_int = self.generate(seq_len)
        return "".join([self.rev_mapping[i] for i in seq_int])

    @lru_cache(maxsize=None)  # elts in the cache: EPS_RATE * seq_len
    def _cached_generate(self, seq_len: int) -> list[int]:
        n = seq_len + 1
        return [self.mapping["A"] if i % 2 == 0 else self.mapping["B"] for i in range(n)]  # [A, B, A, B, ...]

    def generate(self, seq_len: int) -> list[int]:
        seq = self._cached_generate(seq_len)

        start = random.randint(0, 1)
        return seq[start : start + seq_len]

    def evaluate(self, sequence: list[int]) -> bool:
        sequence = self.remove_eps(sequence)
        return all(sequence[i] != sequence[i + 1] for i in range(len(sequence) - 1))

    def add_eval(self, sequence: list[int]) -> None:
        pred = self.evaluate(sequence)
        sequence = self.remove_eps(sequence)
        start = self.tokenizer.convert_ids_to_tokens(sequence[0])

        self.eval_counter["HIT" if pred else "MISS"] += 1
        self.eval_counter[start] += 1
        # format: {"HIT": 100, "MISS": 200, "A": 300, "B": 400}


class BalancedAB(Grammar):
    """ "
    Grammar for balanced AB"
    """

    def __init__(self):
        super().__init__()
        self.mapping = {"A": 0, "B": 1}
        super().__post_init__()

    def _generate_str(self, seq_len: int) -> str:
        seq_int = self.generate(seq_len)
        return "".join([self.rev_mapping[i] for i in seq_int])

    @lru_cache(maxsize=None)
    def _cached_generate(self, seq_len: int) -> list[int]:
        n = seq_len // 2
        seq = [self.mapping["A"]] * n + [self.mapping["B"]] * n
        return seq

    def generate(self, seq_len: int) -> list[int]:
        assert seq_len % 2 == 0, "Sequence length must be even"
        seq = self._cached_generate(seq_len)
        random.shuffle(seq)
        return seq

    def evaluate(self, sequence: list[int]) -> bool:
        sequence = self.remove_eps(sequence)
        return sequence.count(self.mapping["A"]) == len(sequence) // 2

    def add_eval(self, sequence: list[int]) -> None:
        pred = self.evaluate(sequence)
        count = sequence.count(self.mapping["A"])
        self.eval_counter["HIT" if pred else "MISS"] += 1
        self.eval_counter[str(count)] += 1
        # format: {0: 100, 1: 200, 2: 300, "HIT": 400, "MISS": 500}


class Palindrome(Grammar):
    """
    Grammar for palindromes
    """

    def __init__(self):
        super().__init__()
        self.letters = list("abcde")  # only 5 letters to reduce complexity
        self.mapping = {char: i for i, char in enumerate(self.letters)}
        super().__post_init__()

    def _generate_str(self, seq_len: int) -> str:
        seq = random.choices(self.letters, k=seq_len // 2)
        return "".join(seq + seq[::-1])

    def generate(self, seq_len: int) -> list[int]:
        seq = self._generate_str(seq_len)
        return [self.mapping[c] for c in seq]

    def evaluate(self, sequence: list[int]) -> bool:
        sequence = self.remove_eps(sequence)
        if len(sequence) % 2 != 0:
            return False
        return sequence[: len(sequence) // 2] == sequence[len(sequence) // 2 :][::-1]

    def _fine_grained_eval(self, sequence: list[int]) -> float:
        sequence = self.remove_eps(sequence)
        if len(sequence) % 2 == 0:
            # even number of letters
            first_half = sequence[: len(sequence) // 2]
            second_half = sequence[len(sequence) // 2 :]
            return sum([1 for i, j in zip(first_half, second_half) if i == j]) / len(first_half)
        else:
            return -1

    def add_eval(self, sequence: list[int]) -> None:
        pred = self.evaluate(sequence)
        self.eval_counter["HIT" if pred else "MISS"] += 1
        self.eval_counter[str(self._fine_grained_eval(sequence))] += 1
        # format: {"HIT": 100, "MISS": 200}


############################################ Dataset Generation ############################################


def merge_eps_seq(
    text_seq: list[int],
    n_epsilon: int,
    epsilon_idx: int,
) -> tuple[list[int], float]:
    """Inserts epsilon tokens at random positions in the sequence."""

    N = len(text_seq) + n_epsilon

    eps_indices = set(random.sample(range(N), n_epsilon))
    new_seq = []
    seq_iter = iter(text_seq)

    for i in range(N):
        new_seq.append(epsilon_idx if i in eps_indices else next(seq_iter))

    return new_seq, n_epsilon / len(new_seq)


def _generate_sample(i, grammar: Grammar, final_seq_len: int, max_epsilon: int) -> dict:
    if max_epsilon == 0:
        seq = grammar.generate(final_seq_len)
        seq_eps, rate = seq, 0.0
    else:
        random.seed(i)  # map i to a seed for sequence "uniqueness"
        if grammar.parity_sensitive:
            n_epsilon = random.randint(0, max_epsilon // 2) * 2  # even number of epsilons
        else:
            n_epsilon = random.randint(0, max_epsilon)

        seq = grammar.generate(final_seq_len - n_epsilon)
        seq_eps, rate = merge_eps_seq(seq, n_epsilon, grammar.mapping["[EPS]"])

    return {"input_ids": seq_eps, "label": rate, "attention_mask": 1}  # dummy attention mask


def _generate_sample_fixed(i, grammar: Grammar, final_seq_len: int, max_epsilon: int) -> dict:
    if max_epsilon == 0:
        seq = grammar.generate(final_seq_len)
        seq_eps, rate = seq, 0.0
    else:
        random.seed(i)
        seq = grammar.generate(final_seq_len - max_epsilon)
        seq_eps, rate = merge_eps_seq(seq, max_epsilon, grammar.mapping["[EPS]"])

    return {"input_ids": seq_eps, "label": rate, "attention_mask": 1}  # dummy attention mask


def generate_dataset(grammar: Grammar, n_samples: int, seq_len: int, n_epsilon: int, offset: int) -> DataFrame:
    """
    Generates a dataset of sequences that satisfy the grammar rules
    """

    print(
        f"Generating {n_samples} samples of length {seq_len} with max {n_epsilon} \
epsilon tokens with grammar {grammar.__class__.__name__}"
    )

    """gen_fn = partial(
        _generate_sample,
        grammar=grammar,
        final_seq_len=seq_len,
        max_epsilon=n_epsilon,
    )"""

    gen_fn = partial(
        _generate_sample_fixed,
        grammar=grammar,
        final_seq_len=seq_len,
        max_epsilon=n_epsilon,
    )

    with Pool(N_WORKERS) as pool:
        samples = list(tqdm(pool.imap(gen_fn, range(offset, offset + n_samples)), total=n_samples))

    return DataFrame(samples)


############################################ Dataset Loading ############################################


def _transform(examples, block_size):
    ids = []
    labels = []

    # examples is a dict with keys: input_ids, label, attention_mask (lists inside)

    for i in range(len(examples["input_ids"])):
        ids.append(examples["input_ids"][i])
        labels.append(examples["label"][i])

    return {
        "input_ids": torch.tensor(ids).long(),
        "label": torch.tensor(labels).float(),
        "attention_mask": torch.ones(len(ids), block_size).long(),
    }


CLS_MAP = {
    "balanced_parentheses": BalancedParentheses,
    "parity": Parity,
    "alternating_ab": AlternatingAB,
    "balanced_ab": BalancedAB,
    "palindrome": Palindrome,
}


def get_grammar_dataset(name: str, block_size: int, mode: str, cache_dir: str, max_eps_rate: float) -> DataFrame:
    assert 0 <= max_eps_rate <= 1, f"max_eps_rate must be in [0, 1], got {max_eps_rate}"

    name = name.removeprefix("grammar_")

    grammar = CLS_MAP[name]()

    # if os.path.isfile(os.path.join(cache_dir, f"{name}_{mode}.parquet")):
    if False:  # temp disable cache
        print(f"Loading {name}_{mode}.parquet from cache")
        dataset = pd.read_parquet(os.path.join(cache_dir, f"{name}_{mode}.parquet"))

    else:
        n_samples = CARDINAL_MAP[name][mode]
        add_offset = "validation" in mode  # add train len to validation offset
        offset = CARDINAL_MAP[name]["train"] if add_offset else 0

        print(
            f"Generating a total of {int(max_eps_rate * block_size)} epsilon tokens \
(resp. {block_size - int(max_eps_rate * block_size)} non-epsilon tokens)"
        )

        dataset = generate_dataset(grammar, n_samples, block_size, int(max_eps_rate * block_size), offset)
        os.makedirs(cache_dir, exist_ok=True)
        dataset.to_parquet(os.path.join(cache_dir, f"{name}_{mode}.parquet"))

    # log first example
    logger.info(f"First example in {name}_{mode}.parquet:")
    logger.info(dataset.iloc[0].to_dict())

    dataset = Dataset.from_pandas(dataset)

    dataset = dataset.with_transform(partial(_transform, block_size=block_size))

    return dataset


def get_grammar_tokenizer(name: str) -> CharTokenizer:
    name = name.removeprefix("grammar_")

    grammar: Grammar = CLS_MAP[name]()
    return grammar.tokenizer


############################################ Main ############################################


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", "-g", type=str, default="balanced_parentheses")
    args = parser.parse_args()

    if args.grammar == "balanced_parentheses":
        grammar = BalancedParentheses()
    elif args.grammar == "parity":
        grammar = Parity()
    elif args.grammar == "alternating_ab":
        grammar = AlternatingAB()
    elif args.grammar == "balanced_ab":
        grammar = BalancedAB()
    elif args.grammar == "palindrome":
        grammar = Palindrome()
    else:
        raise ValueError(f"Grammar {args.grammar} not implemented")

    i = input("seq")

    print(grammar.evaluate([grammar.mapping[c] for c in i]))


if __name__ == "__main__":
    main()
