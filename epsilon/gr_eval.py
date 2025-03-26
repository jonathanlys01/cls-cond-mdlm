"""
Evaluation module for grammar-constrained sequence generation.
"""

import time
from collections import Counter

import matplotlib.pyplot as plt

from epsilon.grammar import CLS_MAP, Grammar


def _generate_plot(counter: Counter, title: str, grammar_name: str) -> float:
    # returns accuracy

    if len(counter) == 2:  # noqa: PLR2004
        print(counter)

        plt.bar(counter.keys(), counter.values())
        acc = counter["HIT"] / (counter["HIT"] + counter["MISS"])
        plt.title(title + f" (acc: {acc:.4f})")
        plt.savefig(f"{grammar_name}.png")
        plt.close()

    else:
        acc = counter["HIT"] / (counter["HIT"] + counter["MISS"])
        plt.subplot(1, 2, 1)
        plt.bar(["HIT", "MISS"], [counter["HIT"], counter["MISS"]])
        plt.title(f"Accuracy: {acc:.4f}")

        other = {k: v for k, v in counter.items() if k not in ["HIT", "MISS"]}

        print(other)

        # check if str are castable to float
        if all(isinstance(k, str) for k in other.keys()):
            temp = {
                float(k): v
                for k, v in other.items()
                if k.replace(".", "", 1).isdigit() or k.replace("-", "", 1).isdigit()
            }
            if len(temp) > 0:
                other = temp  # else, do nothing
        else:
            other = {float(k): v for k, v in other.items() if isinstance(k, (int, float))}

        other = dict(sorted(other.items()))

        if isinstance(list(other.keys())[0], (float, int)) and list(other.keys())[0] < 0:
            neg = {k: v for k, v in other.items() if k < 0}
            other = {k: v for k, v in other.items() if k >= 0}

            print(neg)

        if all(isinstance(k, float) for k in other.keys()):
            avg = sum(other.keys()) / len(other.keys())
            postfix = f" (avg: {avg:.4f})"
        else:
            postfix = ""

        title = title + postfix

        plt.subplot(1, 2, 2)
        plt.bar(other.keys(), other.values())
        plt.title(title)

        plt.savefig(f"{grammar_name}.png")
        plt.close()

    return acc


def _gen_stats(counter: Counter) -> dict:
    stats = {}
    stats["HIT"] = counter["HIT"]
    stats["MISS"] = counter["MISS"]
    stats["ACCURACY"] = counter["HIT"] / (counter["HIT"] + counter["MISS"])
    stats["TOTAL"] = sum(counter.values())
    return stats


def count_unique(seqs: list[str]) -> int:
    return len(set(seqs))


def _dump_count(title, count: int, acc: float) -> None:
    with open(f"{title}.txt", "w") as f:
        f.write(f"Unique count: {count:,}\n accuracy: {acc:.4f}\n")


def grammar_eval(
    sequences: list[str],
    data_train_name: str,
    has_eps: bool,
) -> None:
    """
    Evaluate a list of sequences according to a given grammar.

    Parameters
    ----------
    sequences : list[str] or list[int] or torch.Tensor or np.ndarray
        List of sequences to evaluate.
    data_train_name : str
        Name of the dataset used for training the grammar. (maps to a grammar class)
    """

    name = data_train_name.removeprefix("grammar_")
    grammar: Grammar = CLS_MAP[name]()

    count = count_unique(sequences)
    sequences_ = [grammar.tokenizer.encode(seq) for seq in sequences]

    grammar.reset_metrics()

    # Compute evaluation metrics
    for seq in sequences_:
        grammar.add_eval(seq)

    # Get evaluation metrics (counter)
    metrics = grammar.eval_counter

    # Generate plot
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    title = f"{grammar.__class__.__name__}_{['', 'eps'][has_eps]}{now}"

    acc = _generate_plot(metrics, "Evaluation", title)
    _dump_count(title, count, acc)
    stats = _gen_stats(metrics)

    print(f"Grammar: {grammar.__class__.__name__}")
    print(f"Total: {stats['TOTAL']}")
    print(f"Hit:   {stats['HIT']}")
    print(f"Miss:  {stats['MISS']}")
    print(f"Accuracy: {stats['ACCURACY']:.2f}")
    print()
