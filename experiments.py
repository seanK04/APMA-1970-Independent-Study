"""Ablation and text-length experiments for the substitution-cipher decoder.

A configuration is a "+"-joined list of options:
    trigram   score with a trigram model instead of a bigram model
    space     the space symbol is known (encrypted to itself and pinned in the search)
    anneal    raise beta from 0.5 to 3 over the run instead of holding it at 1

Examples:
    python experiments.py ablation
    python experiments.py ablation --configs bigram,trigram+space+anneal --restarts 20
    python experiments.py length-curve --config trigram+space+anneal --lengths 25,50,100,150,200
    python experiments.py ablation --messages messages/a.txt messages/b.txt --cipher atbash
    python experiments.py ablation --corpus path/to/training_text.txt

Each run uses an independent random starting key; the encryption key depends only
on --seed, so every configuration of the same type attacks the same message.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from alphabet import encode, normalize
from cipher import accuracy, invert_key
from cipher_types import SPACE, atbash_key, random_key
from ngram import build_model
from sampler_ext import constant, linear_ramp, metropolis

DEFAULT_MESSAGE = Path(__file__).parent / "messages" / "hamlet.txt"
DEFAULT_CONFIGS = ["bigram", "trigram", "bigram+space", "bigram+anneal", "trigram+space+anneal"]
ANNEAL_START, ANNEAL_END = 0.5, 3.0


def parse_config(name: str):
    options = set(name.split("+"))
    unknown = options - {"bigram", "trigram", "space", "anneal"}
    if unknown:
        raise ValueError(f"unknown option(s) {sorted(unknown)} in config '{name}'")
    order = 3 if "trigram" in options else 2
    return order, "space" in options, "anneal" in options


def load_training_text(corpus_path):
    """Training text: a user-supplied file, or War and Peace via corpus.py."""
    if corpus_path:
        return Path(corpus_path).read_text(encoding="utf-8", errors="ignore")
    from corpus import download_book, strip_gutenberg
    return strip_gutenberg(download_book())


def load_message(paths, length):
    """Normalize each file, join them with spaces, optionally truncate."""
    text = " ".join(normalize(Path(p).read_text()) for p in paths)
    if length is not None:
        text = text[:length].strip()
    return encode(text)


def run_config(config, plaintext, models, args):
    """Run all restarts of one configuration; return a list of per-run dicts."""
    order, fix_space, anneal = parse_config(config)
    model = models[order]

    if args.cipher == "atbash":
        encrypt_key = atbash_key()
    else:
        encrypt_key = random_key(np.random.default_rng(args.seed), fix_space)
    ciphertext = encrypt_key[plaintext]
    true_score = model.score(invert_key(encrypt_key)[ciphertext])

    schedule = linear_ramp(ANNEAL_START, ANNEAL_END) if anneal else constant(1.0)
    fixed = {SPACE: SPACE} if fix_space else None

    rows = []
    for r in range(args.restarts):
        rng = np.random.default_rng([args.seed, r])
        res = metropolis(ciphertext, model, args.iters, rng, schedule=schedule, fixed=fixed)
        rows.append({
            "config": config, "length": len(plaintext), "run": r,
            "accuracy": accuracy(res.best_key[ciphertext], plaintext),
            "best_score": res.best_score, "true_score": true_score,
            "accept_rate": res.accept_rate,
        })
    return rows


def summarize(rows):
    """One-line summary of a configuration's runs."""
    acc = np.array([r["accuracy"] for r in rows])
    top = max(rows, key=lambda r: r["best_score"])
    solved = sum(r["best_score"] >= r["true_score"] - 1e-9 for r in rows)
    return (f"mean {acc.mean():6.1%}  min {acc.min():6.1%}  max {acc.max():6.1%}  "
            f"most-plausible run {top['accuracy']:6.1%}  "
            f"search reached true-key score {solved}/{len(rows)}")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-run results saved to {path}")


def plot_length_curve(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = sorted({r["length"] for r in rows})
    mean, best = [], []
    for n in lengths:
        acc = [r["accuracy"] for r in rows if r["length"] == n]
        top = max((r for r in rows if r["length"] == n), key=lambda r: r["best_score"])
        mean.append(np.mean(acc))
        best.append(top["accuracy"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lengths, mean, marker="o", label="mean over runs")
    ax.plot(lengths, best, marker="o", label="most plausible run")
    ax.set_xlabel("message length (characters)")
    ax.set_ylabel("fraction of characters correct")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["ablation", "length-curve"])
    p.add_argument("--messages", type=Path, nargs="+", default=[DEFAULT_MESSAGE],
                   help="one or more message files; several are pooled into one text")
    p.add_argument("--corpus", type=Path, default=None, help="training text (default: War and Peace)")
    p.add_argument("--cipher", choices=["random", "atbash"], default="random")
    p.add_argument("--iters", type=int, default=20_000)
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS), help="ablation: comma-separated")
    p.add_argument("--config", type=str, default="trigram+space+anneal", help="length-curve: single config")
    p.add_argument("--lengths", type=str, default="25,50,100,150,200", help="length-curve: comma-separated")
    p.add_argument("--csv", type=Path, default=None, help="save per-run results here")
    p.add_argument("--plot", type=Path, default=None, help="length-curve: save the plot here")
    return p.parse_args()


def main():
    args = parse_args()
    training_text = load_training_text(args.corpus)
    models = {2: build_model(training_text, 2), 3: build_model(training_text, 3)}

    all_rows = []
    if args.mode == "ablation":
        plaintext = load_message(args.messages, None)
        print(f"Message: {len(plaintext)} characters   cipher: {args.cipher}   "
              f"{args.restarts} runs x {args.iters:,} iterations\n")
        for config in args.configs.split(","):
            rows = run_config(config, plaintext, models, args)
            all_rows += rows
            print(f"{config:<24} {summarize(rows)}")
    else:
        lengths = [int(x) for x in args.lengths.split(",")]
        print(f"Config: {args.config}   cipher: {args.cipher}   "
              f"{args.restarts} runs x {args.iters:,} iterations\n")
        for n in lengths:
            plaintext = load_message(args.messages, n)
            rows = run_config(args.config, plaintext, models, args)
            all_rows += rows
            print(f"length {len(plaintext):>4}   {summarize(rows)}")
        if args.plot:
            plot_length_curve(all_rows, args.plot)

    if args.csv:
        write_csv(all_rows, args.csv)


if __name__ == "__main__":
    main()
