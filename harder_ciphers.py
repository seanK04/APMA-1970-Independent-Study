"""Goal 2: make the cipher harder.

Subcommands:
    nospaces     word breaks removed from the message
    typos        random typos in the plaintext before encryption
    homophonic   several cipher symbols per plain symbol (the symbol count is read
                 off the ciphertext, not given to the decoder)

Examples:
    python goal2.py nospaces
    python goal2.py nospaces --from-corpus 1000
    python goal2.py typos --rates 0,0.02,0.05,0.1
    python goal2.py homophonic --from-corpus 1500 --symbols 40 --iters 100000

The message is a file (default messages/hamlet.txt) or, with --from-corpus N, the
last N characters of the training text, which are held out of training.
"""

import argparse
from pathlib import Path

import numpy as np

import homophonic
from alphabet import encode, normalize
from cipher import accuracy, invert_key
from cipher_types import SPACE, add_typos, random_key
from experiments import DEFAULT_MESSAGE, load_training_text, summarize
from ngram import build_model
from sampler_ext import linear_ramp, metropolis
from text_utils import hold_out_tail, letters_only, unigram_probs

def load_texts(args):
    """Return (training_text, message_text), both normalized; the message is held out."""
    raw = load_training_text(args.corpus)
    if args.from_corpus:
        return hold_out_tail(normalize(raw), args.from_corpus)
    message = normalize(args.message.read_text())
    if args.length:
        message = message[:args.length].strip()
    return raw, message


def annealing(args):
    """Inverse-temperature ramp; long texts and many-to-one searches need a cooler start."""
    default_start, default_end = (0.05, 1.0) if args.mode == "homophonic" else (0.5, 3.0)
    start = default_start if args.beta_start is None else args.beta_start
    end = default_end if args.beta_end is None else args.beta_end
    return linear_ramp(start, end)


def permutation_runs(label, plaintext, model, args, rng_seed=0):
    """Encrypt with a random key (space pinned), decode `restarts` times, return result rows."""
    encrypt_key = random_key(np.random.default_rng(rng_seed), fix_space=True)
    ciphertext = encrypt_key[plaintext]
    true_score = model.score(invert_key(encrypt_key)[ciphertext])
    rows = []
    for r in range(args.restarts):
        res = metropolis(ciphertext, model, args.iters, np.random.default_rng([args.seed, r]),
                         schedule=annealing(args), fixed={SPACE: SPACE})
        rows.append({"config": label, "run": r,
                     "accuracy": accuracy(res.best_key[ciphertext], plaintext),
                     "best_score": res.best_score, "true_score": true_score})
    return rows


def run_nospaces(args):
    train, message = load_texts(args)
    plaintext = encode(letters_only(message))
    train_letters = letters_only(train)
    print(f"Message: {len(plaintext)} letters, no spaces   {args.restarts} runs x {args.iters:,} iterations\n")
    for order in (2, 3):
        model = build_model(train_letters, order)
        rows = permutation_runs(f"{'bi' if order == 2 else 'tri'}gram, no spaces", plaintext, model, args)
        print(f"{rows[0]['config']:<22} {summarize(rows)}")


def run_typos(args):
    train, message = load_texts(args)
    clean = encode(normalize(message))
    model = build_model(train, 3)
    print(f"Message: {len(clean)} characters   trigram + known space + annealing   "
          f"{args.restarts} runs x {args.iters:,} iterations\n")
    for rate in [float(x) for x in args.rates.split(",")]:
        plaintext = add_typos(clean, rate, np.random.default_rng(args.seed))
        rows = permutation_runs(f"typo rate {rate:.0%}", plaintext, model, args)
        print(f"{rows[0]['config']:<16} {summarize(rows)}")


def run_homophonic(args):
    train, message = load_texts(args)
    plaintext = encode(normalize(message))
    model = build_model(train, 3)
    probs = unigram_probs(train)

    counts = homophonic.allocate_homophones(probs, args.symbols)
    ciphertext, owner = homophonic.encrypt(plaintext, counts, np.random.default_rng(args.seed))
    observed, ciphertext = homophonic.relabel(ciphertext)
    k = len(observed)
    truth = owner[observed]
    true_score = model.score(truth[ciphertext])

    print(f"Message: {len(plaintext)} characters   cipher symbols in use: {k}   "
          f"{args.restarts} runs x {args.iters:,} iterations\n")
    rows = []
    for r in range(args.restarts):
        res = homophonic.metropolis(ciphertext, k, model, probs, args.iters,
                                    np.random.default_rng([args.seed, r]), schedule=annealing(args))
        rows.append({"config": "homophonic", "run": r,
                     "accuracy": accuracy(res.best_key[ciphertext], plaintext),
                     "best_score": res.best_score, "true_score": true_score})
        print(f"  run {r + 1:>2}: accuracy {rows[-1]['accuracy']:6.1%}   best score {res.best_score:9.1f}"
              f"   (true assignment {true_score:9.1f})")
    print(f"\n{summarize(rows)}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["nospaces", "typos", "homophonic"])
    p.add_argument("--message", type=Path, default=DEFAULT_MESSAGE)
    p.add_argument("--length", type=int, default=None, help="truncate the message file to N characters")
    p.add_argument("--from-corpus", type=int, default=None,
                   help="use the last N characters of the training text as the message (held out)")
    p.add_argument("--corpus", type=Path, default=None, help="training text (default: War and Peace)")
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--iters", type=int, default=None, help="default: 20,000 (150,000 for homophonic)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta-start", type=float, default=None, help="annealing: starting inverse temperature")
    p.add_argument("--beta-end", type=float, default=None, help="annealing: final inverse temperature")
    p.add_argument("--rates", type=str, default="0,0.02,0.05,0.1", help="typos: comma-separated typo rates")
    p.add_argument("--symbols", type=int, default=40, help="homophonic: total cipher symbols")
    return p.parse_args()


def main():
    args = parse_args()
    if args.iters is None:
        args.iters = 150_000 if args.mode == "homophonic" else 20_000
    {"nospaces": run_nospaces, "typos": run_typos, "homophonic": run_homophonic}[args.mode](args)


if __name__ == "__main__":
    main()
