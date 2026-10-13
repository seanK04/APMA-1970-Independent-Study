"""Crack every example in messages/examples/ and print a readable report.

Each example is a pair of files: N.cipher.txt (the scrambled text) and N.txt
(the answer, used only for grading). Each example gets several independent runs;
the decryption shown is the one with the highest plausibility (the sampler never
sees the answer).

    python run_examples.py                          # 10 runs x 20,000 iterations each
    python run_examples.py --runs 20 --iters 50000 --power 0.5
"""

import argparse
import textwrap
from pathlib import Path

import numpy as np

from alphabet import decode
from cipher import accuracy, apply_key
from corpus import load_transition_matrix
from main import read_symbols
from sampler import metropolis

EXAMPLES_DIR = Path(__file__).parent / "messages" / "examples"
WIDTH = 90


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=10, help="independent runs (seeds) per example")
    p.add_argument("--iters", type=int, default=20_000, help="Metropolis iterations per run")
    p.add_argument("--power", type=float, default=1.0, help="acceptance power p (1 = the paper's algorithm)")
    p.add_argument("--seed", type=int, default=0, help="base random seed; run r uses seed (seed, r)")
    return p.parse_args()


def show(label, text):
    """Print `text` wrapped to WIDTH, with `label` in a left-hand column."""
    print(textwrap.fill(text, WIDTH, initial_indent=f"  {label:<11}", subsequent_indent=" " * 13))


def crack(ciphertext, answer, log_M, args):
    """Run the sampler `args.runs` times. Return the most plausible decryption,
    its accuracy, and every run's accuracy."""
    results = [metropolis(ciphertext, log_M, args.iters, np.random.default_rng([args.seed, r]),
                          power=args.power)
               for r in range(args.runs)]
    accs = [accuracy(apply_key(res.best_key, ciphertext), answer) for res in results]
    best = int(np.argmax([res.best_log_pl for res in results]))
    return apply_key(results[best].best_key, ciphertext), accs[best], accs


def main():
    args = parse_args()
    log_M = np.log(load_transition_matrix())
    print(f"{args.runs} runs x {args.iters:,} iterations per example (power p = {args.power:g})")

    summary = []
    for cipher_path in sorted(EXAMPLES_DIR.glob("*.cipher.txt")):
        name = cipher_path.name.removesuffix(".cipher.txt")
        answer_path = cipher_path.with_name(f"{name}.txt")
        decoded, best_acc, accs = crack(read_symbols(cipher_path, None),
                                        read_symbols(answer_path, None), log_M, args)

        print(f"\n=== Example {name} " + "=" * (WIDTH - 13 - len(name)))
        show("Scrambled", cipher_path.read_text().strip())
        show("Answer", answer_path.read_text().strip())
        show("Decrypted", decode(decoded).lower())
        show("Accuracy", f"{best_acc:.1%} (most plausible run); per run: "
                         + " ".join(f"{a:.0%}" for a in accs))
        summary.append((name, best_acc, accs))

    print(f"\n=== Summary " + "=" * (WIDTH - 12))
    print(f"  {'Example':<9}{'Shown':>8}{'Mean':>8}{'Min':>8}{'Max':>8}")
    for name, best_acc, accs in summary:
        print(f"  {name:<9}{best_acc:>8.1%}{np.mean(accs):>8.1%}{min(accs):>8.1%}{max(accs):>8.1%}")


if __name__ == "__main__":
    main()
