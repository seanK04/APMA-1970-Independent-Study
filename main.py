"""Crack a substitution cipher with MCMC, after Diaconis, "The Markov Chain Monte
Carlo Revolution" (the Coram-Beineke Hamlet example).

Examples:
    python main.py                                  # one run on the Hamlet passage
    python main.py --seed 7 --iters 20000
    python main.py --restarts 20                    # how often does a run recover the text?
    python main.py --restarts 10 --compare-powers 0.5,1,2,4 --plot powers.png
    python main.py --message my.txt --length 100    # only use the first 100 characters

    # Crack a ciphertext you already have; --message is the known answer, used for grading.
    python main.py --ciphertext messages/examples/1.cipher.txt --message messages/examples/1.txt
    python run_examples.py                          # crack every example in messages/examples/
"""

import argparse
from pathlib import Path

import numpy as np

from alphabet import decode, encode, normalize
from cipher import accuracy, apply_key, random_key
from corpus import load_transition_matrix
from sampler import metropolis, text_log_pl

DEFAULT_MESSAGE = Path(__file__).parent / "messages" / "hamlet.txt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--message", type=Path, default=None,
                   help="plaintext file to encrypt and crack (default: Hamlet); "
                        "with --ciphertext, the known answer used for grading")
    p.add_argument("--ciphertext", type=Path, default=None,
                   help="crack this already-encrypted text file instead of encrypting --message")
    p.add_argument("--length", type=int, default=None, help="use only the first N characters of the message")
    p.add_argument("--iters", type=int, default=10_000, help="Metropolis iterations per run")
    p.add_argument("--seed", type=int, default=0, help="random seed (cipher and sampler)")
    p.add_argument("--report-every", type=int, default=100, help="print the decryption every N iterations")
    p.add_argument("--restarts", type=int, default=1, help="number of independent runs from random starts")
    p.add_argument("--power", type=float, default=1.0,
                   help="accept with probability (Pl(f*)/Pl(f))^p; 1 = the paper's algorithm")
    p.add_argument("--compare-powers", type=str, default=None,
                   help="comma-separated powers to compare, e.g. 0.5,1,2 (overrides --power)")
    p.add_argument("--plot", type=Path, default=None, help="save a log-plausibility trace plot here (.png)")
    p.add_argument("--verbose", action="store_true",
                   help="print progress every --report-every iterations even with multiple runs")
    args = p.parse_args()
    if args.message is None:
        if args.ciphertext:
            p.error("--ciphertext needs --message (the answer) to grade the decryption")
        args.message = DEFAULT_MESSAGE
    return args


def read_symbols(path, length):
    """Read a text file, normalize it, optionally truncate it, and encode it."""
    text = normalize(path.read_text())
    if length is not None:
        text = text[:length].strip()
    return encode(text)


def run_experiment(ciphertext, plaintext, log_M, true_log_pl, power, args, show_progress):
    """Run `args.restarts` independent chains at one power; print per-run results."""
    results = []
    for r in range(args.restarts):
        # Seeding each run by (seed, run index) means run r starts from the same
        # random key at every power, so power comparisons are like-for-like.
        rng = np.random.default_rng([args.seed, r])

        def report(it, key, log_pl):
            # One line per report, like Figure 3 of the paper.
            print(f"{it:>6} {decode(apply_key(key, ciphertext))[:100]}")

        if show_progress and args.restarts > 1:
            print(f"\n--- run {r + 1} (p = {power:g}) ---")
        res = metropolis(ciphertext, log_M, args.iters, rng, power=power,
                         report_every=args.report_every,
                         on_report=report if show_progress else None)
        res.accuracy = accuracy(apply_key(res.best_key, ciphertext), plaintext)
        results.append(res)
        if args.restarts > 1:
            print(f"  p={power:<4g} run {r + 1:>3}: accuracy {res.accuracy:6.1%}   "
                  f"best log Pl {res.best_log_pl:8.1f}   accept {res.accept_rate:5.1%}")
    return results


def summarize(power, results, true_log_pl):
    """One line: how many runs recovered the text, and how many got close."""
    n = len(results)
    full = sum(r.accuracy == 1.0 for r in results)
    good = sum(r.accuracy >= 0.9 for r in results)
    # A run whose best key is at least as plausible as the true key has "solved"
    # the optimization problem; any remaining errors are the model's fault, not the sampler's.
    solved = sum(r.best_log_pl >= true_log_pl - 1e-9 for r in results)
    mean_acc = np.mean([r.accuracy for r in results])
    print(f"p = {power:<4g}  full recovery {full}/{n}   >=90% correct {good}/{n}   "
          f">= true-key plausibility {solved}/{n}   mean accuracy {mean_acc:.1%}")


def main():
    args = parse_args()
    powers = ([float(x) for x in args.compare_powers.split(",")]
              if args.compare_powers else [args.power])

    log_M = np.log(load_transition_matrix())

    plaintext = read_symbols(args.message, args.length)
    if args.ciphertext:
        ciphertext = read_symbols(args.ciphertext, args.length)
    else:
        # The cipher depends only on --seed, so every run and power attacks the same ciphertext.
        encrypt_key = random_key(np.random.default_rng(args.seed))
        ciphertext = apply_key(encrypt_key, plaintext)
    # Plausibility of the true text. When we did the encryption, this is exactly
    # log Pl(true key); for a supplied ciphertext it's the benchmark to beat.
    true_log_pl = text_log_pl(plaintext, log_M)

    print(f"Plaintext  ({len(plaintext)} chars): {decode(plaintext)}")
    print(f"Ciphertext ({len(ciphertext)} chars): {decode(ciphertext)}")
    print(f"log Pl(true text) = {true_log_pl:.1f}\n")

    show_progress = args.verbose or (args.restarts == 1 and len(powers) == 1)
    all_results = {p: run_experiment(ciphertext, plaintext, log_M, true_log_pl, p, args, show_progress)
                   for p in powers}

    # Overall best key across every run and power.
    best = max((r for rs in all_results.values() for r in rs), key=lambda r: r.best_log_pl)
    print("\nBest decryption:")
    print(decode(apply_key(best.best_key, ciphertext)))
    print(f"\nlog Pl(best) = {best.best_log_pl:.1f}   (true text: {true_log_pl:.1f})")
    print(f"Accuracy: {best.accuracy:.1%} of characters correct")

    if args.restarts > 1 or len(powers) > 1:
        print("\nSummary")
        for p, rs in all_results.items():
            summarize(p, rs, true_log_pl)

    if args.plot:
        from plotting import plot_traces
        title = f"Log-plausibility over {args.iters:,} iterations ({len(plaintext)}-char message)"
        plot_traces({p: [r.trace for r in rs] for p, rs in all_results.items()},
                    true_log_pl, args.plot, title)
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
