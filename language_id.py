"""Goal 3: identify the language of an encrypted message.

For each candidate language, decode the ciphertext with MCMC under that
language's trigram model and record the best log-probability per character. The
language whose model reaches the highest score is the prediction.

Text for each language comes from the Europarl corpus (parliamentary proceedings,
the same register in every language), downloaded once into data/languages/.
Accents are stripped so every language fits the 26-letter alphabet. The last
characters of each language's text are held out of training and used to make the
test messages, so no test message is ever seen during training.

Examples:
    python language_id.py
    python language_id.py --languages english,french,german,spanish,italian,dutch
    python language_id.py --length 200 --tests 5 --restarts 6 --csv language_id.csv
"""

import argparse
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from alphabet import encode
from cipher_types import SPACE, random_key
from ngram import build_model
from sampler_ext import linear_ramp, metropolis
from text_utils import clean_text, hold_out_tail

EUROPARL_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/europarl_raw.zip"
DATA_DIR = Path(__file__).parent / "data" / "languages"
HELD_OUT_CHARS = 40_000
MAX_TRAIN_CHARS = 2_000_000


def decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def prepare_corpora(languages, data_dir=DATA_DIR):
    """Download Europarl once and write one cleaned text file per language."""
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [lang for lang in languages if not (data_dir / f"{lang}.txt").exists()]
    if not missing:
        return
    print(f"Downloading Europarl from {EUROPARL_URL} ...")
    with urllib.request.urlopen(EUROPARL_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for lang in missing:
        prefix = f"europarl_raw/{lang}/"
        names = sorted(n for n in archive.namelist() if n.startswith(prefix) and not n.endswith("/"))
        if not names:
            raise SystemExit(f"No Europarl text found for language '{lang}'")
        raw = " ".join(decode_bytes(archive.read(n)) for n in names)
        text = clean_text(raw)[:MAX_TRAIN_CHARS + HELD_OUT_CHARS]
        (data_dir / f"{lang}.txt").write_text(text)
        print(f"  {lang}: {len(text):,} characters")


def make_tests(held_out: str, n_tests: int, length: int):
    """Non-overlapping snippets of `length` characters from the held-out text."""
    step = len(held_out) // n_tests
    return [held_out[i * step: i * step + length].strip() for i in range(n_tests)]


def best_score_per_char(ciphertext, model, args, rng_seed):
    """Highest log-probability per character found over all restarts under one model."""
    schedule = linear_ramp(args.beta_start, args.beta_end)
    best = -np.inf
    for r in range(args.restarts):
        res = metropolis(ciphertext, model, args.iters, np.random.default_rng([rng_seed, r]),
                         schedule=schedule, fixed={SPACE: SPACE})
        best = max(best, res.best_score)
    return best / len(ciphertext)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", type=str, default="english,french,german,spanish,italian")
    p.add_argument("--length", type=int, default=300, help="characters per test message")
    p.add_argument("--tests", type=int, default=3, help="test messages per language")
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--iters", type=int, default=20_000)
    p.add_argument("--beta-start", type=float, default=0.3)
    p.add_argument("--beta-end", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--csv", type=Path, default=None, help="save every (message, candidate) score here")
    return p.parse_args()


def main():
    args = parse_args()
    languages = args.languages.split(",")
    prepare_corpora(languages, args.data_dir)

    models, tests = {}, {}
    for lang in languages:
        text = (args.data_dir / f"{lang}.txt").read_text()
        train, held_out = hold_out_tail(text, HELD_OUT_CHARS)
        models[lang] = build_model(train[:MAX_TRAIN_CHARS], 3)
        tests[lang] = make_tests(held_out, args.tests, args.length)

    print(f"\n{len(languages)} languages, {args.tests} messages each of ~{args.length} characters; "
          f"{args.restarts} runs x {args.iters:,} iterations per (message, language)\n")
    print(f"{'true':<10} {'predicted':<10} " + " ".join(f"{lang[:7]:>8}" for lang in languages))

    confusion = {t: {p: 0 for p in languages} for t in languages}
    rows = []
    for t_index, true_lang in enumerate(languages):
        for i, snippet in enumerate(tests[true_lang]):
            seed = args.seed * 1000 + t_index * 100 + i
            plaintext = encode(snippet)
            ciphertext = random_key(np.random.default_rng(seed), fix_space=True)[plaintext]
            scores = {lang: best_score_per_char(ciphertext, models[lang], args, seed) for lang in languages}
            predicted = max(scores, key=scores.get)
            confusion[true_lang][predicted] += 1
            print(f"{true_lang:<10} {predicted:<10} " + " ".join(f"{scores[l]:8.3f}" for l in languages)
                  + ("" if predicted == true_lang else "   <-- wrong"))
            rows += [{"true": true_lang, "message": i, "candidate": lang, "score_per_char": scores[lang]}
                     for lang in languages]

    correct = sum(confusion[l][l] for l in languages)
    total = sum(sum(row.values()) for row in confusion.values())
    print(f"\nCorrectly identified: {correct}/{total}\n")
    print(f"{'true \\ predicted':<17}" + " ".join(f"{l[:7]:>8}" for l in languages))
    for t in languages:
        print(f"{t:<17}" + " ".join(f"{confusion[t][p]:>8}" for p in languages))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nScores saved to {args.csv}")


if __name__ == "__main__":
    main()
