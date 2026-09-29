"""Reference corpus (War and Peace) and the first-order transition matrix M.

M[x, y] is an estimate of P(next symbol = y | current symbol = x), learned from
a long stretch of ordinary English. The sampler uses it to score how
"English-like" a candidate decryption looks.

Both the downloaded book and the finished matrix are cached in ./data so later
runs skip the download and the counting.
"""

import urllib.request
from pathlib import Path

import numpy as np

from alphabet import N_SYMBOLS, encode, normalize

DATA_DIR = Path(__file__).parent / "data"
BOOK_PATH = DATA_DIR / "war_and_peace.txt"
MATRIX_PATH = DATA_DIR / "transitions.npy"
BOOK_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"

# Project Gutenberg wraps every book in license boilerplate between these markers.
_START_MARK = "*** START OF"
_END_MARK = "*** END OF"


def download_book() -> str:
    """Return the raw text of War and Peace, downloading it once if needed."""
    if not BOOK_PATH.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"Downloading War and Peace from {BOOK_URL} ...")
        with urllib.request.urlopen(BOOK_URL) as resp:
            BOOK_PATH.write_bytes(resp.read())
    return BOOK_PATH.read_text(encoding="utf-8-sig")


def strip_gutenberg(raw: str) -> str:
    """Keep only the text between the Gutenberg START and END markers."""
    start = raw.find(_START_MARK)
    if start != -1:
        start = raw.index("\n", start) + 1  # skip the rest of the marker line
    else:
        start = 0
    end = raw.find(_END_MARK)
    if end == -1:
        end = len(raw)
    return raw[start:end]


def build_transition_matrix(text: str) -> np.ndarray:
    """Count symbol bigrams in `text` and convert them to row-normalized probabilities.

    Add-one (Laplace) smoothing: every count starts at 1, so no transition has
    probability zero. Without it, a single "impossible" bigram (say, QZ) would
    make a key's plausibility exactly 0 and log-plausibility -inf.
    """
    symbols = encode(text)
    counts = np.ones((N_SYMBOLS, N_SYMBOLS))
    # np.add.at accumulates correctly when the same (x, y) pair repeats.
    np.add.at(counts, (symbols[:-1], symbols[1:]), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def load_transition_matrix() -> np.ndarray:
    """Return M, building and caching it on the first call."""
    if MATRIX_PATH.exists():
        return np.load(MATRIX_PATH)
    text = normalize(strip_gutenberg(download_book()))
    print(f"Building transition matrix from {len(text):,} characters ...")
    M = build_transition_matrix(text)
    np.save(MATRIX_PATH, M)
    return M
