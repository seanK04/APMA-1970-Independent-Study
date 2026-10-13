"""Metropolis sampler with pluggable language models, fixed symbols and annealing.

Extends sampler.py. Differences:
  * the score comes from a model object (bigram or trigram) via model.score
  * some cipher symbols can be pinned to known plaintext symbols (e.g. the space)
  * the inverse temperature can change over the run (annealing)

A proposal is accepted if log U < beta * (score_new - score_old), so beta = 1 is
the standard algorithm, small beta explores freely, and large beta is greedy.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from alphabet import N_SYMBOLS

Schedule = Callable[[int, int], float]  # (iteration, n_iter) -> beta


def constant(beta: float = 1.0) -> Schedule:
    return lambda it, n_iter: beta


def linear_ramp(start: float, end: float) -> Schedule:
    """Raise beta linearly from `start` to `end` over the run."""
    return lambda it, n_iter: start + (end - start) * it / max(n_iter - 1, 1)


@dataclass
class RunResult:
    best_key: np.ndarray
    best_score: float
    accept_rate: float
    trace: np.ndarray  # score of the current key after each iteration


def initial_key(rng: np.random.Generator, fixed: Dict[int, int]) -> np.ndarray:
    """Random key that respects the pinned symbols."""
    key = np.empty(N_SYMBOLS, dtype=np.int64)
    free_cipher = [s for s in range(N_SYMBOLS) if s not in fixed]
    free_plain = [s for s in range(N_SYMBOLS) if s not in set(fixed.values())]
    key[free_cipher] = rng.permutation(free_plain)
    for cipher_symbol, plain_symbol in fixed.items():
        key[cipher_symbol] = plain_symbol
    return key


def metropolis(
    ciphertext: np.ndarray,
    model,
    n_iter: int,
    rng: np.random.Generator,
    schedule: Optional[Schedule] = None,
    fixed: Optional[Dict[int, int]] = None,
) -> RunResult:
    """Run one chain from a random key and return the best key seen."""
    schedule = schedule or constant(1.0)
    fixed = fixed or {}
    movable = np.array([s for s in range(N_SYMBOLS) if s not in fixed])

    key = initial_key(rng, fixed)
    score = model.score(key[ciphertext])
    best_key, best_score = key.copy(), score
    trace = np.empty(n_iter)
    accepted = 0

    for it in range(n_iter):
        i, j = rng.choice(movable, size=2, replace=False)
        proposal = key.copy()
        proposal[i], proposal[j] = proposal[j], proposal[i]
        new_score = model.score(proposal[ciphertext])

        delta = new_score - score
        if delta >= 0 or np.log(rng.random()) < schedule(it, n_iter) * delta:
            key, score = proposal, new_score
            accepted += 1
            if score > best_score:
                best_key, best_score = key.copy(), score
        trace[it] = score

    return RunResult(best_key, best_score, accepted / n_iter, trace)
