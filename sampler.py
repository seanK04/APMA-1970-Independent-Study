"""Metropolis sampler over decoding keys.

A decoding key f maps cipher symbols to plaintext symbols. Its plausibility is

    Pl(f) = prod_i M[f(s_i), f(s_{i+1})]

where s is the ciphertext. We only ever need ratios of Pl, so we work with
log Pl(f) = sum_i log M[f(s_i), f(s_{i+1})], which avoids numerical underflow
(a product of ~200 probabilities is far below the smallest float).
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from alphabet import N_SYMBOLS
from cipher import apply_key, random_key


def text_log_pl(symbols: np.ndarray, log_M: np.ndarray) -> float:
    """Sum of log transition probabilities along a (decoded) text."""
    return float(log_M[symbols[:-1], symbols[1:]].sum())


def log_plausibility(key: np.ndarray, ciphertext: np.ndarray, log_M: np.ndarray) -> float:
    """log Pl(f): the log-plausibility of the text that key f decrypts to."""
    return text_log_pl(apply_key(key, ciphertext), log_M)


@dataclass
class RunResult:
    best_key: np.ndarray
    best_log_pl: float
    final_key: np.ndarray
    trace: np.ndarray                 # log Pl of the current key after each iteration
    accept_rate: float


def metropolis(
    ciphertext: np.ndarray,
    log_M: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
    power: float = 1.0,
    report_every: int = 100,
    on_report: Optional[Callable[[int, np.ndarray, float], None]] = None,
) -> RunResult:
    """Run the Metropolis chain for `n_iter` steps from a random key.

    power: the "inverse temperature" p. A proposal is accepted with probability
           min(1, (Pl(f*)/Pl(f))^p). p=1 is the paper's algorithm; larger p is
           greedier (rarely goes downhill), smaller p explores more freely.
    on_report: called as on_report(iteration, current_key, current_log_pl)
               every `report_every` iterations.
    """
    key = random_key(rng)
    log_pl = log_plausibility(key, ciphertext, log_M)
    best_key, best_log_pl = key.copy(), log_pl

    trace = np.empty(n_iter)
    accepted = 0

    for it in range(1, n_iter + 1):
        # Propose f* by swapping the images of two distinct cipher symbols.
        # (Drawing j from the 26 other symbols is a fast way to guarantee i != j.)
        i = rng.integers(N_SYMBOLS)
        j = rng.integers(N_SYMBOLS - 1)
        j += j >= i
        proposal = key.copy()
        proposal[i], proposal[j] = proposal[j], proposal[i]
        new_log_pl = log_plausibility(proposal, ciphertext, log_M)

        # Accept if uphill; otherwise with probability exp(p * (log Pl* - log Pl)).
        # Comparing in log space: accept iff log(U) < p * delta.
        delta = new_log_pl - log_pl
        if delta >= 0 or np.log(rng.random()) < power * delta:
            key, log_pl = proposal, new_log_pl
            accepted += 1
            if log_pl > best_log_pl:
                best_key, best_log_pl = key.copy(), log_pl

        trace[it - 1] = log_pl
        if on_report is not None and it % report_every == 0:
            on_report(it, key, log_pl)

    return RunResult(best_key, best_log_pl, key, trace, accepted / n_iter)
