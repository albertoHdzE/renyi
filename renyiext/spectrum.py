"""Rényi entropy and the Rényi spectrum (docs/01-METHODS.md sect. 1).

    H_a(p) = 1/(1-a) * log2( sum_i p_i^a )                        a > 0, a != 1

with the limiting cases taken exactly rather than numerically:

    H_0   = log2 |{i : p_i > 0}|      support richness
    H_1   = -sum p_i log2 p_i         Shannon (the a -> 1 limit)
    H_2   = -log2 sum p_i^2           collision entropy
    H_inf = -log2 max_i p_i           min-entropy

**Base is 2 (bits), not nats** -- decision D2. ``dtwre/entropy.py`` uses natural
logs, which is fine in isolation, but this project mixes entropies with CTM and
BDM values, which are in bits by construction (``CTM(s) ~ -log2 D(s)``).

**Bias correction: none, deliberately.** The plug-in estimator is used, and the
sample-size dependence is handled by design rather than by correction -- see
decision D3' in ``bitacora/02_h1_amendment.md``. There is no standard correction
for general ``a`` comparable to Chao-Shen at ``a = 1``, so pretending to one
would be worse than stating the bias and controlling for it downstream. The bias
is real, it is largest at small ``a``, and :func:`spectrum_vs_n` measures it.

The spectrum is *not* tuned over ``a``. Tuning ``a`` is one-dimensional feature
selection with a multiple-comparisons hazard, and the DTWRE replication already
showed what that search finds when there is nothing there (alpha_sweep.json:
0.0098 AUC spread against a seed sigma of 0.02). The whole vector is the feature.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "renyi_entropy", "spectrum", "spectrum_labels",
    "counts_to_probabilities", "log_bin_counts",
    "spectrum_vs_n", "SPECTRUM_ALPHAS",
]

SPECTRUM_ALPHAS: tuple = (0.0, 0.5, 1.0, 2.0, 4.0, np.inf)

_TOL = 1e-12
# Switch to the series expansion inside this band of alpha = 1; see renyi_entropy.
_NEAR_ONE = 1e-6
_LOG2 = np.log(2.0)


def renyi_entropy(p: np.ndarray, alpha: float) -> float:
    """Rényi entropy of order ``alpha``, in **bits**.

    ``p`` is renormalised to sum to one; zeros are dropped. An empty or
    all-zero input returns 0.0, the entropy of a degenerate distribution.

    The limits at ``alpha`` 0, 1 and infinity are evaluated in closed form, not
    approached numerically: ``1/(1-a)`` is singular at ``a = 1`` and the naive
    expression loses all precision in a neighbourhood of it.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()

    if np.isinf(alpha):
        return float(-np.log(p.max()) / _LOG2)
    if abs(alpha) < _TOL:
        return float(np.log(p.size) / _LOG2)

    # Near alpha = 1 the closed form 1/(1-a) * log2(sum p^a) is catastrophically
    # ill-conditioned: the log tends to 0 while the prefactor tends to infinity,
    # so the double-precision error in the log is amplified by 1/|1-a|. At
    # |1-a| = 1e-9 that is an absolute error of ~1e-7, which the P2 check found.
    #
    # Expand instead. With g(a) = sum p^a: g(1) = 1, g'(1) = sum p ln p, and
    # g''(1) = sum p (ln p)^2, so
    #
    #     H_a = -[ g'(1) + (a-1)/2 * Var_p[ln p] ] / ln 2 + O((a-1)^2)
    #         = H_1 - (a-1) * Var_p[ln p] / (2 ln 2) + O((a-1)^2)
    #
    # which is exact at a = 1 and first-order accurate around it. The crossover
    # where the expansion beats the closed form is |1-a| ~ 1e-8; _NEAR_ONE sits
    # safely above it.
    if abs(alpha - 1.0) < _NEAR_ONE:
        logp = np.log(p)
        h1 = -float(np.sum(p * logp)) / _LOG2
        if alpha == 1.0:
            return h1
        mean_logp = float(np.sum(p * logp))
        var_logp = float(np.sum(p * (logp - mean_logp) ** 2))
        return h1 - (alpha - 1.0) * var_logp / (2.0 * _LOG2)

    # log-sum-exp for stability: sum p^a can under/overflow for large |a|
    logp = np.log(p)
    m = np.max(alpha * logp)
    log_sum = m + np.log(np.sum(np.exp(alpha * logp - m)))
    return float(log_sum / ((1.0 - alpha) * _LOG2))


def spectrum(p: np.ndarray, alphas=SPECTRUM_ALPHAS) -> np.ndarray:
    """The Rényi spectrum as a vector, in bits. The feature, not a tuned point."""
    return np.array([renyi_entropy(p, a) for a in alphas], dtype=np.float64)


def spectrum_labels(alphas=SPECTRUM_ALPHAS, prefix: str = "H") -> list[str]:
    return [f"{prefix}_inf" if np.isinf(a) else f"{prefix}_{a:g}" for a in alphas]


def counts_to_probabilities(counts: np.ndarray) -> np.ndarray:
    """Normalise a count vector, dropping empty cells."""
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    return c / c.sum() if c.size else c


def log_bin_counts(x: np.ndarray, n_bins: int = 24,
                   lo: float = 1.0, hi: float | None = None) -> np.ndarray:
    """Histogram positive values on a log grid.

    Inter-arrival times span 1 ms to ~1 year, nine orders of magnitude, so a
    linear grid puts essentially all mass in one cell and reports ``H = 0`` for
    every account. The grid is an invented parameter and is swept in the P1
    gate rather than assumed (datasaurus G1).

    Zeros are kept as their own leading bin: simultaneous posting is real and
    highly informative for automated accounts, and folding it into the first
    log bin would discard exactly the behaviour we are looking for.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(n_bins + 1, dtype=np.float64)
    n_zero = int((x <= 0).sum())
    pos = x[x > 0]
    if pos.size == 0:
        return np.concatenate([[n_zero], np.zeros(n_bins)])
    hi = float(hi if hi is not None else max(pos.max(), lo * 10))
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    h, _ = np.histogram(pos, bins=edges)
    return np.concatenate([[n_zero], h]).astype(np.float64)


def spectrum_vs_n(sampler, n_grid, alphas=SPECTRUM_ALPHAS,
                  n_rep: int = 50, rng=None) -> dict:
    """Measure the finite-sample bias of every order (property P7).

    ``sampler(n, rng)`` draws ``n`` observations from a fixed distribution and
    returns a count vector. Because the distribution is fixed, any variation of
    ``H_a`` with ``n`` is estimator bias and nothing else.

    Returns mean and SD per ``(n, alpha)``. H_0 is expected to rise steeply with
    n -- observed support underestimates true support -- and the higher orders
    to be nearly flat. Reporting this, rather than correcting it, is decision D3'.
    """
    rng = np.random.default_rng(rng)
    mean = np.zeros((len(n_grid), len(alphas)))
    sd = np.zeros_like(mean)
    for i, n in enumerate(n_grid):
        vals = np.array([spectrum(counts_to_probabilities(sampler(n, rng)), alphas)
                         for _ in range(n_rep)])
        mean[i] = vals.mean(axis=0)
        sd[i] = vals.std(axis=0)
    return {"n_grid": list(n_grid), "alphas": list(alphas),
            "mean": mean, "sd": sd}
