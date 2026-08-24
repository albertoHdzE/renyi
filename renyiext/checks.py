"""Properties the mathematics must satisfy (docs/01-METHODS.md).

This repository has no test suite. Correctness is asserted inline against
properties, following ``dtwre``, ``disinfo`` and ``botsage``. Each function
raises ``AssertionError`` on failure and returns a one-line human-readable
record on success; ``run_all`` collects them into ``results/checks.json``.

P1-P8 cover the spectrum estimator (phase P1). P8 is superseded by P8' per
``bitacora/02_h1_amendment.md`` and is implemented in its amended form.
"""

from __future__ import annotations

import numpy as np

from .spectrum import (renyi_entropy, spectrum, counts_to_probabilities,
                       log_bin_counts, spectrum_vs_n, SPECTRUM_ALPHAS)
from .generators import generate_control_set

__all__ = ["run_all"]

_ALPHAS_DENSE = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, np.inf)


def check_uniform_gives_log_n():          # P1
    msgs = []
    for n in (2, 5, 17, 128):
        p = np.full(n, 1.0 / n)
        for a in _ALPHAS_DENSE:
            got, want = renyi_entropy(p, a), np.log2(n)
            assert abs(got - want) < 1e-10, f"n={n} a={a}: {got} != {want}"
        msgs.append(str(n))
    return ("P1 check_uniform_gives_log_n: H_a(uniform on n) == log2 n for every "
            f"alpha, n in {{{', '.join(msgs)}}} (err < 1e-10)")


def check_shannon_at_alpha_one():         # P2
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        p = rng.dirichlet(np.full(rng.integers(2, 40), 0.7))
        want = -np.sum(p * np.log2(p))
        for a in (1.0, 1.0 + 1e-9, 1.0 - 1e-9):
            worst = max(worst, abs(renyi_entropy(p, a) - want))
    assert worst < 1e-8, f"Shannon limit off by {worst}"
    return (f"P2 check_shannon_at_alpha_one: H_1 == -sum p log2 p over 200 random "
            f"simplex points, and is continuous at 1 +- 1e-9 (max err {worst:.2e})")


def check_monotone_in_alpha():            # P3
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        p = rng.dirichlet(np.full(rng.integers(2, 40), 0.5))
        h = np.array([renyi_entropy(p, a) for a in _ALPHAS_DENSE])
        d = np.diff(h)
        worst = max(worst, float(d.max()))
        assert d.max() < 1e-10, f"H_a increased in alpha by {d.max()}"
    return ("P3 check_monotone_in_alpha: H_a is non-increasing in alpha on 300 "
            f"random distributions (largest increase {worst:.2e})")


def check_ordering_bounds():              # P4
    # NB the tolerance is two-sided and applied per pair. An earlier version
    # chained the comparisons, which in Python evaluates as a conjunction and
    # applied the tolerance in one direction only -- it then failed on the last
    # ulp between log(n)/log(2) and np.log2(n), which is a difference of 1e-15
    # in the *check*, not in the estimator.
    rng = np.random.default_rng(2)
    tol = 1e-9
    worst = 0.0
    for _ in range(300):
        n = int(rng.integers(2, 40))
        p = rng.dirichlet(np.full(n, 0.5))
        k = int((p > 0).sum())               # support may be < n after underflow
        h0, h1, h2 = (renyi_entropy(p, a) for a in (0.0, 1.0, 2.0))
        hinf = renyi_entropy(p, np.inf)
        for lo, hi, name in ((hinf, h2, "H_inf <= H_2"), (h2, h1, "H_2 <= H_1"),
                             (h1, h0, "H_1 <= H_0"),
                             (h0, np.log2(k), "H_0 <= log2 |support|")):
            assert lo <= hi + tol, f"{name} violated by {lo - hi}"
            worst = max(worst, float(lo - hi))
    return ("P4 check_ordering_bounds: H_inf <= H_2 <= H_1 <= H_0 <= log2|support| "
            f"on 300 draws (largest excess {worst:.2e}, tol {tol:.0e})")


def check_degenerate_is_zero():           # P5
    for a in _ALPHAS_DENSE:
        assert abs(renyi_entropy(np.array([1.0]), a)) < 1e-12
        assert abs(renyi_entropy(np.array([0.0, 7.0, 0.0]), a)) < 1e-12
    assert renyi_entropy(np.array([]), 2.0) == 0.0
    return ("P5 check_degenerate_is_zero: a point mass has H_a = 0 for every alpha; "
            "empty input returns 0")


def check_permutation_invariance():       # P6
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        p = rng.dirichlet(np.full(rng.integers(2, 40), 0.6))
        q = rng.permutation(p)
        worst = max(worst, float(np.abs(spectrum(p) - spectrum(q)).max()))
    assert worst < 1e-12, f"permutation changed the spectrum by {worst}"
    return ("P6 check_permutation_invariance: the spectrum is invariant to relabelling "
            f"(max err {worst:.2e}) -- unlike BDM on an adjacency matrix, cf. R3")


def check_bias_direction_with_n():        # P7
    """H_0 must rise with n on a FIXED distribution: that rise is pure bias."""
    true_p = np.array([0.4, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01, 0.007, 0.003])
    true_p = true_p / true_p.sum()

    def sampler(n, rng):
        return np.bincount(rng.choice(len(true_p), size=n, p=true_p),
                           minlength=len(true_p))

    out = spectrum_vs_n(sampler, (8, 16, 32, 64, 128, 512, 4096),
                        alphas=SPECTRUM_ALPHAS, n_rep=80, rng=4)
    h0 = out["mean"][:, 0]
    hinf = out["mean"][:, -1]
    assert np.all(np.diff(h0) > -1e-9), "H_0 did not increase with n"
    drift0 = float(h0[-1] - h0[0])
    driftinf = float(abs(hinf[-1] - hinf[0]))
    assert drift0 > driftinf, "H_0 should be far more n-biased than H_inf"
    return (f"P7 check_bias_direction_with_n: on a fixed distribution H_0 drifts "
            f"{drift0:.3f} bits from n=8 to n=4096 while H_inf drifts {driftinf:.3f} "
            f"-- the bias is real, largest at small alpha, and is reported not corrected (D3')")


def check_overflow_mass_conserved():      # P16
    """WP-D (review C1): log_bin_counts must conserve mass exactly.

    Before the sentinel cells existed, np.histogram silently DROPPED every
    interval outside [lo, hi] -- class-dependently on the real corpus. P16
    asserts the complete-partition identity elementwise: total == len(x),
    zero-cell == (x <= 0).sum(), underflow == (0 < x < lo).sum(), overflow ==
    (x > hi).sum(). The underflow cell itself was FOUND by an earlier draft of
    this check: the first overflow-only fix still leaked 9 of 437 intervals on
    a grid with lo above the smallest sample.
    """
    rng = np.random.default_rng(9)
    for trial in range(200):
        n_bins = int(rng.integers(4, 40))
        hi = float(10.0 ** rng.uniform(1, 8))
        lo = hi / 10.0 ** rng.uniform(1, 4)
        x = rng.lognormal(mean=np.log(hi / 10), sigma=2.0,
                          size=int(rng.integers(1, 500)))
        if rng.random() < 0.3:                     # exercise the zero cell
            x = np.concatenate([x, [0.0, -5.0]])
        h = log_bin_counts(x, n_bins=n_bins, lo=lo, hi=hi)
        assert len(h) == n_bins + 3, f"length {len(h)} != {n_bins + 3}"
        assert h.sum() == x.size, (
            f"mass lost: {h.sum()} != {x.size} (n_bins={n_bins})")
        assert h[0] == (x <= 0).sum()
        assert h[1] == ((x > 0) & (x < lo)).sum()
        assert h[-1] == (x > hi).sum()
    return ("P16 check_overflow_mass_conserved: log_bin_counts partitions x "
            "elementwise over 200 randomized grids -- zero, underflow, "
            "interior bins and overflow sum exactly to len(x) (review C1 fix; "
            "the underflow leak was found by this check's first draft)")


def check_spectrum_separates_generators_at_matched_n():   # P8'
    """The decisive control (S4.2, P8').

    Periodic, Poisson and heavy-tailed accounts, all with **identical** event
    counts. Event count therefore carries zero information, so any separation
    is distributional shape. If this fails, the spectrum is a volume proxy and
    nothing measured on the real corpus can be interpreted -- P0 found volume
    alone worth AUC 0.939 there.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # The generators are deliberately made HARD (periodic jitter 0.5, Pareto
    # tail 1.2). At the easy default they separate at 1.000 and so does Shannon
    # alone -- a control everything passes licenses nothing (datasaurus G4:
    # "only the richest control licenses a claim").
    series, y, names = generate_control_set(n_per_class=120, n_events=200,
                                            rng=5, jitter=0.5)
    # The grid is PINNED to the corpus-wide default (WP-D): an account-
    # dependent hi would let each series carry its own maximum gap into the
    # feature -- exactly what features.temporal_blocks_ts warns against.
    X = np.array([spectrum(counts_to_probabilities(
        log_bin_counts(s, n_bins=24, lo=1.0, hi=400 * 86_400_000.0)))
        for s in series])
    counts = np.array([len(s) for s in series])
    assert counts.std() == 0, "the control set is not matched on n"

    acc = cross_val_score(
        LogisticRegression(max_iter=3000),
        X, y, cv=5, scoring="accuracy").mean()
    assert acc > 0.80, f"spectrum separates the 3 known generators at only {acc:.3f}"

    # Reported, NOT asserted. Whether the spectrum beats its own alpha = 1 point
    # is hypothesis H1, and a check must not assert the hypothesis it exists to
    # make testable. The measured value here is a standing warning: see
    # bitacora/03.
    acc_h1 = cross_val_score(
        LogisticRegression(max_iter=3000),
        X[:, [2]], y, cv=5, scoring="accuracy").mean()
    return (f"P8' check_spectrum_separates_generators_at_matched_n: at identical n=200, "
            f"the spectrum separates periodic/poisson/bursty at accuracy {acc:.3f} "
            f"(chance 0.333) -- shape, not volume. Shannon alone {acc_h1:.3f} "
            f"(gain {acc - acc_h1:+.3f}, REPORTED not asserted -- that is H1)")


CHECKS = [
    check_uniform_gives_log_n,
    check_shannon_at_alpha_one,
    check_monotone_in_alpha,
    check_ordering_bounds,
    check_degenerate_is_zero,
    check_permutation_invariance,
    check_bias_direction_with_n,
    check_overflow_mass_conserved,
    check_spectrum_separates_generators_at_matched_n,
]


def run_all(verbose: bool = True) -> list[str]:
    out = []
    for fn in CHECKS:
        msg = fn()
        out.append(msg)
        if verbose:
            print("  " + msg)
    return out
