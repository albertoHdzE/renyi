# Bitácora 03 — P1, the spectrum estimator

**Date:** 2026-08-18
**Branch:** `main`
**Gate G1:** **PASS** — 8/8 properties, including P8′ in its amended form.
**Artefacts:** `renyiext/{spectrum,generators,checks}.py`
**Seeds:** fixed per check (0–7); two runs agree to the digit.

---

## 1. Gate G1: all eight properties pass

| # | Property | Result |
|---|---|---|
| P1 | uniform → log₂ n for every α | err < 1e-10, n ∈ {2, 5, 17, 128} |
| P2 | Shannon at α = 1, continuous at 1 ± 1e-9 | max err **1.04e-09** over 200 simplex points |
| P3 | H_α non-increasing in α | largest increase **0.00** over 300 draws |
| P4 | H_∞ ≤ H₂ ≤ H₁ ≤ H₀ ≤ log₂\|supp\| | largest excess 8.88e-16 |
| P5 | point mass → 0 for every α | exact |
| P6 | permutation invariance | max err 2.66e-15 |
| P7 | finite-sample bias, direction | H₀ drifts **1.314 bits** from n=8 to n=4096 on a *fixed* distribution; H_∞ drifts 0.133 |
| P8′ | separates known generators at matched n | **0.908** (chance 0.333) |

## 2. Two defects the checks caught

**P2 — catastrophic cancellation near α = 1.** The closed form
`1/(1−α)·log₂ Σpᵅ` is ill-conditioned around α = 1: the log tends to 0 while the
prefactor tends to ∞, so double-precision error in the log is amplified by 1/|1−α|.
At |1−α| = 1e-9 that is an absolute error of **7.55e-07**. Fixed with a series
expansion rather than by widening the tolerance:

```
H_α = H₁ − (α−1)·Var_p[ln p] / (2 ln 2) + O((α−1)²)
```

exact at α = 1, first-order accurate around it, switched in for |1−α| < 1e-6.
Error is now 1.04e-09.

**P4 — a defect in the check, not the estimator.** The original used a Python
chained comparison, which evaluates as a conjunction and applied the tolerance in
one direction only; it then failed on the last ulp between `log(n)/log(2)` and
`np.log2(n)` — a 1e-15 discrepancy in the *check*. Recorded because it is the
cheap version of a real hazard: a failing check is not evidence of a failing
estimator until you have read the check.

## 3. P7 quantifies R1, and it is large

On a **fixed** distribution, so every bit of movement is estimator bias:

| | n = 8 → 4096 |
|---|---|
| H₀ drift | **+1.314 bits** |
| H_∞ drift | 0.133 bits |

The bias is real, it is an order of magnitude worse at small α, and Cresci-2015's
per-account n ranges from 0 to ~3,200. Under D3′ this is reported and controlled by
conditioning on count, not corrected. P7 is the number that justifies making count a
mandatory covariate.

## 4. The finding that matters — a standing warning on H1 clause (ii)

The first version of P8′ used the easy generator settings and returned **1.000**, with
**Shannon alone also 1.000**. A control that everything passes licenses nothing (G4:
only the richest control licenses a claim), so the difficulty was swept.

**Spectrum vs Shannon-alone, 3-class synthetic, matched n = 200, 5-fold CV:**

| periodic jitter | Pareto tail | spectrum | Shannon alone | gain |
|---|---|---|---|---|
| 0.02 | 1.2 | 0.997 | 1.000 | **−0.003** |
| 0.50 | 1.2 | 0.925 | 0.911 | **+0.014** |
| 1.00 | 2.0 | 0.953 | 0.947 | **+0.006** |
| 1.50 | 3.0 | 0.939 | 0.939 | **+0.000** |
| 2.00 | 5.0 | 0.933 | 0.947 | **−0.014** |

**Across the whole difficulty range the gain is within ±0.014 — inside the 0.02
effect-size floor, and negative at both ends.** The single retained configuration
(jitter 0.5, tail 1.2) gives +0.031 at the check's own seed, which the sweep shows is
not stable across settings.

This is on synthetic data **engineered to have exactly the mechanism H1 postulates**:
periodic, Poisson and heavy-tailed inter-arrivals differ precisely in tail shape, which
is what α is supposed to resolve. If the spectrum cannot clear 0.02 over its own α = 1
point *there*, clause (ii) of H1 is in trouble.

**It is not falsified.** Three reasons the real task differs, all testable:

1. Different task — 3-class synthetic against binary real, different readout.
2. **Log-binning may be destroying the tail before the spectrum sees it.** `n_bins = 24`
   over 1 ms–1 year is an invented parameter, swept only for the histogram figure and
   not yet for the classifier. This is the first thing P2 must sweep.
3. Real accounts are mixtures, not pure generators; the spectrum may separate mixtures
   better than pure processes.

**But the pattern is the DTWRE α-flatness reappearing** (charter §1d: 0.0098 AUC spread
across α ∈ [0.2, 5] against seed σ 0.02), now on a second, unrelated substrate. That
convergence is worth more than either observation alone, and it is recorded before P2
runs so it cannot be rationalised afterwards.

**P8′ therefore reports the gain rather than asserting it.** A check must not assert the
hypothesis it exists to make testable.

## 5. Carried into P2

1. **Sweep the log-binning grid** (`n_bins`, `lo`, `hi`) as a first-class parameter of
   the classifier, not just the figure. §4 item 2.
2. H1 clause (i) — beating count alone at AUC 0.939 — is untouched by §4 and remains the
   harder of the two floors.
3. The α-curve figure must be rendered per class **with error bands** before any AUC is
   quoted (G1).

## 6. Multiple comparisons counted

Difficulty sweep: 5 (jitter, tail) pairs, all five reported in §4. One configuration
retained for the standing check, chosen as the hardest that still clears the 0.80 floor
— chosen *before* the gain was inspected, but recorded here as a post-hoc-adjacent choice
so a reader can discount it. No hypothesis was tested in this phase.
