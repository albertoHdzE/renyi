# Bitácora 09 — WP-D, estimator defects: the sentinel cells and their consequences

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** G1-equivalent (property checks) — **9/9 pass**, now including P16.
**Artefacts:** `renyiext/spectrum.py` (sentinel cells),
`renyiext/checks.py` (P16; P8′ pinned grid; CHECKS list),
`scripts/run_p0_events.py` (dead stub removed), `renyiext/__init__.py`,
`renyiext/config.py`, `results/p2_temporal.json` (regenerated),
`results/p2b_decomposition.json` (regenerated + overflow render),
`results/p2c_probe.json` (fidelity re-run), `figures/p2d_overflow_mass.png`.

---

## 1. The fix, and the second leak the check found in it

Review C1: `log_bin_counts` dropped every interval above `hi`
(`np.histogram` semantics). The first fix added an overflow cell. **P16's own
first draft then failed** — `np.histogram` also drops everything *below* `lo`,
so the one-sided fix still leaked (9 of 437 intervals on a randomized grid).
The shipped contract is a complete partition:

```
[zero_count, underflow_count] + interior_bins + [overflow_count]   (n_bins+3)
```

with P16 asserting, over 200 randomized grids: `total == len(x)` exactly, each
sentinel equal to its indicator count. This is the datasaurus loop working as
designed — the check written to prove the fix refused to pass until the fix
was actually complete.

On Cresci-2015 the underflow cell is always empty (integer-ms gaps ≥ 1 ms =
lo), so no downstream number moves because of it; it exists so no future
corpus or grid can silently lose mass.

## 2. Measured out-of-range mass (G1 render: `p2d_overflow_mass.png`)

At hi = 400 d, computed elementwise from the event cache by committed code:

| class | intervals > hi | share of that class's intervals |
|---|---|---|
| bot | 73 | **0.0393 %** |
| human | 17 | 0.0007 % |

Direction and magnitude confirm the reviewer's back-of-envelope (they measured
63 / 0.035 % and 13 / 0.0005 %); these are now the canonical numbers, reproducible
from `run_p2b_decomposition.py`. Class-dependent in the direction that flattered
the old result — but tiny, as the reviewer said.

## 3. Full P2 re-run: old → new

| arm | AUC old → new | Δ | TPR@1%FPR old → new |
|---|---|---|---|
| COUNT | 0.9400 → 0.9400 | 0.0000 | 0.141 → 0.141 |
| BURST | 0.9141 → 0.9141 | 0.0000 | 0.344 → 0.344 |
| SHAN | 0.9318 → 0.9320 | +0.0001 | 0.489 → 0.503 |
| SPEC_T | 0.9699 → 0.9701 | +0.0002 | 0.779 → 0.789 |
| COUNT+BURST | 0.9594 → 0.9594 | 0.0000 | 0.520 → 0.520 |
| COUNT+SHAN | 0.9731 → 0.9729 | −0.0001 | 0.792 → 0.795 |
| COUNT+SPEC_T | 0.9767 → 0.9764 | −0.0003 | 0.792 → 0.800 |

Clause verdicts, unchanged in kind: (i) +0.0367 → **+0.0364** (still clears,
10/10, p = 0.0020); (ii) +0.0380 → **+0.0381** (still clears); burstiness floor
+0.0173 → **+0.0170** (still fails). Sweep ranges: clause (i) 0.0347–0.0380,
clause (ii) 0.0247–0.0619, still peaking at n_bins = 12. **Kill criterion 3 did
not fire.** The p2b reconciliation vs bitacora 04 remains finding-free at the
0.005 bound (largest drift +0.0007 on SPEC_T−MINUS_H₀ = +0.0111; CB delta
+0.0195 — both still fail the floor exactly as before).

## 4. P8′ under the pinned grid — reported, not tuned

Pinning `hi` to the corpus-wide 400 d (removing the account-dependent default
its own docs ban) changed the synthetic control's numbers:
spectrum accuracy 0.908 → **0.872**, Shannon-alone 0.878 → **0.764**, gain
+0.031 → **+0.108**. Both versions are honest; the pinned-grid version is the
one consistent with production feature construction, and it is the one that
stays. Recorded because a check whose numbers move when its hidden knob is
fixed is telling you the knob mattered.

## 5. Probe fidelity re-run (plan WP-A task 4 discharged)

Trigger verdict unchanged: **FIRED at 1.0000**; max cell delta across all 27
readings ≤ 0.004 (bursty cells rose slightly, as predicted — class A regains
long-gap distinguishing mass). `pipeline_version.overflow_cell` is now
self-detected (`true`). Determinism: byte-identical twice.

## 6. Hygiene and ledger items

- Dead stub removed from `scripts/run_p0_events.py` (review C5).
- `renyiext.__all__` exports the real module list.
- `INFERRED_PARAMETERS[n_events]` restated per D3′ (superseded wording).
- **B2 ledgered:** `config.CRESCI_WINDOW` upper bound is 2015-12-31 while
  METHODS' P14 says 2013-07-01. The widening was deliberate (code comment) so
  the acceptance range cannot be confounded with the observed corpus end; it
  is recorded here as the decision of record rather than left in a comment.

## 7. What failed and was not fixed

Nothing outstanding. The one near-miss — the incomplete first overflow fix —
is §1 and is itself the argument for P16-style identity checks over
smoke tests.

## 8. Multiple comparisons counted

200 randomized grids × P16 identities; 10 arms recomputed; 5 paired tests;
1 probe fidelity comparison against a pre-stated ≤ 0.01 tolerance. No new
hypothesis-level claims.
