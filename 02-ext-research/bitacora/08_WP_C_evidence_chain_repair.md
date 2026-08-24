# Bitácora 08 — WP-C, evidence-chain repair and the notebook audit

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** n/a (ledger repair; no hypothesis tested).
**Artefacts:** `scripts/run_p2b_decomposition.py`,
`results/p2b_decomposition.json`, notebook 04 rebuilt (§6.1 correction, §6.4
added), `docs/04-DECISIONS.md` P2 entry annotated + corrected,
EVIDENCE-INDEX completed for P2/WP-B/WP-C.
**Seeds:** 42–51 (identical protocol to P2). Two runs byte-identical.

---

## 1. A1 is closed — the quoted numbers were right, and now they are reproducible

`run_p2b_decomposition.py` recomputes the four qualification arms bitacora 04
quoted without a producer. Reconciliation vs the quoted values:

| quantity | quoted | measured | \|Δ\| |
|---|---|---|---|
| SHAPE only, level removed | 0.9596 | 0.959630 | < 0.0005 |
| SPEC_T minus H₀ (both) | 0.9594 | 0.959415 | < 0.0005 |
| COUNT+SHAPE | 0.9673 | 0.967297 | < 0.0005 |
| Δ(COUNT+SHAPE − COUNT) | +0.0273 | +0.027302 | < 0.0005 |
| Δ(SPEC_T − MINUS_H₀) | +0.0104 | +0.010446 | < 0.0005 |
| Δ(CB+SPEC_T − CB) | +0.0192 | +0.019227 | < 0.0005 |

**No findings fired** (threshold: any |Δ| > 0.005).

**Fidelity gate (new, elementwise):** the script also recomputes COUNT, BURST
and SPEC_T and asserts per-seed equality with `p2_temporal.json`:
max |diff| = **0.0** across all four metrics × 10 seeds × 3 arms. The two
artefacts are now provably the same pipeline.

## 2. The ambiguity in "SPEC_T minus H₀" is resolved by measurement

bitacora 04 never specified which half's H₀ its arm removed. Three variants
were pre-declared in the JSON config echo:

| variant | dims | AUC |
|---|---|---|
| both H₀ columns removed (**primary**) | 10 | **0.9594** ← matches the quote |
| minus inter-arrival H₀ only | 11 | 0.9642 |
| minus circadian H₀ only | 11 | 0.9692 |

The quoted number is uniquely consistent with removing **both** H₀ columns,
which is also what the stated rationale implies. Recorded so no future reader
has to guess.

## 3. A3 propagated one hop further than the review looked — fixed at the source

Notebook 04 §6.1 printed "DECLINES MONOTONICALLY in n_bins (0.0581 at 8 →
0.0248 at 48)", quoting this project's own error back at it. Corrections made
in `scripts/build_ext_04_notebook.py`, then regenerated and executed:

1. §6.1 sweep sentence rewritten from the artefact: clause (ii) **peaks at
   n_bins = 12 (+0.0626)** and declines to +0.0248 at 48; the correction is
   dated and cited in the cell itself.
2. §6 summary line "every order correlates 0.66–0.82 with log count" was also
   wrong as written — that range holds for the **inter-arrival half only**
   (circadian orders span 0.47–0.69, per `p2_temporal.json`
   `per_order_corr_with_count`). Reworded.
3. §6.2 DECOMP table is now **loaded from `p2b_decomposition.json`** with a
   loud STALE banner if the artefact is absent — a didactic notebook never
   shows an unreproducible figure.
4. New §6.4 presents the censoring probe and the WP-B framing decision
   (amendment 06 §3.4 obligation), likewise artefact-backed.
5. License block updated: the shape claim now carries its censoring bound;
   "TwiBot-20 volume unmeasured" replaced by the measured 0.61.

All five notebooks regenerate and execute (~17 s); content diffs verified to
contain only the intended changes plus execution timestamps.

## 4. Docs brought into the same commit

- `docs/04-DECISIONS.md` P2 entry: monotonicity sentence corrected inline with
  date; backfill note appended pointing at the JSON.
- `02-ext-research/EVIDENCE-INDEX.md`: P2 headline table added; WP-B and WP-C
  sections completed. Every number quoted anywhere in `02-ext-research/` now
  has a row.

## 5. What failed and was not fixed

Nothing operational. One process observation worth keeping: the original P2
session computed these arms but never persisted them — the failure was not
dishonesty but missing plumbing. The standing rule ("no number enters a
document without an EVIDENCE-INDEX row") exists because plumbing is exactly
what memory does not replace.

## 6. Multiple comparisons counted

Ten arms evaluated (three fidelity re-computations + four primary + two
sensitivity variants + CB baseline); five paired tests reported; zero
hypothesis-level claims made.
