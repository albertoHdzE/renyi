# Bitácora 16 — WP-J, tail-statistic arms: the mechanism narrative updates (and it is SURV, not the Hill index)

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** P17 (new property) — **10/10 checks pass**. Regression gate on the
corpus artefact: **max |diff| 0.0 over 3,181 numeric leaves**; probe fidelity
gate: **exact** (every pre-existing reading reproduced, trigger still FIRED
at 1.0000). Two runs of each producer byte-identical (S2.6).
**Artefacts:** `renyiext/tailstats.py` (new), `renyiext/checks.py` (P17),
`scripts/run_p2c_probe.py` (additive extension), `scripts/run_p2f_truncation.py`
(additive extension + self-regression gate), `results/p2c_probe.json`,
`results/p2c_truncation.json` (both regenerated, legacy values bit-identical).
**Seeds:** 42–51.

---

## 1. The verdict (plan task 4): TAIL+SURV beats SPEC_T's dim-matched margin at every K

| K | TAIL+SURV vs BURST+NOISE(1) | SPEC_T vs BURST+NOISE(9) |
|---|---|---|
| 7 | **+0.0689** | +0.0634 |
| 14 | **+0.0440** | +0.0314 |
| **30** | **+0.0555** | +0.0361 |
| 90 | **+0.0546** | +0.0470 |

All `supports_clause` (10/10, p = 0.0020). **The mechanism narrative updates
per the pre-registered rule: separability over the burstiness floor is
tail-magnitude, not multifractal shape.**

## 2. The honest nuance: it is SURV that carries it, not the Hill index

- **TAIL alone vs COUNT = −0.1208** at K = 30: the fitted tail index is far
  *worse* than raw volume as a separator (bots' rapid posting → light
  inter-arrival tails → high α̂ → the index tracks the volume story in
  reverse).
- **SURV alone vs BURST = +0.0564**, and SURV's AUC (0.9260) ≈ TAIL+SURV's
  (0.9261): the margin is carried by three simple survival proportions at
  the fixed lags {1 h, 1 d, 7 d} — nothing fitted.

The updated narrative, stated precisely: *separability over the burstiness
floor is carried by tail-magnitude statistics in their simplest form —
empirical survival proportions — not by the fitted tail index and not by
multifractal shape.* TAIL is retained as the registered arm (and its
failure vs volume is itself informative: the classes differ in how much
they post, not in the shape of their tail decay once volume is separated).

## 3. Probe coverage (plan task 3, "so WP-A's probe covers them too")

The probe now computes TAIL/SURV/TAIL+SURV cells: TAIL+SURV spans
0.549–0.990 across the nine generator × window cells — below the
COUNT+SPEC_T ceiling (0.978–1.000) at short windows, so the censoring null
bounds the tail reading too. **Re-run sanction reading:** §9.7 forbids
re-running closed phases *to get a better number*; WP-J's own task text
pre-registers this extension, and the recursive fidelity gate asserts every
pre-existing reading reproduced exactly (worst |diff| 0.0) before the
artefact was rewritten — the trigger verdict is unchanged (FIRED, 1.0000).
Recorded here as the decision of record.

## 4. P17 and the measured bias direction

`checks.py` P17: Pareto(ν), n = 500, k = 125, 400 draws — median α̂ within
15 % of ν for ν ∈ {1.2, 1.5, 2.0}. Measured bias: **upward +0.9 %…+1.8 %**
(median 1.2210 / 1.5196 / 2.0183) — the Jensen effect of the reciprocal
(E[1/H] > α), matching D8's pre-stated "upward-biased at small n". Docstring
states the stance: plug-in, no correction. Zero-length gaps are excluded
from TAIL only (no tail information; SURV keeps all gaps).

## 5. Decisions taken under the ambiguity protocol

1. **Corpus-side filter replication**: `run_p2f` recomputes the windowed
   event lists with the same three-line filter as
   `temporal_blocks_windowed` and asserts alignment elementwise via the
   COUNT block (log1p of kept in-window counts) — the shared feature module
   stays untouched, and any drift fails loud.
2. **Salt continuity**: BURST+NOISE(1) takes salt 4, continuing this
   producer's declared sequence (0–3 taken); echoed in the JSON
   `dim_matched.tail_knobs`.
3. **σ_config axis** unchanged: population SD across the four published K
   values, now covering every comparison key including the tail rows.

## 6. What failed and was not fixed

One NameError in the probe extension (gate referenced `path` before its
definition) and one malformed lambda draft in the corpus extension — both
fixed before any artefact was written. Nothing outstanding.

## 7. Multiple comparisons counted

Corpus: 6 new comparisons × 4 K values (4 gated + 2 matched), σ_cfg over the
K sweep for all; probe: 3 new arms × 9 cells. P17: 3 ν values × 400 draws.
The mechanism verdict is the pre-registered reading of one comparison pair;
no floor or threshold moved.
