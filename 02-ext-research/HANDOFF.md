# HANDOFF

**Updated:** 2026-08-24
**Branch:** `main` *(the didactic-notebooks work that sat on the local branch
`ext-research-notebooks` was fast-forwarded into `main` on 2026-08-24)*
**Programme:** `PLAN-02-ext-research.md` v1.1 at repo root governs execution.
**State:** P0–P2 as recorded below; **WP-A done — AMENDMENT FIRED** (bitacora
05/06): window truncation alone separates at AUC 0.9224–1.0000 through this
pipeline, so every "shape" reading of P2 is bounded by a censoring ceiling
until WP-F's equal-window arms report. **WP-B done — H4 stands as chartered**
(bitacora 07): TB20 volume AUC 0.6073 ≪ 0.85. **WP-C done — A1/A3 closed**
(bitacora 08): all five quoted qualification numbers reproduce exactly
(`run_p2b_decomposition.py`, fidelity gate 0.0); notebook 04 corrected and now
loads its constants from artefacts.
Next: **WP-D — estimator defects** (overflow cell, P8′ grid, hygiene; then the
WP-A probe fidelity re-run).

## Read first

1. [PLAN-02-ext-research.md](../PLAN-02-ext-research.md) — the governing plan
2. [bitacora/08_WP_C_evidence_chain_repair.md](bitacora/08_WP_C_evidence_chain_repair.md) —
   ledger closed; notebook audit trail
3. [bitacora/07_WP_B_tb20_preflight.md](bitacora/07_WP_B_tb20_preflight.md) —
   framing decision binding for WP-N
4. [bitacora/06_amendment_censoring.md](bitacora/06_amendment_censoring.md) —
   the binding downgrade of P2's "shape" reading
4. [docs/00-CHARTER.md](docs/00-CHARTER.md) — hypotheses (H4 framing may be
   amended by WP-B)

## Confirmed state

**P0.** Snowflake decode confirmed (D1) and gated; D9 discards 63,830
pre-snowflake tweets. Corpus 5,301 users / 2,763,927 events. Count alone = AUC
0.939.

**P1.** Spectrum estimator, 8/8 properties. Base 2, plug-in, no bias correction
by design (D3'). Series expansion near alpha=1.

**P2.** H1 passed both pre-registered clauses (+0.0367 over count, +0.0380 over
Shannon; 10/10 seeds, p = 0.0020); burstiness floor NOT cleared (+0.019 < 0.02).
TPR@1%FPR 0.141 → 0.779. **Reading now bounded:** see bitacora 06 — a
same-generator/different-window null reaches AUC ≈ 1.0 through this pipeline,
so "shape" is not yet separable from censoring.

**WP-A (probe).** 27/27 cell-metric readings ≥ 0.9224; trigger fired at 1.0000
(`results/p2c_probe.json`, bitacora 05). Re-run obliged after WP-D's overflow
fix (plan WP-A task 4).

**WP-B (TB20 preflight).** statuses AUC 0.6073 ± 0.0022 → **H4 as chartered**
(no amendment); followers strongest scalar at 0.7414, still ≪ threshold;
TB20-labelled is bot-majority 0.5572; z-score claim verified corpus-level,
labelled subset drifts (sd ≤ 3.76); R8 scale-artefact refuted for HGB
(0.7864 naive vs 0.7859 recalibrated). Text availability and per-user tweet
counts unrecoverable from the BotRGCN tensors — recorded limitation.

**WP-C (evidence repair).** All five quoted qualification numbers reproduce
exactly from `run_p2b_decomposition.py` → `p2b_decomposition.json`
(max |Δ| < 0.0005); fidelity gate vs `p2_temporal.json` max |diff| 0.0.
"SPEC_T minus H₀" identified as the both-columns variant by sensitivity rows.
Notebook 04 §6.1 monotonicity corrected at the builder; §6.2 constants now
JSON-derived with STALE fallback; new §6.4 carries the probe + framing.

## Single next action

**WP-D — estimator defects**: overflow cell in `log_bin_counts` (+ property
P16), P8′ pinned grid, hygiene items, full P2 re-run with old→new table,
overflow-mass render, then the WP-A probe fidelity re-run (plan §5 WP-D).

## Open items

1. Circadian sign reversal → plan WP-G (blocks SPEC_B).
2. `quote` post-type share implausible → plan WP-H decision D11 collapse.
3. ~~TwiBot-20 volume confound~~ **CLOSED by WP-B** (bitacora 07): volume weak
   on TB20; H4 framing fixed.
4. H4 harder than P2 suggests — strengthened by bitacora 06; now also:
   META's target-side weakness means an H4 pass must be argued via degradation
   comparison, not via META collapse (bitacora 07 §4).
5. Overflow-cell fix (review C1) → **WP-D, now next**; probe re-run after it.
6. ~~Notebook 04 monotonicity error + missing probe/framing~~ **CLOSED by
   WP-C** (bitacora 08): corrected at the builder; constants artefact-backed.
7. LR-side diagnostics under transfer must use marginal-recalibrated features
   (bitacora 07 §2) → carried into WP-N.

## Performance note

Pin `OMP_NUM_THREADS=1`. HGB took 115.6 s vs 0.87 s for the same 5 folds
otherwise — 133x, pure scheduling overhead. Now pinned inside the scripts.
