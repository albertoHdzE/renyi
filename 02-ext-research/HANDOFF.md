# HANDOFF

**Updated:** 2026-08-24
**Branch:** `main` *(the didactic-notebooks work that sat on the local branch
`ext-research-notebooks` was fast-forwarded into `main` on 2026-08-24)*
**Programme:** `PLAN-02-ext-research.md` v1.1 at repo root governs execution.
**State:** P0–P2 as recorded below; then **WP-A (censoring probe) done —
AMENDMENT FIRED** (bitacora 05/06): window truncation alone separates at AUC
0.9224–1.0000 through this pipeline, so every "shape" reading of P2 is bounded
by a censoring ceiling until WP-F's equal-window arms report.
Next: **WP-B — TwiBot-20 preflight** (plan §5).

## Read first

1. [PLAN-02-ext-research.md](../PLAN-02-ext-research.md) — the governing plan
2. [bitacora/06_amendment_censoring.md](bitacora/06_amendment_censoring.md) —
   the binding downgrade of P2's "shape" reading
3. [bitacora/04_p2_temporal.md](bitacora/04_p2_temporal.md) — §3/§4
   qualifications; §2 alone will mislead you
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

## Single next action

**WP-B — TwiBot-20 preflight** (`data/raw/bot/twibot-20/` already on disk):
volume landscape, AUC(volume alone), retention; fires the H4 vs H4′ framing
amendment **before** any front is built (plan §5 WP-B).

## Open items

1. Circadian sign reversal → plan WP-G (blocks SPEC_B).
2. `quote` post-type share implausible → plan WP-H decision D11 collapse.
3. TwiBot-20 volume confound → **WP-B, now next.**
4. H4 harder than P2 suggests — strengthened by bitacora 06: level is corpus-
   specific AND censoring-bounded; plan WP-N executes under WP-B's framing.
5. Overflow-cell fix (review C1) → plan WP-D; probe re-run after it.
6. Notebook 04 prints bitacora 04's monotonicity error and must gain the probe
   result → plan WP-C audit task.

## Performance note

Pin `OMP_NUM_THREADS=1`. HGB took 115.6 s vs 0.87 s for the same 5 folds
otherwise — 133x, pure scheduling overhead. Now pinned inside the scripts.
