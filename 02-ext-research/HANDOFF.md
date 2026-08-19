# HANDOFF

**Updated:** 2026-08-18
**Branch:** `main`
**State:** P0, P1, P2 done. **H1 supported; burstiness floor not cleared.**
Next: **P3, behavioural and text fronts**.

## Read first

1. [bitacora/04_p2_temporal.md](bitacora/04_p2_temporal.md) — §3 and §4 are the
   qualifications; §2 alone will mislead you
2. [bitacora/02_h1_amendment.md](bitacora/02_h1_amendment.md) — H1 as in force
3. [docs/00-CHARTER.md](docs/00-CHARTER.md) — the remaining hypotheses

## Confirmed state

**P0.** Snowflake decode confirmed (D1) and gated; D9 discards 63,830
pre-snowflake tweets. Corpus 5,301 users / 2,763,927 events. Count alone = AUC
0.939.

**P1.** Spectrum estimator, 8/8 properties. Base 2, plug-in, no bias correction
by design (D3'). Series expansion near alpha=1.

**P2.** H1 passes both clauses: (i) +0.0367, (ii) +0.0380, 10/10 seeds,
p = 0.0020, stable across all 14 swept grid configurations. TPR@1%FPR 0.141
(count) -> 0.779 (spectrum).

**But** twelve Renyi orders add only **+0.019** over three burstiness numbers
(B, M, CV) — below the pre-registered 0.02 floor. Protocol floor 6 is NOT
cleared, and the rendered alpha-curves are near-parallel, so H1's tail-resolution
mechanism is not demonstrated. The level-removed SHAPE arm does clear over count
(+0.0273), so shape information is real but modest.

## Single next action

**P3 — behavioural and text fronts** (SPEC_B, SPEC_X), testing H2.

Per **decision D10**, from this phase onward every front carries BURST as a
first-class floor and the level-removed SHAPE arm as a standard arm. P2 nearly
missed the binding comparison by not doing this.

Before using circadian features in P3: render the circadian histograms of
matched-count bots and humans side by side. Every circadian order reverses sign
under conditioning on count and this is uninterpreted
([bitacora/04](bitacora/04_p2_temporal.md) §5).

## Open items

1. Circadian sign reversal, unrendered and uninterpreted. Blocks circadian use.
2. `quote` post-type rule gives an implausible 11.6% share — blocks SPEC_B.
3. TwiBot-20 volume confound unmeasured; needed for H4/P6.
4. H4 is likely *harder* than P2 suggests: SPEC_T's edge is partly level, and
   level is corpus-specific.

## Performance note

Pin `OMP_NUM_THREADS=1`. HGB took 115.6 s vs 0.87 s for the same 5 folds
otherwise — 133x, pure scheduling overhead. Now pinned inside the scripts.
