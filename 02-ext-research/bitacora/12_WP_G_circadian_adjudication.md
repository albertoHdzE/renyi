# Bitácora 12 — WP-G, the circadian reversal is a suppression effect: adjudicated kept

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** G1 (hour-of-day histograms, overall + per-stratum, UTC and both
offsets) + G2 (strata enumerated with sizes; two fidelity gates at 0.0) +
G3 (offset swept and proven inert) + G4 (count-caliper strata remove the
volume world).
**Artefacts:** `scripts/run_p3a_circadian.py` (new),
`results/p3a_circadian.json`, `figures/p3g_circadian_utc.png`,
`figures/p3g_circadian_local.png`, `figures/p3g_circadian_local_plus2.png`.
**Randomness:** none (descriptive statistics; no classifier, no seeds). Two
runs byte-identical (S2.6). Property checks 9/9.

---

## 1. The anomaly, and the branch that fired

`p2_temporal.json`'s circadian half shows a sign reversal: raw correlations
of every H_α(cd) with the bot label are negative (−0.42…−0.26) but the
given-count partials are positive (+0.21…+0.27); the inter-arrival half shows
no such flip. The pre-registered rule (plan WP-G task 2) asked whether the
reversal survives count matching stably across timezone offsets.

**Branch: `kept_suppression_explained`.** Within the D4 count-caliper strata
(8 valid after merging; sizes b412/h36, b863/h25, b518/h28, b452/h49,
b411/h66, b106/h373, b55/h420, b29/h927 — the corpus's count imbalance made
low-count strata bot-dominated and high-count strata human-dominated, all
≥ 20 per class), the matched difference is **positive for every order**:

| matched Δ (bot − human), mean over strata | H_0 | H_0.5 | H_1 | H_2 | H_4 | H_inf |
|---|---|---|---|---|---|---|
| bits | +0.2525 | +0.2757 | +0.2882 | +0.3018 | +0.3016 | +0.2615 |

7–8 of 8 strata positive per order (frac 0.75–1.0); per-stratum TVs
0.19–0.25 (mean 0.2228, overall 0.1969) — far above the 0.05 ambiguity
threshold, so the signal is not marginal. The one negative stratum
([+4.43, +5.79) log-count: 106 bots / 373 humans) is a minority, not a flip.

**Mechanism, one sentence (cited per the rule):** humans post more, and
posting more spreads the hour-of-day distribution, so raw hour-entropy
tracks volume downward; at matched count bots spread their posting hours
more than humans — the matched histograms show the excess night mass
directly (figures).

Consequences per the rule: circadian orders **kept**; SPEC_B remains
alphabet(6) + mention-targets(6) as WP-H defines it; P2's SPEC_T circadian
half carries **no caveat**.

## 2. The timezone offset is provably inert — recorded, not hidden

The plan's drop-branch includes "flips with timezone offset". Measured across
UTC/+1 h/+2 h, every statistic agrees to **2.8e-16** — float noise. This is
structural, not coincidental: a constant hour shift is a cyclic relabelling
of the 24 bins, and TV, Rényi entropies and correlations are all
permutation-invariant (the same property P6 establishes for the spectrum).
So "flips with offset" is unreachable *by construction*, and the
adjudication rests entirely on the matching leg. The sweep is retained (G3)
as the proof of inertness rather than dropped as useless: the plan's question
"does the conclusion depend on CET vs CEST vs UTC?" has the strongest
possible answer — no, provably, for every statistic the rule uses. Any
future analysis that needs a timezone-sensitive statistic must anchor hours
to an external template (e.g., local-daytime masks); none is registered.

## 3. Fidelity to the committed record

Two free elementwise gates, both at **max |diff| 0.0**: (a) the recomputed
circadian spectra at offset 0 equal `p2_temporal.json`'s SPEC_T cd half
(4,770 × 6); (b) the overall raw/given-count partials at offset 0 reproduce
its `partial_correlations` entries. This producer is provably the same
pipeline whose anomaly it adjudicated.

## 4. Decisions taken under the ambiguity protocol

1. **Branch evaluation order:** the ambiguity predicate (mean stratum TV
   < 0.05 at every offset) is evaluated first — it is the no-signal case and
   would also fail the "stable difference" test; ordering it first is the
   conservative reading. Recorded in the JSON `operationalisation` block.
2. **"Agreeing with the conditioned sign"** = matched delta positive for all
   six orders (the conditioned coefficients are all positive); "stable" =
   same sign across all three offsets and ≥ half the strata positive per
   order and offset. Both printed with their values.
3. **Third figure:** the plan names `p3g_circadian_{utc,local}.png` while
   acceptance demands renders for UTC *and both offsets*; +2 h is rendered
   as `p3g_circadian_local_plus2.png` rather than cramming both offsets into
   one unreadable figure. Additive, nothing renamed.

## 5. What failed and was not fixed

One numpy idiom bug (`all(d > 0 …)` over arrays → ambiguous truth value)
crashed the first invocation before any output existed; fixed
(`(d > 0).all()`). Nothing else. The empty top-row panels in the figures are
cosmetic (overall panel shares the grid with strata small-multiples) and are
left as-is.

## 6. Multiple comparisons counted

6 orders × 3 offsets × (overall + 8 strata) correlations reported (raw and
given-count = 336 numbers, all in the JSON, none selected on); 3 predicates
evaluated for the branch; 1 branch fired of 3 pre-registered. No
hypothesis-level claim beyond the adjudication itself.
