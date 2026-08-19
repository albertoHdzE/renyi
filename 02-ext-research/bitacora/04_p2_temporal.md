# Bitácora 04 — P2, the temporal front and the test of H1

**Date:** 2026-08-18
**Branch:** `main`
**Gate G2 (H1):** **PASS** on both pre-registered clauses.
**Protocol floor 6 (burstiness):** **NOT CLEARED.**
**Artefacts:** `renyiext/features.py`, `scripts/run_p2_temporal.py`,
`results/p2_temporal.json`, `results/figures/p2_g1_spectrum.png`
**Seeds:** 42–51 (10), 5-fold stratified CV, `HistGradientBoostingClassifier`.
Two runs agree to the digit.

Per the datasaurus reporting protocol the interpretation is **split**, so that a
partial success cannot be read as the whole one. §2 is what passed. §3 is what
did not. §4 is what the picture says about the mechanism, and it is not what H1
claims.

---

## 1. Sample

`min_events = 5` on 5,301 accounts leaves **n = 4,770** (2,846 bot, 1,924 human).
Excluded: **505 bot, 26 human** — the exclusion is severely class-dependent, as
P0 predicted, and it removes the least active bots. Majority baseline falls from
0.6321 (full corpus) to **0.5966**. At `min_events = 2` (n = 4,994) the headline
difference is unchanged at +0.0369, so the exclusion is not driving the result.

## 2. What passed

| arm | dim | AUC | ±SD | TPR@1%FPR | macro-F1 | acc |
|---|---|---|---|---|---|---|
| COUNT | 1 | 0.9400 | 0.0011 | 0.141 | 0.912 | 0.916 |
| BURST | 3 | 0.9141 | 0.0013 | 0.344 | 0.854 | 0.858 |
| SHAN | 2 | 0.9318 | 0.0012 | 0.489 | 0.860 | 0.865 |
| **SPEC_T** | 12 | **0.9699** | 0.0007 | **0.779** | 0.916 | 0.919 |
| COUNT+BURST | 4 | 0.9594 | 0.0009 | 0.520 | 0.901 | 0.905 |
| COUNT+SHAN | 3 | 0.9731 | 0.0008 | 0.792 | 0.918 | 0.921 |
| **COUNT+SPEC_T** | 13 | **0.9767** | 0.0005 | 0.792 | 0.922 | 0.925 |

majority baseline **0.5966**

**H1, both clauses, 10/10 seeds, paired Wilcoxon p = 0.0020:**

| clause | Δ | verdict |
|---|---|---|
| (i) COUNT+SPEC_T − COUNT alone | **+0.0367** | **CLEARS** (floor 0.02) |
| (ii) SPEC_T − SHAN alone | **+0.0380** | **CLEARS** |

**H1 is supported as pre-registered.** The spectrum beats event count — the
feature that scored AUC 0.939 alone and is already in `META` — and it beats its
own α = 1 point.

**Stability across the swept grid.** 14 configurations of `n_bins` ∈ {8…48},
`hi` ∈ {30, 100, 400, 1000} days, `min_events` ∈ {2, 5, 10, 20}:

- clause (i): **+0.0350 to +0.0381** — every configuration clears, range 0.003
- clause (ii): **+0.0248 to +0.0626** — every configuration clears, but it
  **declines monotonically in `n_bins`** (0.0581 at 8 bins → 0.0248 at 48)

The headline uses the protocol default (24 bins, 400 d, min 5), **not** the
sweep argmax; selecting the argmax would be selecting on the outcome. The full
sweep is in `results/p2_temporal.json`.

**The operational number is the striking one.** At FPR = 1% — the deployment
regime, where the cost of a false positive is suspending a real user — COUNT
recovers **0.141** of bots and SPEC_T recovers **0.779**. That is a 5.5× gain in
the regime that matters, against a 0.03 gain in AUC. AUC understates this
difference badly, which is the reason `docs/02-PROTOCOL.md` §6 requires both.

## 3. What did NOT pass

**Protocol floor 6 — burstiness — is not cleared.**

`docs/02-PROTOCOL.md` §3 makes CV and the Fano/burstiness statistics a mandatory
floor for the temporal front, and states that a family failing its floors *is
reported as failing*. Three numbers (Goh–Barabási **B**, memory **M**, **CV**):

| comparison | Δ | p | verdict |
|---|---|---|---|
| COUNT+SPEC_T vs COUNT+BURST | +0.0173 | 0.0020 | **fails** (< 0.02) |
| COUNT+BURST+SPEC_T vs COUNT+BURST | +0.0192 | 0.0020 | **fails** (< 0.02) |

Both are significant at 10/10 wins, and both sit **below the pre-registered 0.02
effect-size floor**, which the protocol says is not claimed regardless of p.

So the honest statement is: **twelve Rényi orders add 0.019 AUC over three
classical burstiness numbers.** The spectrum is not shown to be worth its
dimensionality against the right incumbent. H1 named Shannon and count as its
floors and cleared both; the protocol named burstiness too, and it did not.

**This is the finding a reader should carry.** H1's clauses were the wrong
comparison to make it interesting — SHAN (two entropies) was too weak a
baseline, and BURST is the baseline a referee will raise.

## 4. The picture disagrees with H1's stated mechanism

H1's rationale: "human inter-event times are bursty and heavy-tailed; scheduled
bots are Poisson or periodic; these regimes differ **specifically in the tail**,
which is what α resolves." The render tests that claim and it does not survive
intact.

**The α-curves are near-parallel.** Bot and human inter-arrival curves are offset
by ~1.0–1.1 bits at *every* order, with similar slope (human falls 3.78 → 2.32,
a drop of 1.46; bot 2.69 → 1.54, a drop of 1.15). If the separation were purely
tail-resolved, the curves would *converge or cross*, not translate.

**Per-order separation is flat and every order is below the count floor.** All
six inter-arrival orders score |2·AUC−1| between 0.81 and 0.88; COUNT alone
scores 0.88. No single order beats volume.

**Every order is substantially volume.** |corr with log count|: H₀ 0.82, H₀.₅
0.73, H₁ 0.70, H₂ 0.68, H₄ 0.68, H_∞ 0.66 (inter-arrival). H₀ is the worst,
which is expected — `H₀ = log₂(occupied bins)` is bounded by
`log₂ min(n, n_bins)` and is therefore *mechanically* a function of n.

### But shape is not nothing — a decomposition

Removing the level entirely (subtract H₁ from every order, leaving only the
curve's shape) and re-running:

| arm | AUC |
|---|---|
| SPEC_T (full) | 0.9699 |
| SPEC_T minus H₀ | 0.9594 |
| **SHAPE only, level removed** | **0.9596** |
| COUNT + SHAPE | 0.9673 |

| comparison | Δ | verdict |
|---|---|---|
| COUNT+SHAPE vs COUNT | **+0.0273** | **CLEARS** |
| SPEC_T vs SPEC_T-minus-H₀ | +0.0104 | fails |

**The α-curve's shape alone, with all level information destroyed, clears the
count floor.** So the curves are less parallel than the eye reads, and there is
genuine shape information. Equally, H₀ — the single most volume-contaminated
order — contributes only +0.0104 of the spectrum's edge, so the result does not
rest on the order most exposed to R1.

**Net reading of the mechanism:** the separation is *mostly* level (which is
largely volume) plus a *real but modest* shape component. H1's specific tail
mechanism is **not** demonstrated; something weaker and less interesting — that
the α-profile carries information beyond one entropy and beyond a count — is.

## 5. An anomaly, flagged and not interpreted

Partial correlation with the label, given log count, **reverses sign** for every
circadian order:

| order | raw | given count |
|---|---|---|
| H₀_cd | −0.425 | **+0.268** |
| H₁_cd | −0.273 | **+0.269** |
| H_∞_cd | −0.257 | **+0.207** |

The inter-arrival orders do not reverse (−0.72 → −0.22, attenuated but same
sign). A sign reversal under conditioning is a suppression effect and can be
real or can be an artefact of the linear partialling. **Not interpreted here.**
It needs its own render — the circadian histograms of matched-count bots and
humans, side by side — before anything is said about it. Recorded as
exploratory; it was not predicted and is not part of H1.

## 6. A performance defect worth recording

`HistGradientBoostingClassifier` took **115.6 s** for 5 folds of 4,770 × 12 with
the default OpenMP pool, and **0.87 s** pinned to one thread — 133×, entirely
scheduling overhead. The run is now pinned inside the script rather than left to
the shell, so wall time is a property of the artefact. Determinism is unaffected.

Recorded because a 133× penalty invisible in the output is exactly the kind of
thing that quietly caps how many seeds and sweep points an experiment gets, and
therefore how much evidence it can produce.

## 7. Multiple comparisons counted

**Pre-registered, no correction:** H1 clauses (i) and (ii).

**Reported and counted:** 14 grid configurations (§2, all reported); 7 arms (§2);
4 further arms in the §4 decomposition (SPEC_T-minus-H₀, SHAPE, COUNT+SHAPE,
COUNT+BURST+SPEC_T). Total 25 configurations evaluated, all listed here or in
`results/p2_temporal.json`. Nothing was dropped.

**Exploratory, Holm-corrected within family if pursued:** §5's sign reversal;
the monotone decline of clause (ii) in `n_bins`.

## 8. Carried into P3

1. **BURST is now a first-class floor in every subsequent front**, not just the
   temporal one. §3 shows it is the binding comparison and it was nearly missed.
2. The §4 SHAPE decomposition (level removed) should become a standard arm — it
   is the cleanest separation of "shape" from "magnitude" this project has, and
   it is what the charter's motivation actually claims.
3. §5's circadian reversal needs a render before P3 uses circadian features.
4. H4 is unaffected and remains primary. Note that SPEC_T's advantage here is
   partly *level*, and level is corpus-specific — so §4 is a reason to expect
   H4 to be harder than P2's result suggests, not easier.
