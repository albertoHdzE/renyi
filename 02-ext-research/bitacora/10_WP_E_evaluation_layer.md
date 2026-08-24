# Bitácora 10 — WP-E, the evaluation layer and what dimension matching does to P2

**Date:** 2026-08-24
**Branch:** `main`
**Gate:** G2 (regression gate, elementwise on JSON arrays) + G3 (noise knobs
printed in the config echo) — **gate PASS at max |diff| 0.0**; property
checks **9/9** before and after.
**Artefacts:** `renyiext/evaluate.py` (new), `renyiext/__init__.py`,
`scripts/run_p2_temporal.py` (refactored), `scripts/run_p2b_decomposition.py`
(consolidated), `results/p2_temporal.json` (regenerated),
`results/p2b_decomposition.json` (regenerated).
**Seeds:** 42–51 everywhere; two consecutive runs of each producer
byte-identical (S2.6).

---

## 1. What shipped

`renyiext/evaluate.py` — one shared evaluation module, generalised from
`run_p2_temporal.py`: `tpr_at_fpr`, `eval_arm`, `run_arms`, `paired`,
`partial_corr`, plus the D2 machinery `noise_padding(X_floor, k, seed,
arm_index)` / `dim_matched_arm(...)` and `interpret_dim_matched` /
`sigma_config`. Layering: features → models → **evaluate**; the module
imports nothing from the package. `eval_arm` additionally returns
`tpr01_foldmean` — mean over folds of TPR@1%FPR computed within each fold
(review C3: the pooled statistic mixes calibrations); the pooled field is
kept for continuity with every published number.

Both temporal producers now import it. The verbatim-copy note in
`run_p2b_decomposition.py`'s docstring (WP-C) is discharged. The probe
(`run_p2c_probe.py`) and preflight (`run_p6b_tb20_preflight.py`) harnesses
are deliberately untouched: their phases are closed and §9.7 forbids
re-running them; refactoring their code without re-running would leave an
artefact no committed code reproduces. Their duplication is noted, not
consolidated.

## 2. Regression gate (G2): the refactor moved nothing

The gate compares every legacy array against the artefact on disk before
overwriting (tolerance 1e-4; acceptance "unchanged to 4 decimals"). Result:
**max |diff| 0.0** — AUC, pooled TPR, macro-F1, accuracy per arm per seed;
all 14 sweep rows; all three clause deltas; baselines. Structural diff vs
the pre-WP-E artefact: **zero removed paths, zero changed values**, 229
added paths (fold-means, two noise arms, sweep `auc_count_burst`,
`dim_matched`, three `sigma_config`s). Byte-identity verified across two
consecutive runs including the new block. p2b's fidelity gate re-passed at
0.0, now also covering the fold-mean field.

## 3. Retroactive dim-matched rows — semantics evaluated AND executed

Plan §8 D2 controls, knobs formula-determined (`k = dim(family) −
dim(floor)`), salts fixed by declaration order (G3 echo in the JSON):
`SHAN+NOISE(10)` arm_index 0; `COUNT+BURST+NOISE(9)` arm_index 1;
draws from `default_rng(seed*1000 + arm_index)`.

| matched row | Δ | wins / p | σ_cfg | rule fired |
|---|---|---|---|---|
| SPEC_T vs SHAN+NOISE(10) | +0.0376 ± 0.0017 | 10/10, 0.0020 | 0.0094 | `supports_clause` |
| COUNT+SPEC_T vs COUNT+BURST+NOISE(9) | +0.0159 ± 0.0011 | 10/10, 0.0020 | 0.0006 | `real_but_subfloor_not_claimable` |

Execution, per the pre-registered rules:

- **Rule 3 fired for clause (ii):** at equal dimensions SPEC_T beats its
  Shannon floor by +0.0376 (vs +0.0381 unmatched). Clause (ii)'s support
  survives dimensionality matching; recorded as supporting.
- **Rule 2 fired for the burstiness verdict:** +0.0159 < 0.02 under matched
  dimensions ⇒ the clause resting on that comparison is recorded **not
  claimable at the registered floor** — executed here, in FINDINGS F7, and
  in HANDOFF. It was already failing as gated (+0.0170), so no support was
  removed; the matched reading says the shortfall is not a floor-handicap
  artefact either (see below).
- **Rule 1 did not fire anywhere:** no Δ ≤ 0 ⇒ **no downgrade** of any
  hypothesis clause. Stated explicitly because the plan demands the firing
  be executed rather than noted — this session's execution is: rule 2's
  recording, made in three places, plus this explicit non-firing statement
  for rule 1.

Mechanism note (G4): the padded floors were **not handicapped** —
SHAN+NOISE(10) scores 0.9325 vs SHAN's own 0.9320, and CB+NOISE(9) scores
0.9605, slightly *above* real COUNT+BURST (0.9594). HGB ignores pure-noise
columns at n = 4,770, so these rows measure equal-dimension information, not
baseline sabotage. That is exactly why the burstiness verdict's subfloor
reading has teeth: give BURST's composite nine noise dims instead of its
three real ones plus SPEC_T's thirteen, and SPEC_T still cannot open a 0.02
gap.

## 4. sigma_config beside every floor verdict (D3)

Population SD (ddof = 0) of each delta across the full published sweep:

| verdict | delta (seed SD) | σ_cfg |
|---|---|---|
| clause (i) CS − C | +0.0364 (0.0011) | 0.0007 |
| clause (ii) S − SHAN | +0.0381 (0.0013) | 0.0094 |
| burstiness CS − CB | +0.0170 (0.0008) | 0.0006 |

Config uncertainty is an order of magnitude below every delta except
burstiness's, where both σs are ≪ 0.02 anyway — the failure there is real,
not noise in either sense. To give the burstiness verdict a *full-sweep*
σ_cfg (none existed: the sweep never carried CB), each sweep row gained an
`auc_count_burst` column — an addition required by D3's definition ("across
the full published config sweep"), not a new sweep. Existing sweep numbers
unchanged (gate-checked).

## 5. What the fold-mean TPR says (C3), recorded not acted on

Pooled vs fold-mean TPR@1%FPR differs by up to ~0.02 per arm (BURST 0.344 →
0.327; SPEC_T 0.789 → 0.767; SHAN 0.503 → 0.513) — direction varies by arm,
so it is calibration mixing, not bias against the method. All published TPR
numbers remain pooled-defined; from WP-E on, producers emit both and any
future claim should state which statistic it quotes.

## 6. Decisions taken under the ambiguity protocol

1. **No new property-check number consumed.** G-correctness requires a
   property for every new *mathematical estimator*; the evaluation layer is
   protocol plumbing around sklearn metrics, not an estimator, so P17 stays
   reserved for WP-J's pre-registered Hill check. The elementwise burden
   here is carried by the regression gate (§2), which is stronger than any
   synthetic property for this purpose. (Most conservative option: do not
   renumber another WP's pre-registration.)
2. **p2b consolidated beyond task 2's letter.** Task 2 names only
   `run_p2_temporal.py`; HANDOFF's next-action wording ("consolidate eval
   onto renyiext.evaluate"), this WP's Goal ("one shared module"), and p2b's
   own committed docstring promise all sanction consolidating it too. Done
   with its fidelity gate as proof of harmlessness.
3. **Probe/preflight untouched** — §9.7 (no re-runs of closed phases)
   outweighs deduplication there.

## 7. What failed and was not fixed

Nothing operational. One process near-miss worth keeping: the first attempt
to launch run 3 detached a background job from the shell wrapper, which
killed it mid-run (log ends after cache load); re-run in foreground
completed and proved byte-identity. Lesson: this environment's shells reap
background children on timeout — long runs go foreground-with-timeout or
properly daemonised.

## 8. Multiple comparisons counted

Nine headline arms evaluated (seven legacy + two D2 floors); fourteen sweep
rows × five arms; five paired tests reported (three gated + two dim-matched);
two pre-registered interpretation rules evaluated; zero hypothesis-level
claims changed (clause support statuses: (ii) confirmed-supporting,
burstiness already-failing now formally not-claimable).
