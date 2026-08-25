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
(bitacora 08); **WP-D done — sentinel cells shipped, verdicts stable**
(bitacora 09): mass loss measured (bot 73 / human 17 intervals), full P2
re-run moves arms ≤ 0.0003 with all clause/floor verdicts unchanged, probe
fidelity re-run still FIRED at 1.0000 (≤ 0.004 drift). Checks 9/9 incl. P16.
**WP-E done — evaluation layer consolidated onto `renyiext.evaluate`**
(bitacora 10): regression gate vs the pre-WP-E artefact max |diff| 0.0;
`tpr01_foldmean` beside every pooled TPR; retroactive dim-matched rows —
clause (ii) `supports_clause` (+0.0376), burstiness verdict
`real_but_subfloor_not_claimable` (+0.0159); no Δ ≤ 0, no downgrade fired;
`sigma_config` beside every floor verdict.
**WP-F done — equal-window controls SPLIT H1** (bitacora 11): clause (i)
survives (+0.0423 at K=30, grows to +0.0681 at K=90; dim-matched supports);
**clause (ii) collapses (+0.0048 at K=30; dim-matched −0.0003 →
`confounded_dimensionality`) — DOWNGRADE EXECUTED**: H1-as-amended support is
now clause-(i)-only. SHAPE vs BURST+N9 survives at every K (new strongest
edge). API cap innocent: excluding the whole cap region moves COUNT −0.0133.
Unwindowed reference gated vs both committed artefacts at max |diff| 0.0.
**WP-G done — circadian reversal adjudicated KEPT** (bitacora 12): within
count-caliper strata the matched Δ(H_cd) is +0.25…+0.30 bits on every order
(7–8/8 strata positive, TV ≈ 0.22) — suppression effect, mechanism cited;
timezone offset provably inert (≤ 2.8e-16, permutation invariance); fidelity
gates vs p2_temporal at 0.0. SPEC_B = alphabet(6) + mention-targets(6); no
SPEC_T circadian caveat. Open item 1 CLOSED.
**WP-H done — behavioural front built; H2 PASS, META confound downgraded**
(bitacora 13): H2 directional gates pass (bot H₂ median 0.0 vs 1.20; mention
H₀ 1 vs 7.24 targets; p ≈ 0); SPEC_B beats COUNT everywhere (+0.0211 /
+0.0414 / +0.0539, all 10/10) but META-lite is near-ceiling (0.9972) and
every vs-META dim-matched row fires `confounded_dimensionality` — DOWNGRADE
EXECUTED (not claimable beyond the incumbent on Cresci-15; transfer is WP-N's
question). Mention block excludes 1,490 bots / 11 humans (class-dependent —
carried as a caveat). Shuffle null exactly silent. D11 quote share 11.64 %
confirmed.
**WP-I done — text front: spectrum beats its Shannon slice at last**
(bitacora 14, exploratory): SPEC_X_CHAR 0.9889, SPEC_X 0.9940; dim-matched
vs Shannon +0.3332/+0.0537/+0.0466/+0.0318 all `supports_clause`; all 20
comparisons survive Holm (adj. p 0.0391); beats COUNT/TOKENS (+0.054/+0.045),
edge grows on the ≥512-token subsample (+0.0636); URL stripping inert
(−0.0077). vs META still `confounded_dimensionality` (−0.0036 — nearly at
the ceiling). Length control: word diversity volume-confounded, char usage
not. Loader generalised to `events.load_cresci_text_side`.
**Correction (bitacora 15) on the WP-I reading:** given TOKENS, the
spectrum's residual over Shannon is **+0.0166 — subfloor** (the +0.3332 is
largely length-mediated); Holm survival is bounded by the 10-seed p-floor
(≡ 10/10 wins, family ≤ 25); "tail orders carry it" is render-suggested,
not ablated.
Next: **WP-J — tail-statistic arms (TAIL, SURV; property P17)**.

## Read first

1. [PLAN-02-ext-research.md](../PLAN-02-ext-research.md) — the governing plan
2. [bitacora/11_WP_F_truncation_controls.md](bitacora/11_WP_F_truncation_controls.md) —
   equal-window split of H1; the clause-(ii) downgrade
3. [bitacora/10_WP_E_evaluation_layer.md](bitacora/10_WP_E_evaluation_layer.md) —
   evaluation layer; dim-matched semantics binding on all fronts
4. [bitacora/06_amendment_censoring.md](bitacora/06_amendment_censoring.md) —
   the censoring amendment, now partially discharged by WP-F
5. [docs/00-CHARTER.md](docs/00-CHARTER.md) — hypotheses (H4 framing may be
   amended by WP-B)

## Confirmed state

**P0.** Snowflake decode confirmed (D1) and gated; D9 discards 63,830
pre-snowflake tweets. Corpus 5,301 users / 2,763,927 events. Count alone = AUC
0.939.

**P1.** Spectrum estimator, 8/8 properties. Base 2, plug-in, no bias correction
by design (D3'). Series expansion near alpha=1.

**P2.** H1 passed both pre-registered clauses (+0.0367 over count, +0.0380 over
Shannon; 10/10 seeds, p = 0.0020); burstiness floor NOT cleared (+0.019 < 0.02).
TPR@1%FPR 0.141 → 0.779. **Reading now split by WP-F (bitacora 11):** clause
(i) survives equal windows (+0.0423 at K=30, +0.0681 at K=90; dim-matched
supports); **clause (ii) is confounded — +0.0048 at K=30, dim-matched −0.0003
(`confounded_dimensionality`); support DOWNGRADED** — H1-as-amended stands on
clause (i) only. The level-removed SHAPE arm beats BURST+NOISE(9) under equal
windows at every K (+0.0234 at K=30) — the strongest surviving edge. API cap
not a factor (COUNT −0.0133 with the whole cap region removed).

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

**WP-E (evaluation layer).** `renyiext/evaluate.py` is the single evaluation
module (both temporal producers import it; probe/preflight harnesses
deliberately untouched — closed phases, §9.7). Regression gate: every legacy
number in `p2_temporal.json` bit-identical (max |diff| 0.0, tolerance 1e-4);
only additive fields differ (`tpr01_foldmean`, two noise arms, sweep
`auc_count_burst`, `dim_matched`, three `sigma_config`s). Dim-matched rows
(plan §8 D2, semantics WP-E task 3): SPEC_T vs SHAN+NOISE(10) **+0.0376**
10/10 p = 0.002 → supports clause (ii); COUNT+SPEC_T vs COUNT+BURST+NOISE(9)
**+0.0159** 10/10 p = 0.002 → real-but-subfloor, **not claimable** at the
registered floor (was already failing as gated; nothing downgraded).
Padded floors are not handicapped: SHAN+NOISE(10) ≈ SHAN (0.9325 vs 0.9320),
CB+NOISE(9) slightly above CB (0.9605 vs 0.9594). σ_cfg beside every verdict:
clause (i) 0.0007, clause (ii) 0.0094, burstiness 0.0006.

## Single next action

**WP-J — tail-statistic arms (TAIL, SURV)** (method upgrade, independent of
H/I/K/L): create `renyiext/tailstats.py` — TAIL = Hill estimator per §8 D8
(α̂ clipped to [0.3, 20], k recorded; plug-in bias stance in the docstring);
SURV = empirical survival P(Δt > t) at t ∈ {1 h, 1 d, 7 d} (fixed
corpus-wide lags, printed, G3). Property **P17**: synthetic Pareto(ν),
n = 500, median α̂ within 15 % of ν for ν ∈ {1.2, 1.5, 2.0}; small-n bias
direction documented. Arms: TAIL, SURV, TAIL+SURV vs BURST and COUNT floors,
dim-matched (SURV 3-d vs BURST 3-d; TAIL 1-d vs COUNT 1-d). Added to
`results/p2c_truncation.json` so WP-A's probe covers them; if TAIL+SURV
beats SPEC_T's dim-matched margin, HANDOFF's mechanism narrative updates
(separability is tail-magnitude, not multifractal shape) — recorded either
way.

## Open items

1. ~~Circadian sign reversal~~ **CLOSED by WP-G** (bitacora 12): suppression
   effect, adjudicated kept; SPEC_B unaffected.
2. `quote` post-type share implausible → plan WP-H decision D11 collapse
   (executes in WP-H, next).
3. ~~TwiBot-20 volume confound~~ **CLOSED by WP-B** (bitacora 07): volume weak
   on TB20; H4 framing fixed.
4. H4 harder than P2 suggests — strengthened by bitacora 06 and now sharpened
   by WP-F (bitacora 11): the surviving SPEC edges are clause-(i)-anchored and
   SHAPE-based, so the transfer claim should feature those arms; META's
   target-side weakness means an H4 pass must be argued via degradation
   comparison, not via META collapse (bitacora 07 §4).
5. ~~Overflow-cell fix + probe re-run~~ **CLOSED by WP-D** (bitacora 09).
6. ~~Notebook 04 monotonicity error + missing probe/framing~~ **CLOSED by
   WP-C** (bitacora 08): corrected at the builder; constants artefact-backed.
7. LR-side diagnostics under transfer must use marginal-recalibrated features
   (bitacora 07 §2) → carried into WP-N.

## Performance note

Pin `OMP_NUM_THREADS=1`. HGB took 115.6 s vs 0.87 s for the same 5 folds
otherwise — 133x, pure scheduling overhead. Now pinned inside the scripts.
