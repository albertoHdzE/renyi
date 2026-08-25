# HANDOFF

**Updated:** 2026-08-25
**Branch:** `main` *(the didactic-notebooks work that sat on the local branch
`ext-research-notebooks` was fast-forwarded into `main` on 2026-08-24)*
**Programme:** `PLAN-02-ext-research.md` v1.1 at repo root governs execution.
**State:** **WP-A…WP-L done; WP-N EXECUTED SCOPED (bitacora 19; corrections
in bitacora 20) — the plan's work packages are complete.** Four doors were
checked for a modality-bearing TwiBot-20; none has one: (1) BotRGCN HF
mirror — 9 files, all tensors/json, none temporal or textual
(`tweets_tensor.pt` is dense pooled 768-d embeddings, non-zero row fraction
1.000000); (2) raw TwiBot-20 on GitHub — gated by its authors; (3) TwiBot-22
Zenodo user.json — profiles only; (4) the TwiBot-22-format conversion
archive — seven corpora, no twibot-20 (bitacora 21).
What ran: the metadata side of H4 measured at **Δ_META +0.3143 ± 0.0075**
(within 0.9974 → transfer AUC 0.6831 with **total calibration collapse** —
target accuracy 0.5585 vs majority 0.5572, macro-F1 0.3612, TPR@1%FPR
0.0137: residual ranking only); +0.3335 under marginal recalibration — R8
verdict: effect under BOTH variants, not an alignment artefact. Volume's
degradation ordering is **variant-dependent** (bitacora 20): on LR it
exceeds META's in 20/20 draws under BOTH variants; on HGB the naive arm is
transform pathology (collapsed column sd breaks tree binning) and reverses
under recalibration (−0.0124, 3/20 draws) — claimed only on LR.
G4 sanity null silent (max |Δ| 0.0088). Composition
caveat carried (−7.50 pp bot share). H4 status: **UNTESTABLE_PENDING_DATA**.
Charter success criteria revisited in FINDINGS F15. Earlier state below.
**Earlier state (2026-08-24), per-WP:** WP-A — AMENDMENT FIRED (bitacora
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
**WP-J done — MECHANISM NARRATIVE UPDATES: tail-magnitude, carried by SURV**
(bitacora 16): TAIL+SURV's dim-matched margin over the burstiness floor
beats SPEC_T's at every K (+0.0555 vs +0.0361 at K=30; all supports_clause).
Honest decomposition: SURV (3 survival proportions at {1h, 1d, 7d}) carries
it (0.9260 alone ≈ TAIL+SURV 0.9261); **TAIL alone is worse than volume**
(vs COUNT −0.1208). P17 added (Hill recovers Pareto within 15 %, measured
Jensen bias upward +0.9…+1.8 %); checks 10/10. Probe extended with tail
cells (fidelity exact, trigger unchanged); corpus artefact regression gate
0.0 over 3,181 leaves.
**WP-K done — era shift degrades nothing SPEC; preview guard not crossed by
any SPEC family** (bitacora 17): split 2,763 train / 2,007 test (bot 0.507→
0.719 — composition caveat leads); Δ_META − Δ_SPEC ≤ 0.0385 (max, SPEC_B_
ALPHA); fired pairs META-vs-TAIL/BURST are inverted-direction (those improve);
META transfers at ceiling (+0.0005); COUNT +0.0217 the only volume
degradation; G4 null max |Δ| 0.0158; disjointness asserted; boundary
sensitivity strong (registered date mid-gradient). All preview-labelled.
**WP-L done — H3 LEG CANCELLED by the kill criterion** (bitacora 18):
provenance branch (ii) (groups not recoverable, evidence quoted); D7
viability all 9 cells monotone-strict (NCD machinery sound); kill criterion
FIRED — **162 bots (4.83 %) survive ≥ 100 events vs required 335** (91 %
human population; the length render is the evidence). R10 BLOCKED (PyPI
`acss` 0.5.2 is a Python-2 name collision, not acss.data — recorded, not
substituted; pybdm 0.1.0 pinned, CTM coverage measured: 1D alphabets
{2,4,5,6,9}×12, 2D binary 4×4 only). P9 + P12 added (P9's plan shorthand
recorded as not holding; R6 intent asserted) — checks **12/12**.
requirements-frozen.txt created.
**WP-N executed scoped** (bitacora 19, 2026-08-25): H4 UNTESTABLE_PENDING_DATA
— metadata side measured (Δ +0.3143 ± 0.0075 naive hgb; R8 clean; null 0.0088);
Δ_ours has no target-side family on the obtainable artefacts.

## Single next action

The plan's work packages are complete. Post-plan actions, in the order a new
session should consider them:

1. **Raw TwiBot-20 acquisition** (the one gate between the programme and its
   primary claim): email the maintainer per the gating quote recorded in
   `results/p6n_transfer.json` → `scoping.raw_dataset_availability`. With
   timestamps/text/post-types, WP-N's runner extends to every family and H4's
   gap becomes computable as pre-registered.
2. Alternatively, a **pre-registered amendment naming a modality-bearing
   target corpus** — only before any such data is analysed (plan §9.3).
3. P7/P8 stretch goals and the write-up phase remain governed by
   `docs/03-PHASES.md`; FINDINGS F15 is the plan-completion statement of the
   charter's success criteria.
4. The didactic series now ends at notebook 05 ("team tour"); if WP-N is ever
   re-run unscoped, notebook 05 §7's last row and §5 should be regenerated
   from the new artefact at the builder.

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

## WP-N as planned (superseded by the scoped execution above)

The plan's task text asked for every family (META aligned per D8/R8, COUNT,
BURST, SHAN, SPEC_T + SHAPE, TAIL/SURV/TAIL+SURV, SPEC_B_ALPHA, SPEC_X) under
D6 with the R8 control, dim-matched arms, era-shift quantification and the G4
pseudo-transfer null. The modality audit collapsed "available" to
META_aligned(4) + VOL_PROFILE(1); everything else in that list is recorded
UNTESTABLE in `results/p6n_transfer.json` (`scoping.uncomputable_on_target`).
What could run did run — D6 machinery identical to WP-K's, R8 both variants
both heads (open item 7 discharged), dim-matched arms, composition caveat,
sanity null. Keep this paragraph so a future session holding raw TwiBot-20
knows exactly which pre-registered protocol to resume.

## Open items

1. ~~Circadian sign reversal~~ **CLOSED by WP-G** (bitacora 12): suppression
   effect, adjudicated kept; SPEC_B unaffected.
2. ~~`quote` post-type share implausible~~ **CLOSED by WP-H** (bitacora 13):
   D11 collapse executed; share-after-collapse reported.
3. ~~TwiBot-20 volume confound~~ **CLOSED by WP-B** (bitacora 07): volume weak
   on TB20; H4 framing fixed.
4. H4 harder than P2 suggests — sharpened again by WP-N (bitacora 19): the
   surviving SPEC edges are clause-(i)-anchored and SHAPE-based, and META's
   transfer degradation is now a **measured +0.3143**, so an H4 pass would be
   argued via the degradation comparison — but the Δ_ours side remains
   uncomputable until the data exists (item 8).
5. ~~Overflow-cell fix + probe re-run~~ **CLOSED by WP-D** (bitacora 09).
6. ~~Notebook 04 monotonicity error + missing probe/framing~~ **CLOSED by
   WP-C** (bitacora 08): corrected at the builder; constants artefact-backed.
7. ~~LR-side diagnostics under transfer~~ **CLOSED by WP-N** (bitacora 19):
   both transform variants reported for both heads; the naive-vs-recal
   contrast is the R8 measurement and it is clean.
8. **Raw TwiBot-20 acquisition** (opened by WP-N): the one gate between the
   programme and its primary claim. Contact per the quote in
   `results/p6n_transfer.json` → `scoping.raw_dataset_availability`; then
   extend `scripts/run_p6n_transfer.py` to every family per the protocol
   paragraph kept in HANDOFF above. Alternatively amend the target corpus,
   pre-registered before any such data is analysed (plan §9.3).

## Performance note

Pin `OMP_NUM_THREADS=1`. HGB took 115.6 s vs 0.87 s for the same 5 folds
otherwise — 133x, pure scheduling overhead. Now pinned inside the scripts.
