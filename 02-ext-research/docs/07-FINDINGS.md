# 07 — Findings of `02-ext-research`

The `DISCREPANCIES`-analogue for this project: one split-interpretation section
per finding, newest last. Maintained per plan WP-M. Every number carries its
`results/*.json` source (see `../EVIDENCE-INDEX.md`). Append-only.

---

## F1 — The corpus has timestamps after all, and half of them were lying

**Finding (P0).** Tweet IDs are snowflakes; `(id >> 22) + epoch` recovers
posting time (decision D1). The first render then caught that pre-snowflake
sequential ids decode to a few ms past the epoch — **63,830 tweets on one
millisecond** — fixed by thresholding at the first snowflake id, not a bit width
(decision D9). The elementwise created_at check passed at 0 violations *before
and after* the fix; only the render saw it.

**Status:** closed. Source: `bitacora/01`, `results/p0_events.json`.

---

## F2 — Volume is the dominant signal in Cresci-2015, and the fixed-n design died of it

**Finding (P0).** Bots post a median of 23 tweets vs humans' 834; event count
alone scores AUC 0.939; D3's fixed-n subsampling retains ≥ 80 % of both classes
only at n ≤ 12. G0 failed; H1 was amended (bitacora 02) to test shape against
volume directly.

**Status:** closed as a gate failure that reshaped the design.
Source: `bitacora/01/02`, `results/p0_events.json`.

---

## F3 — H1 passed its clauses; the mechanism did not survive its own render

**Finding (P2).** SPEC_T beat count (+0.0367) and Shannon (+0.0380), 10/10
seeds, p = 0.0020 — but failed the burstiness floor (+0.019 < 0.02) and the
α-curves are near-parallel offsets: tail-resolution is not demonstrated.

**Status:** verdict stands; **reading superseded by F4.**
Source: `results/p2_temporal.json`, `bitacora/04`.

---

## F4 — Censoring alone can produce ~all of it (WP-A probe)

**Finding (WP-A, 2026-08-24).** Same generator, same rate, shorter observation
window: AUC(B vs A) = **0.9224–1.0000** across all nine generator × window
cells through this exact pipeline (trigger ≥ 0.85, fired at 1.0000). Any claim
that SPEC_T's edge over count measures *behavioural shape* is bounded by a
censoring artefact family whose ceiling here is ≈ 1.

**What this does NOT show:** that P2's +0.0367 *is* censoring. Real accounts
fade gradually rather than cutting off; the equal-window arms (plan WP-F) test
the real corpus directly.

**Status:** open → WP-F adjudicates; every SPEC-family number carries the
ceiling annotation until then. Sources: `results/p2c_probe.json`,
`bitacora/05`, amendment `bitacora/06`.
**Update (2026-08-24, WP-F):** adjudicated — see F8. The ceiling survives as
a bound; one clause's edge was inside it, one was not.

---

## F5 — TwiBot-20 inverts the volume story, and the scale trap is empty

**Finding (WP-B, 2026-08-24).** On TwiBot-20's labelled population (bot-majority
0.5572 — unlike every other corpus here), profile magnitude is *weak*: statuses
alone AUC 0.6073, best scalar followers 0.7414 — versus Cresci-2015 where count
alone scores 0.939. The pre-registered branch therefore did **not** fire: H4
runs as chartered, no incremental-over-volume amendment.

Two secondary facts worth keeping:

1. **The labelled slice is unrepresentative of its own corpus** — z-scoring was
   verified at corpus level (means 0, sds 1), but the annotated 11,826 users
   drift hard (subset sd up to 3.76). Transfer conclusions apply to the
   annotated population, which is the one anyone would deploy on.
2. **The feared standardisation trap is empty for trees**: strict source-scaler
   transfer collapses target columns to sd ≈ 10⁻³ yet META's AUC moves 0.0005
   (0.7864 vs 0.7859 recalibrated) — HGB rescaling invariance. The trap is real
   only for scale-sensitive heads; WP-N keeps LR diagnostics on recalibrated
   features.

**Status:** closed for framing purposes; feeds WP-N's caveat set.
Sources: `results/p6b_tb20_preflight.json`, `bitacora/07`.

---

## F6 — The estimator was silently losing mass; the check caught the fix's own leak

**Finding (WP-D, 2026-08-24).** `log_bin_counts` dropped all intervals outside
`[lo, hi]` — 73 bot vs 17 human intervals at hi = 400 d (0.0393 % / 0.0007 %).
The first fix (overflow cell only) still leaked below `lo`; property P16's
first draft failed on it and forced the symmetric underflow cell. Full P2
re-run: every verdict unchanged (clauses +0.0364/+0.0381 clear at p = 0.002;
burstiness floor +0.0170 still fails; max arm shift 0.0003). Probe re-run:
trigger still fired at 1.0000.

**Status:** closed. Sources: `results/p2_temporal.json` (post-fix),
`results/p2b_decomposition.json`, `figures/p2d_overflow_mass.png`,
`bitacora/09`.

---

## F7 — Dimensionality matching: clause (ii) survives it; the burstiness verdict is subfloor under it

**Finding (WP-E, 2026-08-24).** The pre-registered D2 noise-padded controls
(plan §8, failure semantics in WP-E task 3 [rev1]) attach the equal-dimension
reading to P2's floor verdicts (`results/p2_temporal.json`, block
`dim_matched`):

| matched row | Δ | wins / p | σ_cfg | verdict |
|---|---|---|---|---|
| SPEC_T vs SHAN+NOISE(10) — clause (ii) reading | **+0.0376** | 10/10, 0.0020 | 0.0094 | `supports_clause` |
| COUNT+SPEC_T vs COUNT+BURST+NOISE(9) — burstiness verdict | **+0.0159** | 10/10, 0.0020 | 0.0006 | `real_but_subfloor_not_claimable` at the registered 0.02 floor |

Three things the objects add beyond the labels:

1. **Clause (ii)'s Shannon-floor edge is not a dimensionality artefact.**
   At equal dimension the edge is +0.0376, within noise of the unmatched
   +0.0381.
2. **The padded floors were not handicapped**, so the rows are interpretable
   as equal-dimension readings rather than crippled baselines: SHAN+NOISE(10)
   scores 0.9325 vs SHAN's own 0.9320, and COUNT+BURST+NOISE(9) scores
   0.9605 — slightly *above* the real COUNT+BURST at 0.9594. HistGradient-
   Boosting simply ignores pure-noise columns at this n.
3. **The burstiness verdict was already failing as gated** (+0.0170 < 0.02);
   under matching it stays subfloor (+0.0159). Recorded as *not claimable at
   the registered floor* per the second interpretation rule — nothing was
   downgraded because it was never claimed.

No comparison landed at Δ ≤ 0: the first rule's downgrade semantics did **not**
fire, and no hypothesis clause support changed. The registered gate verdicts
stand exactly as gated either way (plan WP-E task 3).

**Status:** closed — recording executed in HANDOFF and here per plan WP-E
task 3. Sources: `results/p2_temporal.json` (`dim_matched`, `arms`),
`bitacora/10`.

---

## F8 — Equal windows split H1: clause (i) survives everything, clause (ii) was the censoring artefact

**Finding (WP-F, 2026-08-24).** Truncating every Cresci-2015 account to the
first K days of its own activity (`results/p2c_truncation.json`) separates
the two clauses of H1-as-amended that P2 had passed together:

| comparison (headline K = 30 d) | unwindowed | equal-window | dim-matched | verdict |
|---|---|---|---|---|
| clause (i) CS − COUNT | +0.0364 | **+0.0423** (10/10, 0.002) | +0.0488 vs C+N12 | **survives** (grows with K: +0.0681 at 90 d) |
| clause (ii) SPEC_T − SHAN | +0.0381 | **+0.0048** (10/10, 0.002) | **−0.0003** vs SHAN+N10 | **`confounded_dimensionality` — DOWNGRADED** |
| burstiness CS − CB | +0.0170 | +0.0120 (fails) | +0.0066 vs CB+N9 | not claimable (consistent with gated failure) |
| SHAPE vs BURST+N9 | — | **+0.0234** (10/10, 0.002) | supports_clause | **new survivor** |

**The executed downgrade (plan WP-E task 3 rule 1).** Clause (ii)'s support
is withdrawn as confounded: the Shannon-floor edge does not survive equal
observation windows at any K (max +0.0084 at K = 90, floor 0.02) and is ≤ 0
dimension-matched at the headline. What P2 measured as clause (ii) was the
classes' difference in *retained timeline length*, not tail-resolution.
H1-as-amended required both clauses; its support is downgraded to
**clause-(i)-only**. Registered unwindowed verdicts stand as computed — no
floor moved, no history edited.

**What was not known before this control:** the level-removed spectrum
(SHAPE) beats its burstiness floor at matched dimensions under equal windows
at every K — the first result in the programme to clear every registered
control simultaneously. The count-anchored spectrum edge (clause i) also
survives, conditional on K ≥ 30 (σ_cfg across K ≈ 0.026, and K = 7 is a
tiny-n, human-majority regime).

**The API cap is innocent (review D2):** cap = 3215 events (mode of the human
upper tail; the spike region below it holds ~375 humans); excluding ≥ cap
(131 humans) moves COUNT −0.0047, excluding the whole ≥ 0.95·cap region
(383 humans, post-hoc) moves it −0.0133. The volume edge is bots' low counts,
not truncated humans.

**Status:** closed for P2's reading; clause-(i) and SHAPE edges carry forward
into WP-H/I/J/N with the equal-window caveat replaced by these measured
controls. Sources: `results/p2c_truncation.json`, `figures/p2f_equal_window.png`,
`figures/p2f_api_cap.png`, `bitacora/11`.

---

## F9 — The circadian sign reversal is a suppression effect, not an artefact

**Finding (WP-G, 2026-08-24).** Within count-caliper strata (D4, 8 valid),
bots' hour-of-day entropies exceed humans' by **+0.25…+0.30 bits on every
order** (7–8 of 8 strata positive; per-stratum TV 0.19–0.25), agreeing with
the conditioned positive coefficients — while the raw correlations are
negative. The reversal is suppression: volume spreads human posting hours,
and at matched count the class difference reverses, with bots carrying more
night mass (rendered per stratum). The timezone offset is **provably inert**
(effects ≤ 2.8e-16: a constant shift is a cyclic relabelling and every rule
statistic is permutation-invariant), so the adjudication rests on matching
alone. Branch: `kept_suppression_explained` — circadian orders stay in
SPEC_T without caveat; SPEC_B remains alphabet(6) + mention-targets(6).

**Status:** closed; HANDOFF open item 1 closed. Sources:
`results/p3a_circadian.json`, `figures/p3g_circadian_*.png`,
`bitacora/12`.

---

## F10 — The behavioural front: real mechanism, no information beyond metadata (on this corpus)

**Finding (WP-H, 2026-08-24).** SPEC_B splits three ways
(`results/p3h_behaviour.json`):

1. **The H2 mechanism is real** (gate PASS): bots are near-deterministic in
   post type (median H₂ 0.0000 vs human 1.1969) and mention one target where
   humans mention seven (H₀ 1 vs 7.24) — both one-sided p ≈ 0, signs as
   predicted.
2. **It beats volume everywhere**: SPEC_B_ALPHA vs COUNT +0.0211, SPEC_B_
   MENTION vs COUNT +0.0414, COUNT+SPEC_B vs COUNT +0.0539 (all 10/10,
   p = 0.002).
3. **It carries no demonstrable information beyond META-lite on Cresci-2015**:
   META alone scores 0.9972/0.9946, and every family-vs-META dim-matched row
   fires `confounded_dimensionality` (Δ −0.020…−0.040, 0/10) — **downgrade
   executed**: the behavioural front is not claimable as beyond-incumbent
   here. Against its own Shannon slices it is real-but-subfloor
   (+0.003…+0.009 < 0.02): the 6-order spectrum adds little beyond H₁
   diversity on this front.

Two structural caveats travel with every mention-block number: the mention
block excludes 1,490 bots (44.4 %) vs 11 humans (0.6 %) with no valid
targets (the sample flips human-majority), and META's near-ceiling score is
corpus-construction-linked (fake-follower bots) — precisely why H4's
transfer test, where WP-B measured TB20's META at only 0.79, is where the
incumbent question is settled.

**Status:** open → WP-N adjudicates the transfer side; the within-corpus
verdict (confounded vs META) stands as recorded. Sources:
`results/p3h_behaviour.json`, `figures/p3h_spec_b_*.png`, `bitacora/13`.

---

## F11 — The text front: the spectrum order-set finally beats its Shannon slice (exploratory, Holm-surviving)

**Finding (WP-I, 2026-08-24).** SPEC_X on raw uncleaned text
(`results/p3i_textfront.json`, all numbers exploratory): **SPEC_X_CHAR
0.9889**, SPEC_X (12) **0.9940** — and, for the first time in the programme,
the full order-set beats its own Shannon slice **at matched dimensions**
(char +0.3332, combined +0.0537, word +0.0466, COUNT+SPEC_X +0.0318 — all
`supports_clause`, all 20 family comparisons **survive Holm**, adjusted
p = 0.0391). The char block's signal lives in the tail orders (H_4/H_inf)
that H₁ throws away — the mechanism the α-grid was designed to find,
finally measured.

Also: beats both volume covariates (vs COUNT +0.0540, vs TOKENS +0.0446);
the edge *grows* on the ≥ 512-token subsample (+0.0636, 631 bots / 1,817
humans survive); URL stripping moves it −0.0077 (27 % of tweets carry URLs).
Length control shows word diversity is volume-confounded raw
(H_0_word −0.801 → +0.042 given tokens) while character usage is not
(+0.351 given tokens) — the two blocks are complementary, not redundant.

**But the incumbent ceiling holds on this corpus**: SPEC_X_WORD vs
META+NOISE(2) −0.0233 and SPEC_X vs META+NOISE(8) −0.0036 both fire
`confounded_dimensionality` (downgrade executed) — SPEC_X approaches META
(0.9940 vs 0.9975) without passing it. The within-corpus ranking META >
SPEC_X ≈ META−ε stands; whether it transfers is WP-N's question, where
TB20's META is a different beast (0.79).

**Status:** open → WP-N adjudicates transfer; within-corpus verdicts stand
as recorded. Sources: `results/p3i_textfront.json`,
`figures/p3i_spec_x_objects.png`, `figures/p3i_length.png`, `bitacora/14`.
**Update (2026-08-24, correction `bitacora/15`):** the headline framing
above buried the bounding comparison — given TOKENS as its own feature, the
spectrum's residual over its Shannon slice is **+0.0166, subfloor**
(`real_but_subfloor_not_claimable`): the +0.3332 is largely length-mediated.
Also recorded there: every raw p in the family is the 10-seed Wilcoxon floor
(0.001953), so "all 20 survive Holm" ≡ "all 10/10 wins and family ≤ 25" —
not independent evidence; and the "tail orders carry it" reading is
render-suggested, not ablated.

---

## F12 — The mechanism is tail-magnitude, carried by survival proportions — not the Hill index, not multifractal shape

**Finding (WP-J, 2026-08-24).** The pre-registered verdict fired
(`results/p2c_truncation.json`, `mechanism_verdict`): TAIL+SURV's
dim-matched margin over the burstiness floor beats SPEC_T's at **every**
K (+0.0689/+0.0440/**+0.0555**/+0.0546 vs +0.0634/+0.0314/**+0.0361**/
+0.0470 at K = 7/14/30/90; all `supports_clause`). Per plan WP-J task 4,
the mechanism narrative updates: **separability is tail-magnitude, not
multifractal shape.**

The honest decomposition the plan's verdict requires: **SURV carries it** —
three empirical survival proportions at the fixed lags {1 h, 1 d, 7 d}
score 0.9260 alone at K = 30, equal to TAIL+SURV's 0.9261, while **TAIL
alone is worse than volume** (vs COUNT −0.1208: the fitted Hill index
tracks posting rate in reverse). So the referee-facing statistic is the
simplest one on the table: P(gap > 1 h/1 d/7 d). The probe now bounds this
reading too (TAIL+SURV cells 0.549–0.990 across the nine censoring-null
cells, below the COUNT+SPEC_T ceiling at short windows).

**Status:** open → the tail-magnitude narrative carries into WP-N's
transfer matrix (SURV joins the family list); the probe ceiling annotation
applies as everywhere else. Sources: `results/p2c_truncation.json`
(`mechanism_verdict`, windows), `results/p2c_probe.json`, `bitacora/16`.

---

## F13 — The era shift degrades nothing SPEC; the sides are not in a common coordinate

**Finding (WP-K, 2026-08-24).** Train on pre-2012-07-01 accounts (n = 2,763,
bot 0.5074), transfer to newer accounts (n = 2,007, **bot 0.7195**):
`results/p3k_timesplit.json`. Three facts, in order of importance:

1. **No SPEC family crosses the preview guard** (Δ_META − Δ_family > 0.05):
   max gap +0.0385 (SPEC_B_ALPHA); SPEC_T +0.0184; SPEC_X +0.0018. The
   feared pattern — spectrum degrading faster than metadata under shift —
   does not occur within this corpus.
2. **The pairs that do fire are inverted**: META vs TAIL (+0.1276) and META
   vs BURST (+0.0589) fire because those floor statistics *improve* on the
   test era (negative Δ), not because META collapses — META transfers at
   ceiling (+0.0005, CI [0.9912, 0.9989]); COUNT is the only volume-family
   degradation (+0.0217).
3. **Structural caveat that leads every reading**: the split sides differ
   in class composition (bot 0.5074 → 0.7195), so Δ mixes era shift with
   rebalancing — AUC is not invariant to that. 56.37 % of accounts span the
   boundary. The G4 null (random halves, max |Δ| = 0.0158) calibrates
   sampling noise only. Negative deltas read as "shift + composition";
   near-zero deltas (META ±0.001, SPEC_X −0.0013) are the genuinely flat
   ones.

R8 age-guard: immaterial (META_no_age −0.0010). Boundary sensitivity strong:
at 2012-01-01 train is bot-0.2597; at 2013-01-01 test is bot-0.8781 — the
registered date sits mid-gradient.

**Status:** closed for the within-corpus axis — no preview evidence of SPEC
fragility; WP-N remains where the transfer question is actually decided,
with the composition caveat carried into its design (Protocol C's target is
a different corpus, where balance effects must be handled explicitly).
Sources: `results/p3k_timesplit.json`, `figures/p3k_era_split.png`,
`figures/p3k_degradation.png`, `bitacora/17`.

---

## F14 — H3 is untestable on Cresci-2015: the kill criterion fires (executed cancellation)

**Finding (WP-L, 2026-08-24).** The coordination gate H3 is **cancelled**
on this corpus, by the pre-registered branch-(ii) kill criterion, not by
scoping language (`results/p4l_ait.json`):

- Provenance: groups NOT recoverable (label.csv = `id,label` only; flat
  ids; the source archive unpacks to the same five flat files) →
  cluster-enrichment estimand.
- D7 viability: all nine (compressor, L) NCD cells monotone-strict — the
  NCD machinery is sound; L = 23 is the documented floor length.
- **Kill criterion (≥ 100 events): 162 bots (4.83 %) survive** vs the
  required max(200, 335) — the high-volume restriction leaves an 8.8 % bot
  / 91.2 % human population. The G1 length render is the evidence: bot
  mass sits below the threshold; above it the population is essentially
  human (plus the ~3200 cap spike).

Two tooling findings travel with the cancellation: (a) PyPI `acss` 0.5.2 is
a Python-2 name collision, not the CTM-tables library — the R10
cross-validation is BLOCKED, recorded rather than substituted (pybdm 0.1.0
pinned; its CTM coverage measured: 1D alphabets {2,4,5,6,9} × 12-blocks, 2D
binary 4×4 only — no 2D alphabet-4 table); (b) the plan's P9 shorthand
("BDM ≪ its block entropy") does not hold under the standard definition —
P9 asserts and passes the R6 intent (periodic BDM = 0.122× random) and the
divergence is recorded.

**Status:** closed — H3 leg cancelled with the survivor table published;
coordination testing flagged for a higher-volume corpus (candidate:
TwiBot-20 timelines, out of scope). WP-N proceeds WITHOUT an AIT family.
Sources: `results/p4l_ait.json`, `figures/p4l_dna_render.png`,
`figures/p4l_dna_lengths.png`, `bitacora/18`.

## F15 — H4 is untestable on the obtainable TwiBot-20: the metadata side measured, the ours side has no data (scoped execution)

**Finding (WP-N, 2026-08-25).** The primary claim could not be sat with
the artefacts that exist. This is a data finding, not a negative result
about the features — and it is itself quantified:

- **The modality wall** (`results/p6n_transfer.json`, `scoping` block;
  corrected phrasing per bitacora 20): four doors checked, none bears the
  modalities — (1) the BotRGCN HF mirror (9 files, all tensors/json; the
  tweet file is dense pooled 768-d embeddings, non-zero row fraction
  1.000000, row norms 9.3–13.6); (2) raw TwiBot-20 on GitHub, gated by
  its authors; (3) TwiBot-22's open Zenodo `user.json`, profiles only;
  (4) the TwiBot-22-format conversion archive, seven corpora and no
  twibot-20 (bitacora 21 lists them). Every family our programme built
  (SPEC_T/SHAPE/BURST/SHAN/TAIL/SURV/SPEC_B_ALPHA/SPEC_X) needs a
  modality no door ships.
- **What was measured instead**: the metadata side of H4's inequality.
  Cresci's near-ceiling META degrades by **Δ +0.3143 ± 0.0075**
  (within 0.9974 → transfer AUC 0.6831, CI [0.6764, 0.6938]) under the
  charter-faithful transform, and +0.3335 under marginal
  recalibration — R8 verdict: effect under both variants, not an
  alignment artefact. **But read the AUC with its calibration: the
  transferred model puts nearly every target user at predict_proba ≈ 1**
  (accuracy 0.5585 vs majority 0.5572; macro-F1 0.3612; TPR@1%FPR
  0.0137) — total calibration collapse with residual ranking only, not
  decay to a mediocre-but-working classifier. Volume's ordering against
  META is variant-dependent (bitacora 20): claimed where it survives the
  R8 rule — LR, both variants, VOL−META +0.0434/+0.0291 in 20/20 draws;
  the hgb-naive volume numbers (+0.3767, dim-matched +0.3957) are
  transform pathology (collapsed column sd breaks tree binning) and the
  hgb ordering reverses under recalibration (−0.0124, 3/20 draws).
  G4 sanity null silent at 0.0088. Composition shift −7.50 pp bot share
  carried as caveat.
- **Why this sharpens rather than settles**: WP-K showed META transfers
  within Cresci at Δ +0.0005; across corpora it collapses by 0.31. Had
  any of our families been computable target-side, the > 0.05 bar would
  have been cleared by Δ_META alone. The exam remains unsat until the
  data exists.

**Charter success criteria revisited** (the plan-completion statement):
H1 supported on clause (i) only after equal-window controls (F6/F7);
H2 directional gates passed, vs-META confounded locally (F10); H3
cancelled by its pre-registered kill rule before any result existed
(F14); **H4 untestable-as-chartered pending gated raw TwiBot-20 or an
amended target** (this finding) — with its metadata side now a measured
number rather than an assumption (bitacora 07 §4 anticipated exactly
this reading). Per charter §3 H0, a programme ending in precise negatives
and one open gate is a publishable outcome; P7/P8 remain governed by
docs/03-PHASES.md.

Sources: `results/p6n_transfer.json`, `figures/p6n_degradation.png`,
`figures/p6n_alignment.png`, `figures/p6n_score_distributions.png`,
`bitacora/19`; corrected in `bitacora/20`.

## F16 — H4-T: the text front fails the transfer exam — it degrades MORE than metadata, and its fingerprint inverts

**Finding (amendment H4-T, 2026-08-25; PRELIMINARY pending the authorized
copy).** On the only modality-bearing target obtainable (TwiBot-20 raw
labelled release, text-bearing/timestamp-free — bitacora 22), the
registered amendment sat the exam the only way it can be sat:

- **Primary verdict: FAILED.** Δ_META − Δ_SPEC_X = **−0.10 / −0.156 /
  −0.179** for WORD/CHAR/X (hgb, paired per-draw, **0/20 draws** above the
  charter's +0.05 bar). The text spectra degrade more than metadata:
  SPEC_X_CHAR falls from 0.9898 within to **0.4840 transferred (CI95
  [0.4719, 0.4925]) — below chance**, i.e. the source-fitted
  character-usage structure anti-ranks the target's classes.
- **Mechanism**: char-spectrum orders are near-chance measured on TwiBot-20
  alone (H₁ 0.511, H∞ 0.536) — the locally near-perfect fingerprint is
  simply absent there, and the multivariate residue flips sign. Named
  suspect for follow-up: corpus-language shift between Cresci-2015's human
  class and TwiBot-20's English-dominant population (hypothesis, not
  conclusion).
- **Robustness of the negative**: equal-cap window control moves deltas
  ≤ 0.0126 (σ_config); G4 null silent at 0.0134; all six Holm-corrected
  floor comparisons read `confounded_dimensionality`; LR agrees; R8 at raw
  scale confirms the recal-moot prediction (AUC equal, accuracy repaired).

**Where this leaves the transfer picture**: WP-K — shape statistics are
era-robust *within* a corpus (nothing degrades 2011→2013); WP-N-scoped —
metadata collapses cross-corpus (+0.3143 measured); H4-T — the text front
is *language/lineage-bound* across corpora and is LESS transferable than
metadata on this pair. The honest programme-level statement: no family we
built sits H4 as chartered on the corpora obtainable; the claim stays
closed pending the authorized copy, and any future target must carry both
timestamps AND a language profile matched to the question being asked.

Sources: `results/p6n_transfer_text.json`,
`figures/p6nt_degradation.png`, `figures/p6nt_length_distributions.png`,
`figures/p6nt_score_distributions.png`, `bitacora/23` (registration §1–6
committed before execution; results §7).

## F17 — LS-1: the language shift is real and huge — and innocent of the inversion

**Finding (amendment LS-1, 2026-08-25; PRELIMINARY pending the authorized
copy).** The corpus-language hypothesis F16 named as suspect was
adjudicated as pre-registered (bitacora 24, registration committed before
any language computation):

- **The shift itself: confirmed decisively (P1 PASS).** Cresci-2015 is
  language-separated by class — bots 54.18 % English vs humans **5.67 %
  English (93.1 % Italian)** — while TwiBot-20 is English-dominant in
  both classes (bots 93.08 %, humans 75.56 %). Predicates +0.4852 /
  +0.6990 against a 0.20 bar.
- **The mediation: REFUTED (primary FAIL).** Re-running the SPEC_X_CHAR
  transfer on EN∩EN-restricted populations leaves the inversion intact:
  Δ_INV = **−0.0028** (0/20 draws; restricted transfer 0.4867, CI95 low
  0.4754 — still below chance). Budget and window sensitivities inert
  (0.0000 / 0.0270); the EN-source→FULL-target placebo fails too
  (0.4800). Within-Twibot-20 local char separability **worsens** under
  language matching (0.6381 → 0.5969, 0/10 seeds, holm p 0.0059).
- **Combined verdict: PARTIAL-EFFECT** — the pre-committed label for
  "shift real, mediation absent".
- **Carried caveat:** the EN∩EN source holds only 109 humans (of 1,651);
  a repair needing many English source humans was never testable with
  power. The placebo and local-restoration arms corroborate the negative
  without sharing that limitation.

**Where this leaves the transfer picture**: the char fingerprint's
failure is **corpus-lineage-bound, not language-tag-bound** — the
transferred boundary is bimodal-confident on both classes and language
matching moves nothing. F16's "language profile matched to the question"
recommendation is sharpened: language matching is *insufficient*; the
variable no obtainable corpus supplies is collection lineage. Any future
transfer claim leaning on text features must sit a same-lineage exam.

Sources: `results/p6n_ls1.json`, `figures/p6nls_language_census.png`,
`figures/p6nls_tweet_share.png`, `figures/p6nls_inversion.png`,
`bitacora/24` (registration §1–7 committed before execution; results §8).
