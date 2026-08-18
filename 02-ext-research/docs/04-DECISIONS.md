# 04 — Decisions log

The `DISCREPANCIES.md` analogue for this project. Every decision a reader could not
infer from the code, with what was decided and why. Append as phases complete; never
silently revise.

Format: **what the situation is → why it is underdetermined → what was chosen → how it
was verified.** Items marked **blocking** materially change results.

---

## Decisions taken at kick-off (2026-08-18), before any experiment

### D1 — Cresci-2015 timestamps are reconstructed from snowflake IDs — *blocking*

**Situation.** `botsage/text.py` documents that "the Cresci-15 release carries no tweet
timestamps anyway". The node schema confirms it: tweet nodes carry `id` and `text` only.
This would block the temporal front (H1, the strongest hypothesis) on the primary
dataset.

**Why it is underdetermined.** It is not — it is simply wrong. Twitter snowflake IDs
(post-Nov-2010) encode milliseconds since the Twitter epoch in the top 41 bits.

**Choice.** `timestamp_ms = (int(tweet_id) >> 22) + 1288834974657`.

**Verified.** 200,000-tweet random sample: 200,000 decoded, 0 failures, 0 pre-2010
artefacts. Range 2010-11-04 → 2013-06-03, median 2012-11-28 — the Cresci-2015
collection window. Corpus: 5,301 users, 2,827,757 tweets, ~533 per user.

**Consequence.** `botsage/text.py`'s docstring is inaccurate on this point. Per the
non-goal on freezing prior work ([00-CHARTER.md](00-CHARTER.md) §4), it is **not**
edited; the correction lives here.

---

### D2 — Logarithms are base 2, not nats

**Situation.** `dtwre/entropy.py` uses natural logs and says the base "only rescales
every entropy by a constant factor" — true in isolation.

**Why it matters here.** §5 of [01-METHODS.md](01-METHODS.md) mixes entropies with CTM
and BDM values, which are in **bits** by construction (`CTM(s) ≈ −log₂ D(s)`). Mixing
bases in a single feature vector is a silent unit error.

**Choice.** Base 2 throughout this project. Stated in every estimator's docstring.

---

### D3 — Fixed-n subsampling, not bias-corrected estimators — *blocking*

**Situation.** Plug-in Rényi estimators are biased, with the bias depending on both n
and α. `H₀` is worst: observed support grows with sample size. Bots and humans differ in
posting volume.

**Why it is underdetermined.** There is no standard bias correction for general α
comparable to Chao–Shen at α = 1, so a principled correction across the whole spectrum
is not available off the shelf.

**Choice.** Fix `n_events = 128` per account, `B = 100` bootstrap subsamples, mean
spectrum reported with bootstrap SD. Sensitivity at n ∈ {64, 256}. Because n is identical
for every account, the bias is a constant offset per α and cannot differ by class.

**Cost, stated plainly.** Accounts below the cutoff are excluded, and the exclusion is
itself a class-dependent bias. The exclusion count per class is reported with every
result. Efficiency is traded for validity, deliberately.

**Verified by.** Property P8: after subsampling, |ρ(H_α, total event count)| < 0.1 for
every α. This is gate G1 and is not negotiable.

---

### D4 — Front X uses raw, uncleaned text

**Situation.** `botsage/text.py` lowercases and strips URLs, mentions, hashtags,
non-ASCII and all non-alphabetic characters, per the source paper's Sect. 3.1.2.

**Why it matters.** That pipeline removes casing, punctuation and orthographic
irregularity — a large part of what a character- or word-frequency spectrum measures.
Running Front X on cleaned text would test a weakened hypothesis.

**Choice.** Front X operates on raw text. `botsage`'s cleaning is untouched and remains
in use for the `TEXT` reference block only.

---

### D5 — The behavioural alphabet has 4 symbols, not 3 or 5

**Situation.** Cresci et al.'s digital-DNA encoding is usually presented with a 3-symbol
alphabet.

**Why it is constrained.** The CTM lookup tables cover alphabets **2, 4, 5, 6 and 9**.
`acss.data` ships 4.5M strings of length 1–12 over exactly those alphabets; there is
**no alphabet-3 table**. `pybdm` supports 1D binary plus 1D with 4, 5, 6, 9 symbols.
A 3-symbol encoding would force either a lossy binarisation or a padded 4-symbol
encoding with an unused symbol distorting the block statistics.

**Choice.** 4 symbols: `{original, reply, retweet, quote}`. Temporal DNA uses 4 bins by
training-set quartiles.

---

### D6 — BDM 2.0 is built in tiers; conditional CTM is off the critical path

**Situation.** Sakabe et al. (arXiv:2606.23471) provide "a roadmap for implementation",
not an implementation. No conditional CTM tables have been published. Conditional CTM
requires re-running the machine enumeration with `x` on the input tape, once per `x`;
the (5,2) space is `(4·5+2)^10 ≈ 2.66 × 10¹³` machines. Exact reuse selection is proven
NP-hard.

**Why a choice is available.** The paper states (§2.4) that `K̂` "can be instantiated by
different approximations of Kolmogorov complexity and not necessarily by CTM", and Eq. 1
accepts observation-space conditionals `K̂(xⱼ|xᵢ)`.

**Choice.** Tier 1 (CTM tables for unconditional terms, compression for conditionals) and
Tier 2 (transformation library `T`, with cyclic shift load-bearing) in P7. Tier 3
(partial-enumeration conditional CTM) only in P8, only if P7 passes and conditional
estimation quality is shown to be the binding constraint. Greedy selection of `S`, with
the greedy/exact gap measured on tractable synthetic instances.

**Verified by.** P10 (`BDM2 ≤ BDM1 + C_rep`, their Theorem 1) and P11 (reuse detected on
cyclic-shift constructions).

---

### D7 — The classifier is fixed; no architecture search

**Situation.** The source paper's linear SVM was shown to be the binding constraint,
costing 4.2 points (`docs/DISCREPANCIES_BOTSAGE.md` §4).

**Choice.** `HistGradientBoostingClassifier` primary, L2 logistic regression secondary
for per-α coefficient interpretability. Held fixed across all feature families so that
the family is the only thing varying. Linear SVM is not used.

**Why it matters for validity.** With the classifier free to vary, a feature-family
comparison measures the interaction of family and architecture, not the family.

---

### D8 — Protocol C standardisation is fitted on the source corpus only

**Situation.** Protocol C fits on Cresci-2015 and tests on TwiBot-20.

**Choice.** Scalers fitted on Cresci-2015 only. Refitting on TwiBot-20 is target
leakage and would invalidate H4 — the hypothesis is precisely that the features transfer
*without* recalibration.

**Related.** `META` is reduced to the four fields overlapping between corpora; the
non-overlapping fifth is dropped from both sides, and the drop is reported. TwiBot-20's
properties arrive pre-z-scored, which the source corpus's do not — handled by fitting
the source scaler and accepting the resulting offset, which is reported.

---

## Decisions taken during P0 (2026-08-18)

### D9 — Pre-snowflake ids carry no timestamp and are discarded — *blocking*

**Situation.** D1 decodes `(id >> 22) + epoch`. Tweet ids issued before 2010-11-04 are
*sequential*, not snowflakes, and carry no time information — but they are large numbers
(~3 × 10¹⁰) and survive a naive `id ≥ 2²²` guard, decoding to a few milliseconds past the
epoch.

**How it was found.** The G1 render, not the G2 check. The corpus timeline showed ~64,000
tweets on day 0 and hour-of-day showed a peak at hour 1 — both at the epoch instant
01:42:54Z. **G2 passed at 0 violations both before and after the fix**, because a fake
2010 timestamp still postdates a 2007–2009 account creation. Recorded because it is the
clearest possible argument for rendering before quoting.

**Choice.** The threshold is the first snowflake id, `FIRST_SNOWFLAKE_ID =
29_700_859_247`, not a bit width. The corpus confirms the boundary: largest
sub-threshold id 29,700,661,919, smallest snowflake 292,906,606,796,800 — a gap of four
orders of magnitude with nothing in it.

**Cost.** 63,830 tweets (2.26%) discarded; corpus becomes 2,763,927 events. The exclusion
is class-dependent, so it is carried per account in `CresciEvents.dropped_per_user` and
reported with every result.

---

### D3 — amendment **PENDING**, not yet taken

D3's fixed-n subsampling is **infeasible on Cresci-2015**. The G0 retention criterion
(≥80% of both classes) is met only at n ≤ 12, and a Rényi spectrum on 12 events is
bias-dominated (`H₀ ≤ log₂ 12 = 3.58` bits). Raising n to where the spectrum is
meaningful (≥64) keeps 9.7% of bots, non-randomly.

Root cause: bots post a median of **23** tweets against humans' **834**, and **event
count alone scores AUC 0.939**.

Three options are set out in [../bitacora/01_p0_data_layer.md](../bitacora/01_p0_data_layer.md)
§4. This amends a pre-registered protocol and is therefore **not taken unilaterally**;
P1 is blocked on it.

---

## Findings

*(Appended as phases complete. Each entry: gate, pass/fail, measured number with error
bars and seed count, and anything that failed and was not fixed.)*

### P0 — gate G0: **FAIL**

| Item | Result |
|---|---|
| D1 snowflake decode | **CONFIRMED** — G1 rendered, G2 0 violations / 2,763,927 elementwise constraints, G4 circadian TV 0.2248 vs counter-null 0.0002 (peak/trough 12.82 vs 1.00) |
| P14 range in window | pass |
| P15 sorted | pass |
| G0 `n_events` retention | **FAIL** — 4.3% of bots retained at n=128; ≥80% both classes only at n ≤ 12 |

**Failed and not fixed:** the `n_events` criterion. It is a property of the corpus, not a
bug, and it blocks P1 pending the D3 amendment above.

**Not quoted further, pending controls:** inter-arrival distributions separate mainly in
the tail (Δt ≳ 1 day); bot rasters concentrate in days 600–900 while humans span the full
window. Both are confounded with volume, window length and account age — exactly the
failure mode named by deconv-lab rule 13.
