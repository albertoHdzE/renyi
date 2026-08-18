# Bitácora 00 — Kick-off

**Date:** 2026-08-18
**Branch:** `main`
**Status:** design only. **No experiment has been run.** No package exists yet.

---

## 1. Why this package exists

`01-info-propagation/` replicated three papers on information propagation, disinformation
and bot detection. The replications succeeded, and in succeeding they produced four
measurements that make the *original* research questions unaskable on that ground:

| Measurement | Source | Consequence |
|---|---|---|
| 5 metadata numbers + MLP = **0.9775**; full 896-d GraphSAGE+BERT = **0.9779** | `results_ablation.csv` | no headroom; the text and graph branches are decorative |
| 5-fold CV **0.9779** vs official split **0.8860** | `DISCREPANCIES_BOTSAGE.md` §5 | protocol dominates method by 9.2 points |
| split membership predictable from the 5 features at **AUC 0.79** | ibid. | the features fail under mild distribution shift |
| α ∈ [0.2, 5] spans **0.0098** AUC against seed σ **~0.02** | `alpha_sweep.json`, `significance_auc.json` | α was inside noise on that substrate |

Row 1 says the benchmark is saturated. Rows 2 and 3 say the saturation is fragile — the
same features lose 9.2 points under a shift toward quieter accounts, within one corpus.
That is the gap this package targets: **not higher accuracy, but features that survive
the shift.**

Row 4 is the cautionary one. It is the reason this package pre-registers thresholds
before running anything.

## 2. The hypothesis, and why it is not the one we started with

The proximate framing was "use algorithmic complexity and Rényi α to detect
AI-generated text". Two corrections were made before kick-off, both recorded here
because the reasoning matters more than the conclusion.

**2.1 The target is bots, not machine text.** None of these corpora (2015–2022) carries
ground truth for LLM-generated text. Cresci-2015's "bot" is a purchased fake-follower
account — templated, metadata-anomalous, and detectable from five numbers. That is a
different signal from "was this token sequence sampled from a language model", and no
experiment on this data could distinguish them. Non-goal 1 in the charter; risk R11.

**2.2 The object is the spectrum, not a tuned α.** Selecting the α that maximises class
separation is one-dimensional feature selection with a multiple-comparisons hazard, and
row 4 above shows what that search finds when there is nothing there. The vector
`[H₀, H_½, H₁, H₂, H₄, H_∞]` is strictly more informative, needs no tuning, and has
standing: `τ(q) = (q−1)H_q` is the mass-exponent function whose Legendre transform is
the multifractal spectrum. This is multifractal analysis of behaviour, not a parameter
sweep.

Two endpoints recover known measures — `H₂` on word frequency is Yule's K (1944); `H₀`
is log type count. The contribution is the joint use and the fronts, not any single α,
and the write-up must say so.

## 3. What changed on the BDM side

The initial assessment argued that BDM cannot help because (a) Kolmogorov complexity is
generator-agnostic by the invariance theorem, and (b) BDM converges to block Shannon
entropy at scale. Both were argued against **BDM 1.0**, and (b) is confirmed by Zenil's
own group as the motivation for the successor (Sakabe, Abrahão, Hernández-Orozco, Gudwin
& Zenil, arXiv:2606.23471, 22 Jun 2026, §2.4).

**BDM 2.0 answers (b) directly.** It replaces independent per-block descriptions with
conditional ones, so the aggregation is no longer pure multiplicity counting; the reuse
gain is bounded by **algorithmic mutual information** between blocks (their Theorem 2),
with `BDM2 ≤ BDM1 + O(1)` (Theorem 1).

**It also answers (a), for the right framing.** `K(x)` as a per-object feature is
generator-agnostic and cannot identify a generator. `I(x:y)` is not — it measures
*shared* generator, and the paper is explicit that BDM 2.0 measures "the shared **causal
generators** responsible for structural regularities". For detecting a fleet of accounts
driven by one script, that is the correct object. The original objection was aimed at
the wrong quantity.

**Consequence for design.** The AIT branch measures **coordination across accounts**, not
complexity of a single account. NCD is the version available today; BDM 2.0's reuse gain
is the principled version; Cresci et al.'s *Social Fingerprinting* (IEEE TDSC 2017)
detected spambot groups on this very corpus using longest-common-substring, which is a
crude proxy for the same quantity. The lineage is clean.

## 4. Decisive practical finding at kick-off

`botsage/text.py` documents that Cresci-2015 carries no tweet timestamps, and the node
schema agrees: tweet nodes hold `id` and `text` only. This would block the temporal
front — the strongest hypothesis — on the primary dataset.

It is wrong. Twitter snowflake IDs encode milliseconds in the top 41 bits.

```
timestamp_ms = (int(tweet_id) >> 22) + 1288834974657
```

**Measured, on a 200,000-tweet uniform random sample (seed 0):** 200,000 decoded,
0 failures, 0 pre-2010 artefacts. Range 2010-11-04 → 2013-06-03; median 2012-11-28 —
the Cresci-2015 collection window. Corpus totals: 5,301 users, 2,827,757 tweets.

**Datasaurus status: NOT YET GATED.** These are counts and a range, which is exactly what
G2 says is not agreement. The claim "the decoded timestamps are the real posting times"
is *unverified* — a monotone function of the ID would produce an equally plausible range.
P0 must pass G1 (render the full inter-arrival distribution and the per-account
trajectories, not just the range) and G4 (what would this look like if IDs were assigned
by a counter rather than a clock?) before any downstream number is quoted. Recorded as
the first open datasaurus item; see [../docs/04-DECISIONS.md](../docs/04-DECISIONS.md) D1.

## 5. What was decided, and what was frozen

**Frozen before any experiment:** four hypotheses with directional predictions and
thresholds derived from this repository's own measured noise (`docs/00-CHARTER.md` §3);
three protocols that are never mixed; six mandatory floors; ten phases with go/no-go
gates.

**Thresholds and where they come from.** H1 requires > 0.02 AUC over Shannon-alone
because 0.02 is the measured seed-to-seed σ. H4 requires a degradation gap > 0.05.
Effect sizes below 0.02 are not claimed regardless of p.

**Eight decisions** recorded as D1–D8: snowflake reconstruction; base-2 logarithms
(because CTM values are in bits and mixing bases is a silent unit error); fixed-n
subsampling rather than bias correction; raw uncleaned text for the text front;
4-symbol behavioural alphabet (forced — `acss.data` ships alphabets 2, 4, 5, 6, 9 and
there is **no alphabet-3 table**); BDM 2.0 in tiers with conditional CTM off the critical
path; fixed classifier; source-only standardisation for the transfer protocol.

**Adopted from sibling programmes** (`docs/06-STANDING-RULES.md`): the datasaurus gate,
ported as a skill; the bitácora; didactic executable notebooks; marginal-preserving
nulls; and positive controls on synthetic generators.

## 6. The risk that will decide this project

**R1 — activity volume masquerading as distributional shape.** Plug-in Rényi estimators
are size-biased, worst at small α. Bots and humans post at different rates. An
uncorrected spectrum encodes posting volume, which is *already* a metadata feature — so
the "new" family would be a noisy re-encoding of the incumbent, **and it would look like
it works.**

Three independent lines arrived at the same control: property P8 here, G4's "check
effective N", and deconv-lab rule 13 ("check every per-unit feature against sample size
before calling it a property of the unit"). Gate G1 is a hard stop on
|ρ(H_α, event count)| < 0.1, and S4.2's positive control makes it concrete — a periodic
and a Poisson account with **identical event counts** must be separated by the spectrum,
or the spectrum is measuring volume.

## 7. Open items entering P0

1. **Datasaurus gate on D1** (§4). Blocking for everything downstream.
2. Post-type classification into the 4-symbol alphabet is not yet specified — Cresci's
   conversion has no explicit type field, so it must be inferred from text and edge
   structure. Method to be decided and recorded in P0.
3. `n_events = 128` is a guess. G0 may force it down; whatever it becomes is reported
   with every subsequent result, along with per-class exclusion counts.
4. `pybdm` is v0.1.0, "Pre-Alpha", last released 2019. Cross-validation against
   `acss.data` is required before any AIT number (R10).

## 8. Next action

**P0 — data layer.** Decode all 2,827,757 IDs; build per-account event series; pass G0
and the datasaurus gate on D1; render before quoting.
