# 05 — From findings to a new protocol

How F1–F5 produced the design of `02-ext-research/`. This is the document to read if you
are going to build on any of this.

The short form: **the findings do not say "try harder on this benchmark". They say the
benchmark cannot answer the question, and they say precisely which question it can be
replaced with.**

---

## 1. What the findings jointly imply

Taken one at a time, F1–F5 read as criticisms of three papers. Taken together they
describe a single structural problem:

**(a) There is no headroom.** Cresci-2015 under 5-fold CV saturates at **0.9779** with a
fold σ of **0.0063**, and five metadata numbers with an MLP already reach **0.9775**
(F1). A new feature has ~0.002 of room against 0.006 of noise. Any result measured there
is unfalsifiable.

**(b) The features that carry the result are the brittle ones.** What actually does the
work is metadata *magnitude* — follower counts, tweet counts, account age. Follower
counts are purchasable and account age is era-specific.

**(c) And their brittleness is already measurable.** F5(a) is the symptom: the same five
features lose **9.2 points** under a mild within-corpus shift toward quieter accounts,
and split membership is predictable from them at AUC 0.79. The features do not survive a
distribution shift that does not even leave the corpus.

**(d) The evidential standard in the area is too weak to detect this.** F3 and F4: an α
effect of 0.022 reported without error bars against a measured σ of 0.0179; a central
novelty that is +0.0035 at p = 0.625 against its own ablation. F5(b): eight of eight
published rows below a majority baseline nobody computed.

So the gap is not "a better classifier on Cresci-2015". It is **features that describe
the *shape* of behaviour rather than its *magnitude*, evaluated under a protocol that can
actually detect whether they transfer.**

## 2. What `02-ext-research/` therefore does differently

| Finding | Design consequence |
|---|---|
| F1 — no headroom | Within-corpus accuracy is **diagnostic only**. The primary claim (H4) is **cross-dataset transfer**: fit on Cresci-2015, test on TwiBot-20, and compare *degradation* per feature family. |
| F2 — 128 dims of rank 10 | Every estimator must satisfy stated **properties** checked in code (`renyiext/checks.py`), not merely produce a number. Rank, invariance and limiting cases are asserted. |
| F3 — α inside noise | **The spectrum is not tuned over α.** The whole vector `[H₀, H½, H₁, H₂, H₄, H_∞]` is the feature. Tuning α is one-dimensional feature selection with a multiple-comparisons hazard, and F3 shows what that search finds when there is nothing there. |
| F4 — novelty vs its own ablation | Every claim carries **≥10 seeds and a paired Wilcoxon**, and an effect below **0.02 AUC** is not claimed regardless of p — 0.02 being the measured seed σ from F3/F4. |
| F5 — protocol and floors | Three protocols are **named and never mixed**. A **majority baseline** is printed beside every accuracy. Seven **mandatory floors** must be beaten before any claim, including the incumbent metadata features and the trivial ones. |

Full statements: `02-ext-research/docs/00-CHARTER.md` (hypotheses),
`02-ext-research/docs/02-PROTOCOL.md` (floors and protocols).

## 3. The honest status of that project so far

This section exists because the same standard applied to three published papers has to be
applied to the work that criticises them. Source:
`02-ext-research/bitacora/`, entries 00–04.

**Confirmed.** Cresci-2015's tweet timestamps, believed absent, are recoverable from the
Twitter snowflake IDs — 2,763,927 events, validated by 0 violations against an
independently-sourced field and by separating from a counter null (circadian
peak/trough 12.82 vs 1.00).

**A finding that reshaped the design.** Bots in Cresci-2015 post a median of **23** tweets
against humans' **834**, and **event count alone scores AUC 0.939**. The originally
planned bias control (fixed-n subsampling) proved unexecutable, and H1 was amended — on
the record, as a new numbered entry — to require the spectrum to beat *both* Shannon and
event count.

**H1 passed, and is qualified twice.** On the temporal front, 10 seeds, p = 0.0020:
the spectrum beats event count by **+0.0367** and its own α = 1 point by **+0.0380**,
stable across all 14 swept configurations. Operationally, TPR at 1% FPR goes **0.141 →
0.779**.

**But:** twelve Rényi orders add only **+0.019** over three classical burstiness numbers
— below the project's own 0.02 floor, so **that floor is recorded as not cleared**. And
the rendered α-curves are near-parallel rather than converging, so H1's specific
*tail-resolution* mechanism is **not** demonstrated; a weaker claim survives (removing the
level entirely, curve shape alone still clears the count floor at +0.0273).

**Note what this says about F3.** α was inside noise on neighbourhood degree
distributions (a handful of points per node) and is *not* inside noise on inter-arrival
distributions (hundreds to thousands of points per account). F3 is a finding about a
**substrate**, not about Rényi entropy. That distinction is the single most useful thing
this folder carries forward, and it was only visible because F3 measured the noise floor
first.

## 4. What is deliberately not claimed

Recorded here so the argument is not oversold when it is handed to someone else.

1. **None of this is about AI-generated text.** Every corpus involved (2015–2022) predates
   widespread LLM text and carries no ground truth for it. An earlier framing of the
   follow-on work drifted there and was explicitly ruled out
   (`02-ext-research/docs/00-CHARTER.md` §4, non-goal 1).
2. **No claim that the three papers are wrong.** F1–F5 are claims that the published
   *comparisons* do not isolate what they say they isolate. Two of the three replications
   reproduce their headline numbers closely.
3. **No claim about corpora not measured.** Everything is Cresci-2015, TwiBot-20/22,
   CollegeMsg, LIAR, Twitter15/16, PHEME and CED. TwiBot-22's graph and tweets are not in
   the open release, so the analysis there rests only on labels.
4. **No general claim about GNNs.** F5(c) shows one corpus whose graph is 99.88%
   dangling; that is a property of the corpus conversion.

## 5. The transferable lessons

If you take nothing else from this folder:

1. **Measure the noise floor before reading any sweep.** F3's whole argument is that a
   0.022 effect and a 0.018 σ cannot be distinguished — and both numbers were available.
2. **Ablate against your own novelty, not against unrelated baselines.** F4: DTWRE beat
   everything except the one comparison that isolated its contribution.
3. **Print the majority baseline.** F5(b): eight of eight published rows fell below one.
4. **Name the protocol on every number.** F5(a): 9.2 points, unreported.
5. **Look at the object before quoting a statistic of it.** The single biggest error
   caught in the follow-on work — 63,830 tweets decoding to one millisecond — was
   invisible to an elementwise check that passed at 0 violations, and visible immediately
   in a histogram (`02-ext-research/bitacora/01_p0_data_layer.md`).

## 6. Entry points

| To | Read |
|---|---|
| understand the evidence | [01-FINDINGS.md](01-FINDINGS.md) |
| check a specific number | [EVIDENCE-INDEX.md](EVIDENCE-INDEX.md) |
| build on it | `02-ext-research/README.md`, then `docs/00-CHARTER.md`, then `bitacora/` in order |
| see the implementation reasoning | `docs/DISCREPANCIES*.md` (three files, one per paper) |
