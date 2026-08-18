# 05 — Threats to validity

Every risk that could produce a result which looks real and is not, with the control for
each and the phase that owns it. Ordered by how likely each is to silently succeed.

---

## R1 — Activity volume masquerading as distributional shape · **highest**

**Mechanism.** Plug-in Rényi estimators are biased in a way that depends on sample size,
worst at small α (`H₀` = log observed support grows monotonically with n). Bots and
humans post at different rates. An uncorrected spectrum would encode posting volume, and
since volume is already in `META` (tweet count), the "new" features would be a noisy
re-encoding of an incumbent feature — while appearing to work.

**Why it is the top risk.** It produces a *positive* result. Nothing about the output
looks wrong.

**Control.** D3: fixed-n subsampling at `n_events = 128`, `B = 100` draws. Verified by
property **P8**: |ρ(H_α, total event count)| < 0.1 for every α. This is gate **G1** and
is a hard stop.

**Owner.** P1. **Residual risk after control:** low, but P8 must be re-checked on each
corpus separately, not just Cresci-2015.

---

## R2 — Text length confounding on Front X

**Mechanism.** Word- and character-frequency spectra are length-sensitive by the same
mechanism as R1, and bot tweets differ in length distribution from human ones.

**Control.** Fixed-n applies at the *token* level for Front X, not the post level.
Report length distributions per class alongside. H2's requirement of **consistent sign
across fronts** is the detector: a length artefact will not produce coherent directional
effects across independent fronts.

**Owner.** P3.

---

## R3 — Graph BDM measuring node ordering or edge density

**Mechanism.** BDM over an adjacency matrix is not permutation-invariant, so its value
partly reflects the arbitrary node labelling. Separately, on sparse matrices most blocks
are all-zero and BDM becomes a proxy for edge density.

**Control.** Canonical ordering (degree, BFS tie-break); relabelling SD over K = 50
permutations reported as part of the feature; **degree-preserving configuration-model
z-score** (100 double-edge-swap rewirings) rather than raw BDM. Property **P13**.
Pre-committed decision rule: if relabelling SD exceeds the between-class effect, the
feature is dropped.

**Owner.** P5. **Note:** this repository's existing `checks.py` files assert permutation
equivariance for every layer; graph BDM would fail that house standard, which is why the
control is explicit rather than assumed.

---

## R4 — No headroom on Cresci-2015

**Mechanism.** The benchmark saturates at 0.9779, and five metadata features with an MLP
reach 0.9775. Fold σ is 0.0065. Any new feature will appear to help or not at random,
and a 0.002 "improvement" is unfalsifiable.

**Control.** Cresci-2015 within-corpus results are treated as *diagnostic*, never as the
claim. The claim is **H4**, measured under Protocol C where there is real headroom. This
is why P6 is designated primary rather than a follow-up.

**Owner.** P6, and the charter (non-goal 2).

---

## R5 — Protocol conflation

**Mechanism.** 5-fold CV, official split and cross-dataset differ by up to 9.2 points on
this pipeline — larger than the entire published spread between competing methods.
Placing numbers from two protocols in one table without labels produces a comparison
that means nothing. The source paper did exactly this in its Table 5, where eight of
eight rows sit below their own majority baseline.

**Control.** §4 of [02-PROTOCOL.md](02-PROTOCOL.md): three named protocols, never mixed
in a column without a label; majority baseline printed beside every accuracy.

**Owner.** every phase.

---

## R6 — BDM 1.0 rediscovering block entropy and being reported as a complexity result

**Mechanism.** For objects long relative to block size, BDM 1.0's `log₂ mᵢ` term
dominates and it converges to block Shannon entropy — stated by Sakabe et al. §2.4 as
the motivation for BDM 2.0. A BDM feature could "work" purely as a badly-normalised
entropy.

**Control.** Block Shannon entropy at the same block size is a **mandatory floor**
(floor 5), as is gzip ratio (floor 4). Property **P9** requires BDM 1.0 to be
demonstrably below its own block-entropy value on a periodic construction, confirming the
implementation can see algorithmic structure at all.

**Owner.** P4.

---

## R7 — Multiple comparisons across α, fronts and families

**Mechanism.** Six α values × six fronts × several families is a large comparison
surface. Something will reach p < 0.05.

**Control.** H1–H4 are pre-registered in [00-CHARTER.md](00-CHARTER.md) §3 with
directional predictions and thresholds, so no correction applies to them. Everything else
is labelled **exploratory** and Holm-corrected within its family. The 0.02 AUC effect-size
floor applies regardless of p. The spectrum is used as a **whole vector** rather than a
tuned α, which removes the largest single source of selection.

**Owner.** P2–P6.

---

## R8 — Transfer failing for boring reasons rather than the hypothesis

**Mechanism.** TwiBot-20's features are not Cresci's (screen-name length replaces
favourites) and arrive already z-scored. A `META` collapse under Protocol C could be a
schema artefact rather than genuine brittleness — which would make H4 pass for the wrong
reason.

**Control.** D8: `META` reduced to the four overlapping fields, dropped from **both**
sides, and the drop reported. Report `META` transfer both with and without the schema
alignment, so the artefact's size is visible. If the H4 effect exists only without
alignment, it is an artefact and is reported as one.

**Owner.** P6. **This is the control most likely to be forgotten**, because it weakens a
result we want.

---

## R9 — Greedy reuse selection producing arbitrary BDM 2.0 values

**Mechanism.** Exact reuse selection is NP-hard; a greedy approximation may be far from
optimal, and its error may correlate with object properties (length, repetitiveness) that
themselves correlate with class.

**Control.** Measure the greedy/exact gap on small synthetic instances where exhaustive
search is tractable, across the range of repetitiveness present in the real data.
Property **P10** bounds the result from above (`BDM2 ≤ BDM1 + C_rep`). Report the gap
distribution alongside any BDM 2.0 number.

**Owner.** P7.

---

## R10 — Building on an unmaintained dependency

**Mechanism.** `pybdm` is v0.1.0, classified "2 - Pre-Alpha", last released 2019. A
silent bug in its CTM tables or block handling would propagate into every AIT result.

**Control.** Cross-validate `pybdm` against `acss.data` (CRAN, independently ported) on
a sample of short strings — two independent implementations of the same published
tables. Any disagreement is a finding. Pin the version. Vendor the tables if the package
proves unreliable.

**Owner.** P4.

---

## R11 — Scope drift back toward AI-generated-text detection

**Mechanism.** The proximate framing of this work drifted there once already. None of
these corpora carries ground truth for machine-generated text, so any such claim would be
unfalsifiable on this data.

**Control.** Non-goal 1 in [00-CHARTER.md](00-CHARTER.md) §4, stated explicitly. Any
LLM-era claim requires a new corpus and is a separate project. This risk is listed
because documenting a known failure mode is cheaper than repeating it.

**Owner.** the charter.
