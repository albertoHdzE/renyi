# 00 — Charter

Objectives, hypotheses, scope. This document defines what would count as success and
what would count as failure. It is written **before** any experiment so that the
hypotheses are pre-registered and cannot be retrofitted to whatever the data shows.

---

## 1. Motivation, and what the prior work established

Three replications in `01-info-propagation/` produced four findings that motivate and
constrain this project. Each is a measured number in this repository, not an opinion.

**(a) Metadata magnitude already saturates the standard benchmark.** On Cresci-2015,
five raw metadata counts plus a neighbourhood mean, with a small MLP, reach **0.9775**
accuracy — against **0.9779** for the full 896-dimensional GraphSAGE+BERT pipeline
(`01-info-propagation/bot-detection-paper/results/results_ablation.csv`). The text and
graph branches contribute essentially nothing. There is no headroom on this benchmark,
so *peak accuracy on Cresci-2015 cannot be the objective of this project.*

**(b) Protocol dominates method.** The same pipeline scores **0.9779** under 5-fold CV
and **0.8860** under Cresci-2015's official split — a 9.2-point gap, larger than the
entire published spread between competing methods. The official split is not random:
test users are systematically quieter, and split membership is predictable from the
five features at **AUC 0.79** (`docs/DISCREPANCIES_BOTSAGE.md` §5). Any claim about
generalisation must therefore be made against an explicit, stated protocol.

**(c) Metadata features are brittle by construction.** Follower counts are purchasable
and account age is era-specific. Finding (b) is the local symptom: the features fail
even under a mild within-corpus distribution shift toward less active accounts. This is
the gap this project targets.

**(d) A cautionary precedent on α.** The DTWRE replication swept α ∈ [0.2, 5] on
CollegeMsg link prediction and found a total AUC spread of **0.0098** against a
seed-to-seed σ of **~0.02** (`results/alpha_sweep.json`, `results/significance_auc.json`).
α was inside noise there. That result is about the *degree distribution of a
neighbourhood* in a *link-prediction* pipeline, and does not transfer to the
distributions and task studied here — but it sets the effect size we must clear and the
standard of evidence we must meet.

## 2. Objectives

**O1 — Spectrum.** Determine whether the Rényi spectrum `{H_α}` of account-level
behavioural distributions separates bots from humans **beyond Shannon entropy alone**.

**O2 — Algorithmic structure.** Determine whether algorithmic-information measures over
behavioural sequences detect **coordination** — accounts sharing a generating program —
beyond what marginal distributional statistics capture.

**O3 — Generalisation (primary).** Determine whether O1/O2 features degrade **less than
metadata features** under dataset and era shift.

**O4 — Artefact.** Deliver a reusable, property-checked package and a reproducible
notebook, to the standard of the three existing replications.

## 3. Pre-registered hypotheses

Each states a directional prediction, a threshold, and a test. Thresholds are set from
the measured noise floors in §1: within-dataset fold σ ≈ 0.0065, seed σ ≈ 0.02.

---

**H1 — The spectrum beats its own α = 1 point.**
On the temporal front, the 6-vector `H_α`, α ∈ {0, ½, 1, 2, 4, ∞}, achieves
`AUC − AUC(H₁ alone) > 0.02`, paired Wilcoxon over ≥10 seeds, p < 0.05.

*Rationale.* Human inter-event times are bursty and heavy-tailed (Barabási 2005);
scheduled bots are Poisson or periodic. These regimes differ **specifically in the
tail**, which is what α resolves and what α = 1 averages over.

*Falsified if* the α-curve is flat, i.e. the spread across α is within seed noise — the
DTWRE outcome (§1d) repeating on a new substrate.

---

**H2 — The discrimination is tail-directional, not arbitrary.**
Bots show **higher** collision probability (lower H₂) on behavioural-alphabet and
mention-target distributions, and **lower** support richness (lower H₀) on client-source
distributions, both at p < 0.05.

*Rationale.* H₂ = −log Σp² is a repetition statistic; template reuse raises Σp². H₀ =
log|support| is a richness statistic; single-client API automation collapses it.

*Falsified if* the effect exists but with inconsistent sign across fronts, which would
indicate we are measuring activity volume rather than shape (see H2-confound in
[05-RISKS.md](05-RISKS.md) R2).

---

**H3 — Algorithmic mutual information detects coordination.**
Within-group NCD cohesion over digital-DNA strings separates bot groups from human
groups with `AUC > 0.75`, and **exceeds** what H₂ on the same strings achieves by
> 0.02.

*Rationale.* Cresci et al.'s *Social Fingerprinting* (IEEE TDSC 2017) detected spambot
groups on this very corpus by longest-common-substring over behavioural strings. LCS is
a crude proxy for shared algorithmic information; NCD is the principled version, and
BDM 2.0's reuse gain is the principled version of *that*.

*Falsified if* NCD adds nothing over H₂, i.e. the coordination signal is fully explained
by marginal repetition.

---

**H4 — Generalisation (the primary claim).**
Transferring Cresci-2015 → TwiBot-20, the AUC degradation of spectrum + AIT features is
smaller than that of the five metadata features by more than **0.05**:
`Δ_metadata − Δ_ours > 0.05`, over ≥10 seeds.

*Rationale.* Tail shape of temporal behaviour is a behavioural invariant and is costly
to fake while remaining economically viable as an automation operation. Follower counts
are a purchasable scalar.

*Falsified if* both families degrade equally, or if ours degrades more.

---

**H0 — The null is a publishable outcome.** If H1–H4 all fail, the result is a
negative finding with a measured effect size and a clear precedent (§1d), reported as
such. This project does not require a positive result to be worth completing.

## 4. Scope

**In scope.** Account-level bot-vs-human classification on Cresci-2015, TwiBot-20 and
TwiBot-22 (labels only). The four fronts: temporal, behavioural, textual, network.
Rényi spectrum estimation with finite-sample bias control. BDM 1.0 and NCD.
BDM 2.0 Tiers 1–2 (see [01-METHODS.md](01-METHODS.md) §5).

**Out of scope — explicit non-goals.**

1. **AI-generated-text detection.** None of these corpora (2015–2022) carries ground
   truth for machine-generated text. That is a different project needing a different
   dataset. Recorded here because an earlier framing of this work drifted toward it.
2. **Beating state of the art on Cresci-2015 accuracy.** Ruled out by §1a; there is
   nothing to demonstrate at a 0.978 ceiling with 0.0065 fold noise.
3. **Reimplementing CTM.** Lookup tables exist (`pybdm`, `acss.data`). Tier 3
   conditional CTM by partial enumeration is optional and off the critical path.
4. **Modifying the three existing replications.** `dtwre/`, `disinfo/` and `botsage/`
   are frozen. This project may *import* from them; it may not change them.
5. **Deep architecture search.** The classifier is held fixed so that feature families
   are the only thing varying.

## 5. Success criteria

**Minimum (the project is complete and reportable):** every phase gate resolved
pass or fail with a measured number and stated error bars; the notebook reproduces
every figure from a clean checkout; `checks.py` passes.

**Target:** H1 and H4 supported. This is the publishable result — a feature family that
is *more transferable* than metadata, with a mechanism (tail shape as behavioural
invariant) rather than an empirical accident.

**Stretch:** H3 supported and BDM 2.0 Tier 1–2 reuse gain beating NCD, giving the first
empirical evaluation of BDM 2.0 on real data.

## 6. Standard of evidence

Non-negotiable, inherited from §1 and from the failures documented in the three
`DISCREPANCIES` files:

- **≥10 seeds** and **paired Wilcoxon** for every comparison. Single-run numbers are
  not reportable. The DTWRE paper's central claim could not be evaluated because it
  lacked this.
- **Majority baseline printed beside every accuracy**, always. Eight of eight rows of
  the botsage paper's Table 5 sit below their own majority baseline.
- **Floors before ceilings.** A new feature must beat: majority, the 5 metadata
  features, Shannon-only, gzip compression ratio, and block entropy — before any claim
  is made about it.
- **TPR at FPR = 1%** reported alongside AUC. Suspending a real user is the operational
  cost of a false positive, and AUC hides the low-FPR regime where deployment happens.
- **Protocol stated with every number.** 5-fold CV, official split, and cross-dataset
  are three different tasks (§1b).
