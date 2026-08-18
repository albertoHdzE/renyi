# 03 — Phases and gates

Ten phases. Each has a deliverable, a gate, and a decision rule. **A gate that fails
stops the branch it belongs to** — it is not tuned until it passes. A failed gate is a
result and is written up as one.

Estimates assume one person working with the existing repository as a base. They are
effort, not calendar.

---

## P0 — Data layer and event reconstruction  ·  ~2 days

**Deliverable.** `renyiext/events.py`: snowflake decoding, per-account chronological
event series, post-type classification, mention extraction. Cached to
`data/processed/ext/`.

**Work.**
1. Decode all 2,827,757 Cresci-2015 tweet IDs (`(id >> 22) + 1288834974657`).
2. Build per-account event tables: `(timestamp, type, mentions, text)`.
3. Classify post type into the 4-symbol alphabet from text/edge structure.
4. Report the event-count distribution per class, and how many accounts survive the
   `n_events = 128` cutoff.

**Gate G0.** P14–P15 pass (all timestamps in [2010-11-01, 2013-07-01], ordering agrees
with ID order); ≥ 80% of labelled accounts survive the cutoff **in both classes**.

**Decision rule.** If one class survives the cutoff far less often than the other, lower
`n_events` until both clear 80%, and record the value. If no value clears it, the
temporal front runs on a restricted subpopulation and **that restriction is reported in
every subsequent result**.

---

## P1 — Rényi spectrum estimator  ·  ~2 days

**Deliverable.** `renyiext/spectrum.py` + the first half of `renyiext/checks.py`.

**Work.** Implement `H_α` with the stable log-sum-exp form and exact limiting cases at
α ∈ {0, 1, ∞}; fixed-n bootstrap subsampling; the 6-vector readout.

**Gate G1.** Properties **P1–P8** all pass. In particular **P8** — after fixed-n
subsampling, |ρ(H_α, total event count)| < 0.1 for every α.

**Decision rule.** P8 is not negotiable. If it fails, every subsequent result is
confounded with activity volume and the project has measured nothing. Fix the estimator,
not the threshold.

---

## P2 — Temporal front  ·  ~3 days

**Deliverable.** `SPEC_T` features; `results/p2_temporal.{json,csv}`; the α-curve figure
with error bands, per class.

**Work.** Spectra on inter-arrival and circadian distributions. Protocol A, ≥10 seeds.
Compare against floors 1, 2, 3, 6 (§3 of [02-PROTOCOL.md](02-PROTOCOL.md)).

**Gate G2 = H1.** `AUC(SPEC_T) − AUC(Shannon only) > 0.02`, paired Wilcoxon p < 0.05,
and `SPEC_T` beats coefficient of variation and Fano factor.

**Decision rule.** If the α-curve is flat — spread across α within seed noise — that is
the DTWRE outcome repeating, H1 is falsified, and P3 proceeds only to test whether other
fronts behave differently. If **two** fronts show flat curves, stop and write up the
negative result.

---

## P3 — Behavioural and text fronts  ·  ~3 days

**Deliverable.** `SPEC_B`, `SPEC_X`; `results/p3_fronts.{json,csv}`.

**Work.** Spectra on post-type, mention-target, word- and character-frequency. Front X
on **raw uncleaned text** (decision D4).

**Gate G3 = H2.** Directional predictions hold: bots show lower H₂ on behavioural
alphabets and lower H₀ on client/interaction support, p < 0.05, **with consistent sign
across fronts**.

**Decision rule.** Inconsistent signs indicate a volume or length confound rather than a
shape effect. If signs are inconsistent, return to P1 and re-examine P8 before
proceeding.

**Note.** `H₂` on word frequency ≈ Yule's K. If Front X succeeds, the write-up must say
so and position the contribution as the *spectrum*, not the rediscovery.

---

## P4 — Digital DNA, BDM 1.0, NCD  ·  ~4 days

**Deliverable.** `renyiext/dna.py`, `renyiext/ait.py`; `AIT` features;
`results/p4_ait.{json,csv}`.

**Work.** 4-symbol action-DNA and temporal-DNA encodings. BDM 1.0 via `pybdm`. Pairwise
NCD (`zlib`, `bz2` check). Group cohesion against size-matched random null.

**Gate G4 = H3.** NCD group cohesion separates bot from human groups at AUC > 0.75
**and** exceeds `H₂` on the same strings by > 0.02. Floors 4 and 5 (gzip ratio, block
entropy) must both be beaten.

**Decision rule.** If BDM 1.0 fails to beat block entropy, that is the expected outcome
per Sakabe et al. §2.4 and is *not* a reason to stop — it is the motivation for P7.
If **NCD** fails to beat `H₂`, the coordination hypothesis is falsified and **P7/P8 are
cancelled**, since BDM 2.0's reuse gain measures the same quantity more expensively.

---

## P5 — Network front  ·  ~3 days

**Deliverable.** `SPEC_N`, graph-BDM with controls; `results/p5_network.{json,csv}`.
TwiBot-20 only.

**Work.** Ego-degree spectra. Graph BDM under canonical ordering, with relabelling SD
(K = 50) and configuration-model z-scores (100 rewirings).

**Gate G5.** Relabelling SD **below** the between-class effect (property P13), and the
configuration-model z-score is non-zero at p < 0.05.

**Decision rule.** If relabelling SD exceeds the class effect, graph BDM is dropped —
decided in advance, per §5 of [01-METHODS.md](01-METHODS.md). `SPEC_N` may still
proceed; it is permutation-invariant by construction.

---

## P6 — Cross-dataset generalisation  ·  ~5 days  ·  **PRIMARY**

**Deliverable.** `results/p6_transfer.{json,csv}`; the degradation figure — AUC within
corpus vs AUC transferred, per feature family.

**Work.** Protocol C: fit on Cresci-2015, test on TwiBot-20, standardisation fitted on
source only. Every family from §2 of [02-PROTOCOL.md](02-PROTOCOL.md), ≥10 seeds.

**Gate G6 = H4.** `Δ_META − Δ_ours > 0.05` over ≥10 seeds, paired Wilcoxon p < 0.05.

**This is the phase the project exists for.** P2–P5 earn the right to run it; a positive
G6 with weak G2–G5 is still the publishable result, and a negative G6 with strong
G2–G5 is a within-corpus curiosity and should be reported as such.

---

## P7 — BDM 2.0, Tiers 1–2  ·  ~5 days  ·  gated on G4

**Deliverable.** `renyiext/bdm2.py`; `AIT2` features; `results/p7_bdm2.{json,csv}`.

**Work.** Hybrid estimator: CTM tables for unconditional terms, compression-based
conditionals (Tier 1), transformation library `T` (Tier 2, cyclic shift load-bearing).
Greedy selection of `S`, with the greedy/exact gap measured on small synthetic instances.

**Gate G7.** Properties **P10–P11** pass (`BDM2 ≤ BDM1 + C_rep`; reuse detected on
cyclic-shift constructions), **and** `AIT2` beats `NCD` on the H3 task by > 0.02.

**Decision rule.** If P10 fails, the implementation is wrong — it violates the paper's
Theorem 1. If P10 passes but `AIT2` does not beat NCD, report that: *on this data, the
hybrid reuse gain adds nothing over compression-based mutual information.* That is a
genuinely useful first empirical datum on BDM 2.0 and should be written up either way.

---

## P8 — Conditional CTM  ·  ~2 months  ·  optional, gated on G7

**Deliverable.** Partial-enumeration conditional CTM tables for the 4-symbol alphabet at
short block length, plus the Tier-3 estimator.

**Only if** G7 passes *and* the Tier 1/2 conditionals are shown to be the binding
constraint (i.e. reuse gain is limited by conditional estimation quality, not by absence
of reuse). Off the critical path by design.

**Decision rule.** Before starting, budget the machine-space sample size and validate
against known unconditional values by setting `x = blank tape` — the conditional
estimator must reproduce the published unconditional table in that limit. If it does
not, stop.

---

## P9 — Write-up and artefacts  ·  ~4 days

**Deliverable.** `notebooks/replication.ipynb` (build artefact of
`scripts/build_ext_notebook.py`, pinned to its own kernel); all figures; a
`DISCREPANCIES`-style findings document; `README.md` phase board updated.

**Gate G9.** The notebook reproduces every figure from a clean checkout, `checks.py`
passes end to end, and every reported number carries seeds, error bars and its majority
baseline.

---

## Dependency graph

```
P0 ──► P1 ──┬──► P2 ──┐
            ├──► P3 ──┤
            ├──► P4 ──┼──► P6 (PRIMARY) ──► P9
            └──► P5 ──┘      │
                 P4 ──► P7 ──┴──► P8 (optional)
```

P2, P3, P4, P5 are independent given P1 and can run in any order or in parallel.
P6 needs whichever of them passed. P7 needs P4's gate. P8 needs P7's.

## Kill criteria for the project as a whole

Stop and write up the negative result if **either**:

- G1 (P8, bias neutrality) cannot be met at any `n_events` — the features would be
  confounded with activity volume and nothing is measurable; or
- **two or more** of G2, G3, G4 fail — the distributional-shape hypothesis has no
  support on any front, and G6 has nothing left to transfer.

Both are honourable outcomes. Per [00-CHARTER.md](00-CHARTER.md) §3, H0 is a
pre-registered, publishable result, and this repository already contains a precedent for
a negative finding reported as such (`disinfo` claim `multiclass_below_50`, contradicted
by 35 of 36 accuracies in the survey's own tables).
