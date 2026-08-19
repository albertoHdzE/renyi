# 03 — The α question

Deep dive on **F3** and **F4**. The question this answers: *the entropia paper's central
object is a Rényi entropy of order α. How much does α matter?*

Answer, on the task the paper introduced it for: **not at all measurably.** And the
paper's own reported numbers already showed this, if the noise floor had been measured.

All numbers are CollegeMsg link prediction. Sources named per table; regeneration in
[EVIDENCE-INDEX.md](EVIDENCE-INDEX.md).

---

## 1. What α is doing in this pipeline

Rényi entropy of order α interpolates a whole family of distributional summaries:

```
H_α(p) = 1/(1−α) · log( Σᵢ pᵢ^α )
```

with `H₀` = log support size, `H₁` = Shannon, `H₂` = collision entropy, `H_∞` =
min-entropy. Lowering α weights the tail; raising it weights the mode. That is a real and
useful degree of freedom **when the distributions being compared differ in their tails.**

In the entropia paper, the distribution is the **degree distribution of a node's
neighbourhood** in one temporal snapshot (Eq. 1), and the entropy enters as one feature
block inside a 67-dimensional GraphSAGE link-prediction pipeline. The paper reports
α = 0.6 as optimal (Sect. 4.2.1).

## 2. The measurement

`results/alpha_sweep.json` — 3 seeds, error bars from the seeds:

| α | AUC | ±SD | accuracy | ±SD |
|---|---|---|---|---|
| **0.2** | **0.8409** | 0.0076 | 0.7934 | 0.0109 |
| 0.6 | 0.8348 | 0.0050 | 0.7418 | 0.0424 |
| 1.0 | 0.8360 | 0.0063 | 0.7688 | 0.0092 |
| 1.5 | 0.8391 | 0.0091 | 0.7500 | 0.0377 |
| 2.0 | 0.8329 | 0.0080 | 0.7418 | 0.0419 |
| 5.0 | 0.8311 | 0.0060 | 0.7465 | 0.0505 |

**Spread across a 25-fold range of α: 0.0098 AUC.**

Now the noise floor. From `results/significance_auc.json`, the same pipeline over
**10 seeds**, DTWRE configuration:

> seed-to-seed **σ = 0.0179**

**The entire α effect is 0.55 σ.** There is no ordering to read off this table that is
not consistent with pure seed variation, and the replication's argmax (α = 0.2) is not
the paper's stated optimum (α = 0.6).

## 3. The paper's own numbers already showed it

`dtwre/config.py`, `PAPER_ALPHA_AUC`, transcribed from Sect. 4.2.1:

| α | 0.2 | 0.6 | 1.0 | 1.5 | 2.0 | 5.0 |
|---|---|---|---|---|---|---|
| published AUC | 0.950 | **0.966** | 0.959 | 0.955 | 0.952 | 0.944 |

**Published spread: 0.022.** Measured σ on this pipeline: **0.0179**. The paper's entire
α effect is **1.2 σ**, reported from a protocol with **no seeds, no repetitions and no
error bars anywhere in the paper.**

This is the important point, and it is not a criticism that requires our replication to
be right about anything else: *a single-run sweep cannot distinguish a 0.022 effect from
run-to-run variation, and the paper provides nothing with which to try.*

## 4. Why the flatness is not surprising, mechanically

α resolves differences in the **tail** of a distribution. For α to matter, the
distributions being compared must differ in shape, not just in scale.

The distribution here is `p_u ∝ deg(u)` over the neighbours of `v`, renormalised per
neighbourhood. For most nodes in a message graph, that is a small set — often 2 to 20
neighbours — with degrees spread over one or two orders of magnitude. A distribution on a
handful of points has very little tail for α to resolve, and `H_α` for such a `p` is
dominated by the support size, which is the degree of `v`.

So α is being asked to distinguish shapes in objects that barely have a shape. That is a
statement about the *substrate*, not about Rényi entropy — and it is why the follow-on
project applies the same family to **inter-arrival time distributions** with hundreds to
thousands of observations per account instead.

## 5. F4 — the paper's novelty against its own ablation

The paper's contribution is not the Rényi entropy; it is the **time weighting** (Eq. 3–4)
that turns a static entropy into DTWRE. The right ablation is therefore DTWRE against
static Rényi.

`results/significance_auc.json` — 10 seeds, paired Wilcoxon:

| method | mean AUC | ±SD |
|---|---|---|
| Node2vec | 0.7897 | 0.0201 |
| Node PageRank | 0.8213 | 0.0195 |
| Node Degree | 0.8230 | 0.0192 |
| Rényi (static) | 0.8369 | 0.0145 |
| **DTWRE** | **0.8404** | 0.0179 |

| DTWRE vs | mean diff | ±SD | wins | p | significant |
|---|---|---|---|---|---|
| Node2vec | +0.0507 | 0.0184 | 10/10 | 0.0020 | yes |
| Node PageRank | +0.0191 | 0.0244 | 8/10 | 0.0371 | yes |
| Node Degree | +0.0174 | 0.0251 | 8/10 | 0.0488 | yes |
| **Rényi (static)** | **+0.0035** | 0.0221 | **6/10** | **0.625** | **no** |

**DTWRE beats every baseline that is not its own ablation, and does not beat its own
ablation.** 6 wins in 10 is what a coin does.

The published version of the same comparison is 0.9742 vs 0.9487, a gap of **0.0255** —
from a single run, against a measured σ of 0.0179.

### The time weighting does do something, just not where the paper says

`results/lambda_sweep.json` — λ is the exponential decay constant in Eq. 4:

| λ | 0.1 | 0.4 | 0.8 | 1.2 | 2.0 |
|---|---|---|---|---|---|
| AUC | 0.8660 | **0.8711** | 0.8410 | 0.8354 | 0.8143 |
| ±SD | 0.0177 | 0.0068 | 0.0124 | 0.0079 | 0.0074 |

**Spread 0.0568 ≈ 3 σ** — a real effect, unlike α. But the replication's optimum is
**λ = 0.4**, and the paper's stated optimum is **λ = 1.2**, which sits near the bottom of
the range measured here.

So the honest summary of the entropia paper's mechanism is: *the time weighting matters,
the order α does not, and the reported operating point for the one that matters is not
where the optimum is.*

## 6. A further internal inconsistency, for completeness

The paper reports its own DTWRE result three times under nominally identical settings
(`docs/DISCREPANCIES.md` §5):

| source | setting | AUC |
|---|---|---|
| Table 1 | DTWRE, CollegeMsg | 0.9742 |
| Figure 6 | α = 0.6, the stated optimum | 0.966 |
| Table 2 | 604,800 s window, the stated optimum | 0.9680 |

A spread of **0.0062** between three descriptions of the same configuration. With no
seeds or error bars reported it is impossible to tell whether that is run-to-run variance
or three different protocols — which is the same diagnosis as §3.

## 7. What this does and does not license

**Licensed:**

- On this task and this substrate, α ∈ [0.2, 5] moves AUC by 0.0098 against a σ of
  0.0179. α is not a useful knob here.
- The paper's own reported α effect (0.022) is 1.2 σ and was measured without error bars.
- DTWRE vs static Rényi is **not** significant: +0.0035, 6/10, p = 0.625.
- λ *is* significant (3 σ) but the paper's stated optimum is not the measured one.

**Not licensed:**

- *"Rényi entropy is useless."* This is one substrate — neighbourhood degree
  distributions with few points — inside one link-prediction pipeline. The family is not
  on trial; this application of it is.
- *"α never matters."* The follow-on project tests the same family on inter-arrival
  distributions with hundreds of points per account and finds a **different** answer
  there (`02-ext-research/bitacora/04_p2_temporal.md`). See
  [05-WHY-A-NEW-PROTOCOL.md](05-WHY-A-NEW-PROTOCOL.md) §3.
- Any claim about the paper's Weibo results. The corpus the paper used is
  under-specified and the replication had to substitute a different subset
  (`docs/DISCREPANCIES.md` §9), so those numbers are not comparable.

## 8. Where to see it

- `01-info-propagation/entropia-paper/replication.ipynb` — narrative version, including
  the α and λ sweeps with error bands
- `results/figures/figure6_alpha.png` — the α sweep with error bands
- `results/figures/figure7_lambda.png` — the λ sweep
- `results/figures/figure5_comparison.png` — Table 1 replication vs published
- `docs/DISCREPANCIES.md` §§1, 2, 5 — why Eq. 1 needs renormalising, why Eq. 4's units
  had to be reinterpreted, and the internal inconsistency in §6 above
