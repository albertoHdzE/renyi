# 01 — The five findings

Each finding states the claim, the table behind it, the source file, and what it does and
does not license. Every number is verified against the result file named beside it; see
[EVIDENCE-INDEX.md](EVIDENCE-INDEX.md) for regeneration commands.

Two noise floors are used throughout, because a difference below them is not a
difference:

| Pipeline | Noise floor | Source |
|---|---|---|
| bot-detection (Cresci-2015) | fold σ ≈ **0.0063** | `results_replicate.csv`, `accuracy_std` |
| entropia (CollegeMsg) | seed σ ≈ **0.0179** | `significance_auc.json`, 10 seeds, DTWRE row |

---

## F1 — The text branch is nearly free-standing, and the graph branch nearly free

**Claim.** On Cresci-2015 the published architecture concatenates a 128-dimensional
"GraphSage embedding" with a 768-dimensional BERT vector. Ablating the concatenation
shows the text half carries almost all of it, and that five raw metadata numbers with a
non-linear head match the whole pipeline.

**Cresci-2015, 5-fold stratified CV, majority baseline 0.6321**
Source: `01-info-propagation/bot-detection-paper/results/results_ablation.csv`

| feature set | dim | accuracy | ±SD | F1 |
|---|---|---|---|---|
| raw 5 metadata features | 5 | 0.8976 | 0.0116 | 0.9236 |
| effective 10 dims `[x ‖ mean N(v)]` | 10 | 0.9262 | 0.0110 | 0.9441 |
| GraphSage[128] | 128 | 0.9359 | 0.0063 | 0.9511 |
| **BERT[768] only** | 768 | **0.9730** | 0.0039 | 0.9784 |
| raw 5 + BERT[768] | 773 | 0.9764 | 0.0035 | 0.9811 |
| **GraphSage[128] + BERT[768]** (the paper) | 896 | **0.9779** | 0.0054 | 0.9823 |

**Read the last three rows.** Adding the entire 128-dimensional graph branch to BERT is
worth **+0.0049** (0.9730 → 0.9779), against a fold σ of 0.0054. Adding it to *raw 5 +
BERT* is worth **+0.0015**. Both are inside noise.

**And the converse.** From `results_trained.csv`:

| arm | accuracy | ±SD |
|---|---|---|
| untrained SAGEConv[128] + linear SVM (the paper) | 0.9359 | 0.0063 |
| **untrained SAGEConv[128] + MLP head — no text at all** | **0.9775** | 0.0051 |
| trained SAGEConv[128] + head | 0.9739 | 0.0049 |

**0.9775 using no text whatsoever**, against 0.9779 for the full 896-dimensional method.
Five metadata numbers, their neighbourhood means, and a small MLP.

**What this licenses.** The two branches are *substitutes*, not complements: either alone
reaches ~0.973–0.978 and the combination reaches 0.978. **What it does not license** is
"BERT is useless" — BERT alone (0.9730) beats the graph branch alone (0.9359) by 3.7
points. The claim is that on *this benchmark* the two are redundant with each other, and
the benchmark saturates below what either can express.

Detail: [02-THE-TEXT-BRANCH.md](02-THE-TEXT-BRANCH.md).

---

## F2 — The "GraphSage embedding" is 128 columns of rank 10, and is never trained

**Claim.** The source paper states (Sect. 3.5) that "training epochs and optimization
tasks are not required due to the lack of a prediction head", so the layer is used at
random initialisation. A single untrained `SAGEConv(5,128)` is an affine map of the
10-vector `[x_v ‖ mean N(v)]`, so its 128 columns have rank at most 10.

Source: `01-info-propagation/bot-detection-paper/results/checks.json`

- closed form reproduced to **< 1e-05**
- measured rank **10**, with σ₁₀/σ₁₁ = **6.8 × 10⁶** — 118 of 128 dimensions are
  linearly dependent on the other 10
- linear SVM on the 128-dim embedding **0.880** vs on the 10 raw dims **0.880**

**The apparent counterexample, and why it is not one.** At the default `C = 1` the
128-dim embedding scores +0.0096 over the 10 numbers it is computed from. L2
regularisation is not invariant under a change of basis, so this is preconditioning, not
information. Weaken the penalty and it must vanish — and it does
(`results_regularization.csv`):

| C | 10 dims | GraphSage[128] | Δ |
|---|---|---|---|
| 0.001 | 0.8249 | 0.8581 | +0.0332 |
| 0.01 | 0.8738 | 0.8885 | +0.0147 |
| 0.1 | 0.9055 | 0.9195 | +0.0140 |
| 1.0 | 0.9262 | 0.9359 | +0.0096 |
| 10 | 0.9408 | 0.9432 | +0.0025 |
| **100** | **0.9442** | **0.9445** | **+0.0004** |

Monotone to zero.

**Independent confirmation.** Varying *only* the untrained layer's random seed over 10
seeds moves accuracy by **0.0019** (0.9359 → 0.9377, `results_seeds.csv`), against a fold
σ of 0.0063. A random basis change cannot alter what a linear model can express, and it
does not.

**And training the layer buys nothing.** Trained (0.9739) is *below* untrained (0.9775),
inside noise. The binding constraint is the **linear kernel**, not the untrained layer:
replacing the linear SVM with an MLP is worth **+4.2 points** (0.9359 → 0.9775).

---

## F3 — Rényi's α has no measurable effect on the task it was introduced for

**Claim.** The entropia paper's central object is a Rényi entropy of order α, and it
reports α = 0.6 as optimal. Sweeping α on the replication finds the whole range inside
noise.

**CollegeMsg link prediction, 3 seeds.** Source: `results/alpha_sweep.json`

| α | AUC | ±SD | accuracy | ±SD |
|---|---|---|---|---|
| 0.2 | **0.8409** | 0.0076 | 0.7934 | 0.0109 |
| 0.6 | 0.8348 | 0.0050 | 0.7418 | 0.0424 |
| 1.0 | 0.8360 | 0.0063 | 0.7688 | 0.0092 |
| 1.5 | 0.8391 | 0.0091 | 0.7500 | 0.0377 |
| 2.0 | 0.8329 | 0.0080 | 0.7418 | 0.0419 |
| 5.0 | 0.8311 | 0.0060 | 0.7465 | 0.0505 |

**Total spread across a 25× range of α: 0.0098 AUC.** The seed-to-seed σ measured on the
same pipeline over 10 seeds is **0.0179** — the entire α effect is **half of one standard
deviation**.

The replication's best α is **0.2**, not the paper's 0.6.

**The paper's own numbers already showed this, if you knew the noise.** Its reported
sweep (`dtwre/config.py`, `PAPER_ALPHA_AUC`) spans 0.944 (α=5) to 0.966 (α=0.6) — a
spread of **0.022**, which is **1.2 σ**. A single-run protocol with no error bars cannot
separate that from noise, and the paper reports no seeds, no repetitions and no error
bars anywhere.

Detail: [03-THE-ALPHA-QUESTION.md](03-THE-ALPHA-QUESTION.md).

---

## F4 — The paper's own novelty is not significant against its own ablation

**Claim.** The entropia paper's contribution is the *time weighting* — DTWRE against a
static Rényi entropy. A 10-seed paired test cannot distinguish them.

Source: `results/significance_auc.json` — 10 seeds, paired Wilcoxon

| baseline | mean AUC | ±SD |
|---|---|---|
| Node2vec | 0.7897 | 0.0201 |
| Node PageRank | 0.8213 | 0.0195 |
| Node Degree | 0.8230 | 0.0192 |
| Rényi (static) | 0.8369 | 0.0145 |
| **DTWRE** | **0.8404** | 0.0179 |

| DTWRE vs | mean diff | wins | Wilcoxon p | significant |
|---|---|---|---|---|
| Node2vec | +0.0507 | 10/10 | 0.0020 | yes |
| Node PageRank | +0.0191 | 8/10 | 0.0371 | yes |
| Node Degree | +0.0174 | 8/10 | 0.0488 | yes |
| **Rényi (static)** | **+0.0035** | **6/10** | **0.625** | **no** |

DTWRE beats the *unrelated* baselines and does **not** beat the ablation that isolates
its own novelty. The published gap for the same comparison is 0.9742 − 0.9487 = **0.0255**
(`PAPER_TABLE1`), reported from a single run with no error bars — against a measured σ of
0.0179 on that comparison.

**This is the cleanest statement of the problem in the whole folder:** the paper's design
could not have determined whether its central mechanism contributes 0.004 or 0.04.

Corroborating: λ, the time-decay constant, *does* move the result — 0.8143 to 0.8711
across λ ∈ [0.1, 2] (`results/lambda_sweep.json`), a spread of 0.0568, about 3 σ. But the
replication's optimum is **λ = 0.4**, not the paper's stated 1.2. So the mechanism has an
effect; the reported operating point is not where it is.

---

## F5 — Protocol and baseline choices dominate method choices

**Claim.** Three separate measurements, each showing that a choice the papers treat as
incidental moves the result more than the method does.

### (a) 5-fold CV vs the corpus's own split — 9.2 points

Source: `results_protocol.csv`

| protocol | accuracy | F1 | majority |
|---|---|---|---|
| 5-fold CV (the paper) | 0.9779 | 0.9823 | 0.6321 |
| official train/test split | **0.8860** | 0.9036 | 0.6318 |

**A 9.2-point gap**, larger than the entire published spread between competing methods in
the source paper's own comparison table. It is not class imbalance — the balance is
essentially identical in both. The official split is simply not random: its test users
are systematically less active, and split membership is predictable from the five
features at **AUC 0.79** (`docs/DISCREPANCIES_BOTSAGE.md` §5).

### (b) TwiBot-22 — 8 of 8 published rows sit below the majority baseline

Source: `twibot22_baselines.json`, computed from the released `label.csv`

| split | n | bot % | majority baseline |
|---|---|---|---|
| corpus | 1,000,000 | 13.99 | **0.8601** |
| train | 700,000 | 7.80 | 0.9220 |
| val | 200,000 | 27.96 | 0.7204 |
| test | 100,000 | 29.44 | **0.7056** |

The source paper evaluates by 5-fold CV over the corpus, so **0.8601** is the floor its
accuracies must clear. Its own best row is 74.62 and the best baseline it cites is 79.66
— all eight rows are below 86.01. Worse, the cited baselines come from the TwiBot-22
leaderboard, computed on the **official test split** whose floor is 0.7056. Two columns
describing tasks with class balances differing by a factor of two are placed side by side
without comment.

The same pattern appears in the survey replication: on LIAR, every architecture lands
0.2238–0.2427 against a majority baseline of **0.2056**, with AUC 0.57–0.59
(`results/disinfo/results_gnn_comparison.csv`).

### (c) The Cresci-2015 graph is 99.88% unusable

Source: `cresci_graph_degeneracy.json`

| quantity | value |
|---|---|
| user–user edges | 6,994,858 |
| edges joining two users that both have metadata | **8,550 (0.12%)** |
| labelled users isolated in that subgraph | **3,381 of 5,301 (63.8%)** |

For roughly two thirds of labelled users the graph layer reduces to a random linear map
of their own five features, with no network contribution at all. Restricting to the
usable edges changes accuracy by −0.0066 (0.9359 → 0.9293, `results_graph_scope.csv`) —
i.e. the graph branch is doing almost nothing either way.

Detail: [04-PROTOCOL-AND-FLOORS.md](04-PROTOCOL-AND-FLOORS.md).

---

## What the survey replication adds

The third paper is a survey, so it has no experiments of its own. Running the
architectures it catalogues produces one result worth carrying:

**Twitter15, GCN, propagation graph, 3 seeds** (`results/disinfo/results_feature_ablation.csv`)

| feature set | accuracy | ±SD |
|---|---|---|
| all features | 0.7539 | 0.0312 |
| without lexical | **0.3702** | 0.0331 |
| without syntactic | 0.7528 | 0.0312 |
| without profile | 0.7539 | 0.0312 |
| without propagation | 0.7517 | 0.0293 |
| without temporal | 0.7562 | 0.0273 |
| **lexical only** | **0.7685** | 0.0497 |

Removing lexical features costs **38 points**. Removing anything else costs nothing —
*"without profile" is identical to "all features" to four decimal places*. And
**lexical-only (0.7685) scores above the full model (0.7539).**

This is F1 again on a different corpus and a different architecture: the pipeline has one
load-bearing feature family, and the rest is decoration.

Separately, the survey's Sect. 7 claim that multiclass accuracies are "typically below
50%" is contradicted by its own tables — **1 of 36** reported accuracies is below 0.5,
median **0.881** (`results/disinfo/claim_verification.csv`). Five of its six checkable
prose claims hold; that one does not.
