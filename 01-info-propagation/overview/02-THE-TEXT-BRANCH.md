# 02 — The text branch, and what the graph branch adds

Deep dive on **F1** and **F2**. The question this answers: *in a published architecture
that concatenates a graph embedding with a BERT embedding, what does each half actually
contribute?*

Everything here is Cresci-2015, 5-fold stratified CV, majority baseline **0.6321**, fold
σ ≈ **0.0063**. Source files are named per table; regeneration commands in
[EVIDENCE-INDEX.md](EVIDENCE-INDEX.md).

---

## 1. The architecture under test

The source paper (Deshmukh 2025, Sect. 3.6) describes its feature vector as "a rich blend
of network and text features":

```
  5 metadata numbers ──► untrained SAGEConv(5→128) ──► 128 dims ─┐
                                                                 ├──► concat 896 ──► linear SVM
  tweets ──► BERT / DistilBERT, mean-pooled ────────► 768 dims ──┘
```

Two things about this are stated in the paper and are easy to miss:

- the SAGEConv layer is **never trained** (Sect. 3.5: "training epochs and optimization
  tasks are not required due to the lack of a prediction head");
- the classifier is a **linear** SVM, justified in Sect. 3.7 by "relying on the inherent
  separability in this transformed space".

Both turn out to matter more than the architecture does.

## 2. The ablation

`results_ablation.csv` — every block evaluated alone and in combination:

| feature set | dim | accuracy | ±SD | F1 |
|---|---|---|---|---|
| raw 5 metadata features | 5 | 0.8976 | 0.0116 | 0.9236 |
| effective 10 dims `[x ‖ mean N(v)]` | 10 | 0.9262 | 0.0110 | 0.9441 |
| GraphSage[128] | 128 | 0.9359 | 0.0063 | 0.9511 |
| BERT[768] only | 768 | 0.9730 | 0.0039 | 0.9784 |
| raw 5 + BERT[768] | 773 | 0.9764 | 0.0035 | 0.9811 |
| GraphSage[128] + BERT[768] | 896 | 0.9779 | 0.0054 | 0.9823 |

**The marginal contributions:**

| adding | to | gain | vs fold σ 0.0063 |
|---|---|---|---|
| GraphSage[128] | BERT[768] | **+0.0049** | inside noise |
| GraphSage[128] | raw 5 + BERT[768] | **+0.0015** | inside noise |
| BERT[768] | GraphSage[128] | +0.0372 | 5.9 σ |
| raw 5 metadata | BERT[768] | +0.0034 | inside noise |

So the *graph* half is what adds nothing to the text half. Stated the other way round,
the text half adds 3.7 points to the graph half. They are not symmetric — but they are
**substitutes**, because of §3.

## 3. The result that makes them substitutes

`results_trained.csv`:

| arm | accuracy | ±SD | F1 |
|---|---|---|---|
| untrained SAGEConv[128] + **linear SVM** (the paper) | 0.9359 | 0.0063 | 0.9511 |
| untrained SAGEConv[128] + **MLP head** | **0.9775** | 0.0051 | 0.9822 |
| **trained** SAGEConv[128] + head | 0.9739 | 0.0049 | 0.9795 |

**0.9775 with no text at all**, against 0.9779 for the full 896-dimensional pipeline —
a difference of 0.0004 against a fold σ of 0.0054.

Five metadata counts (tweet count, followers, following, listed, account age), their
neighbourhood means, and a small MLP, reproduce the entire method.

**Training the graph layer is worth −0.0036**, i.e. nothing. The binding constraint is
the **linear kernel**: replacing it is worth **+0.0416**, roughly ten times what any
architectural choice in the paper is worth.

## 4. Why the 128-dimensional embedding is 10 numbers

With PyTorch Geometric's defaults an untrained `SAGEConv(5,128)` computes

```
h_v = W_l · mean_{u ∈ N(v)} x_u  +  b  +  W_r · x_v
```

which is an **affine map of the 10-vector** `[x_v ‖ mean N(v)]`. Three consequences, each
asserted in `botsage/checks.py` and recorded in `checks.json`:

| claim | measured |
|---|---|
| the layer is exactly affine in 10 numbers | reconstructed from the weight matrix to **< 1e-05** |
| its 128 columns have rank ≤ 10 | rank **10**, σ₁₀/σ₁₁ = **6.8 × 10⁶** |
| a linear SVM gains nothing from the projection | 128-dim **0.880** vs 10 raw dims **0.880** |

**118 of the 128 "network feature" dimensions are linearly dependent on the other 10.**

### The apparent counterexample

At the default `C = 1` the 128-dim embedding scores **+0.0096** over the 10 numbers it is
computed from, which looks like a refutation. It is not, and the reason is worth being
precise about because it is easy to get backwards.

**L2 regularisation is not invariant under a change of basis.** The random projection
spreads 10 informative directions across 128 coordinates; per-coordinate standardisation
then rescales them, which preconditions the SVM's penalty differently. If the information
is genuinely identical, weakening the penalty must make the gap vanish.

`results_regularization.csv`:

| C | 10 dims | GraphSage[128] | Δ |
|---|---|---|---|
| 0.001 | 0.8249 | 0.8581 | +0.0332 |
| 0.01 | 0.8738 | 0.8885 | +0.0147 |
| 0.1 | 0.9055 | 0.9195 | +0.0140 |
| 1.0 | 0.9262 | 0.9359 | +0.0096 |
| 10 | 0.9408 | 0.9432 | +0.0025 |
| 100 | 0.9442 | 0.9445 | **+0.0004** |

**Monotone to zero.** The gain is regularisation geometry, not information.

### A second, independent confirmation

Since the layer is untrained, its random seed *is* the model. Varying only that seed over
10 seeds (`results_seeds.csv`):

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| acc | 0.9359 | 0.9377 | 0.9370 | 0.9364 | 0.9366 | 0.9368 | 0.9377 | 0.9370 | 0.9368 | 0.9369 |

**Spread 0.0019**, against a fold σ of 0.0063. A random basis change cannot alter what a
linear model can express, and the measurement agrees.

## 5. Why the text branch is weaker than it looks, too

`botsage/text.py` reproduces the paper's Listing A.1: mean over token vectors of the last
hidden state per tweet, then mean over the user's tweets. That is a lot of averaging — a
user's 768-vector ends up close to the corpus mean, and the *between-user* variance a
classifier needs is small.

The cost is visible in the model comparison (`results_replicate.csv`):

| variant | dim | accuracy | ±SD | wall time |
|---|---|---|---|---|
| GraphSage + SVM | 128 | 0.9359 | 0.0063 | 0.4 s |
| GraphSage + **BERT** | 896 | 0.9789 | 0.0020 | 17.3 s |
| GraphSage + **DistilBERT** | 896 | 0.9779 | 0.0054 | 19.5 s |

BERT over DistilBERT is worth **+0.0009** against a fold σ of 0.0054 — inside noise. The
choice of language model does not matter on this benchmark either.

## 6. What this does and does not license

**Licensed:**

- On Cresci-2015 the graph branch adds **+0.0049** to the text branch and **+0.0015** to
  raw-metadata-plus-text. Both are inside fold noise.
- The 128-dimensional embedding contains exactly the information in 10 numbers, and this
  is provable from the layer's closed form, not merely observed.
- Training the graph layer is worth nothing; the linear kernel costs 4.2 points.
- The choice between BERT and DistilBERT is inside noise.

**Not licensed:**

- *"Text embeddings are useless for bot detection."* BERT alone (0.9730) beats the graph
  branch alone (0.9359) by 3.7 points. The finding is redundancy on this benchmark, not
  uselessness.
- *"GraphSAGE does not work."* This paper's GraphSAGE is untrained, one layer, and fed a
  graph in which 63.8% of labelled users have no featured neighbours
  ([04-PROTOCOL-AND-FLOORS.md](04-PROTOCOL-AND-FLOORS.md) §3). None of those is a
  property of GraphSAGE.
- Any claim about a *different* corpus. Everything here is Cresci-2015 under 5-fold CV.

## 7. The consequence for anyone building on this

Cresci-2015 under 5-fold CV **saturates at 0.978 with 0.006 fold noise**. Between the
raw-metadata floor (0.8976) and the ceiling there are 8.0 points, and 4.2 of them are bought
by replacing a linear classifier.

A new feature evaluated on this benchmark therefore has roughly **0.002 of headroom
against 0.006 of noise**. It will appear to work or not to work at random. That is the
single most important consequence of F1 and F2, and it is why the follow-on project
([05-WHY-A-NEW-PROTOCOL.md](05-WHY-A-NEW-PROTOCOL.md)) makes cross-dataset transfer its
primary claim rather than within-corpus accuracy.

## 8. Where to see it

- `01-info-propagation/bot-detection-paper/replication.ipynb` — the narrative version of
  every table above
- `01-info-propagation/bot-detection-paper/results/figures/exp_ablation.png`,
  `exp_trained_vs_untrained.png`, `singular_spectrum.png` (the rank-10 cliff),
  `exp_seed_sensitivity.png`
- `docs/DISCREPANCIES_BOTSAGE.md` §§1, 3, 4 — the implementation reasoning
