# Where the paper is ambiguous, and what this replication does

Tong, Q.; Xu, X.; Zhang, J.; Xu, H. *Public Opinion Propagation Prediction Model
Based on Dynamic Time-Weighted Rényi Entropy and Graph Neural Network.*
**Entropy** 2025, 27, 516. <https://doi.org/10.3390/e27050516>

There is **no official code release**. The Data Availability Statement points only
to the two datasets (SNAP and the ChineseRumorDataset index). Everything below is
therefore an implementation decision I had to make, recorded so the numbers in the
notebook can be judged fairly.

Each entry states what the paper says, why it is underdetermined, and the choice
taken. Anything marked **blocking** materially changes results.

---

## 1. Eq. 1 is not a probability distribution under Algorithm 1 — *blocking*

Equation 1 defines the Local Node Entropy as

    H_a(v,t) = 1/(1-a) * log( sum_{u in N(v,t)} p_u^a(t) )

Algorithm 1 defines the weights globally:

    p(v) <- Node_degree(v) / Sum(Node_degree(v_i))

normalised over **all** nodes. The sum in Eq. 1 runs only over the neighbours of
`v`, so `sum_{u in N(v,t)} p_u < 1` for every non-trivial graph. Two consequences:

- The quantity is not a Rényi entropy of any distribution.
- At `a -> 1` it **diverges**: the limit `1/(1-a) * log(S)` with `S != 1` has a
  non-vanishing numerator. Yet Figure 6 reports a finite AUC at `α = 1`, so the
  authors' code cannot have evaluated the formula as literally written.

**Choice.** `prob_normalisation="local"` (default) renormalises the weights within
each neighbourhood, making Eq. 1 a genuine Rényi entropy for all `α` and recovering
the Shannon entropy exactly at `α = 1`. The literal global variant is available as
`prob_normalisation="global"` for comparison.

**Verified:** uniform neighbourhoods give `log(n)` for every `α`; the value at
`α = 1` matches `-Σ p log p` to 10 decimal places; entropy is monotonically
non-increasing in `α`.

## 2. Units of `t - t_k` in Eq. 4 — *blocking*

Equation 4 is `w(t - t_k) = exp(-λ (t - t_k))` and Section 4.2.2 sweeps
`λ ∈ [0.1, 2]`. Timestamps are unix seconds and windows are 604,800 s, so with raw
seconds `exp(-1.2 × 604800)` underflows to exactly `0.0` and DTWRE is identically
zero for every snapshot — the entire mechanism vanishes.

**Choice.** `t - t_k` is the difference in **time-step indices** (1, 2, 3, …), the
only reading under which the published `λ` range is meaningful.

## 3. DTWRE is a per-snapshot scalar, yet used as a node feature

Eq. 2 sums over nodes and Eq. 3 sums over time, so `H_a^time(G,t)` is one number
per time step. Algorithm 1 line 3 nonetheless concatenates it into a per-node
feature vector, where it is **constant across nodes** and can only shift the
decoder bias — it carries no discriminative signal.

**Choice.** The feature block keeps the paper's global scalar (`dtwre_global`) for
fidelity, and adds the node-level analogue that the method's rationale actually
describes: `lne_time_weighted(v) = Σ_k w(T - t_k) · H_a(v, t_k)`, the same
exponential decay applied per node. Feature columns are named so the notebook can
ablate them.

## 4. Train/test protocol: what counts as a positive — *blocking*

Section 3.1 says positives are "connected node pairs (u, v) extracted from the
actual network" and splits 80/10/10 "using 80% of the total duration and 90% of
the total duration as the demarcation points" — i.e. cut points on the **timeline**,
not on edge count.

Two things are left open, and both move the headline numbers a lot:

- **Are test positives new pairs, or any interaction?** CollegeMsg is bursty and
  front-loaded: 56,964 of 59,835 edges fall in the message-passing period, and the
  test window holds only 230 distinct pairs, of which 168 are genuinely new. Scoring
  repeat pairs is much easier, since the encoder has already propagated over them.
- **How are training positives held out?** Not described.

**Choice.** Positives are pairs occurring in the window (paper-literal), and the
training period is cut again at `mp_frac = 0.90`: earlier edges build the graph,
later edges supervise it. Without this disjointness the model is trained to
recognise *existing* adjacency and scores **AUC 0.36 — below chance — on unseen
future links**. The stricter "new pairs only" variant is reported alongside.

## 5. Internal inconsistency in the reported numbers

The paper's own DTWRE result differs across sections under nominally identical
settings:

| Source | Setting | AUC |
|---|---|---|
| Table 1 | DTWRE, CollegeMsg | 0.9742 |
| Figure 6 | `α = 0.6` (the stated optimum) | 0.966 |
| Table 2 | 604,800 s window (the stated optimum) | 0.9680 |

All three should describe the same configuration. No seed, repetition count or
error bars are reported anywhere, so it is impossible to tell whether the spread
is run-to-run variance or different protocols.

This is more than a presentational quibble. Our 10-seed paired analysis
(`scripts/significance.py`) measures a seed-to-seed standard deviation of
**~0.02 AUC** — roughly five times the 0.004 margin by which DTWRE beats the
static-entropy ablation. A single-run protocol cannot separate those, so the
paper's own design could not have determined whether its central novelty (the time
weighting) contributes 0.004 or 0.04. This replication therefore reports seeds,
error bands on every sweep figure, and paired Wilcoxon tests.

## 6. Figure 8 axis label contradicts its text

The x-axis reads "Positive-to-Negative Sample Ratio", but the text describes the
value 2 as "an excess of negative samples". A positive-to-negative ratio of 2 would
mean *twice as many positives*. The plotted quantity must be `n_neg / n_pos`.

**Choice.** `neg_ratio = n_neg / n_pos`, matching the text rather than the label.

## 7. Unreported hyperparameters

Not stated anywhere; defaults used, all listed in `config.INFERRED_PARAMETERS`:

| Item | Value | Basis |
|---|---|---|
| Node2Vec `walk_length`, `num_walks`, `window`, `p`, `q` | 80, 10, 10, 1, 1 | Grover & Leskovec defaults |
| GraphSAGE depth / hidden / dropout | 2, 64, 0.2 | standard mean-aggregator SAGE |
| Adam `lr`, `weight_decay` | tuned on validation | paper says only "refined via cross-validation" |
| Random seed | 42 | not reported |
| MLP predictor form | Hadamard product → 3-layer MLP | "an MLP" is all that is specified; Hadamard keeps the score symmetric, as undirected link prediction requires |

## 8. Eq. 5 omits the self term

`h_u^(k) = σ(W_k · Aggregate({h_v^(k-1)}))` aggregates only over neighbours.
Canonical GraphSAGE concatenates the node's own previous representation before the
linear map.

**Choice.** `concat_self=True` (canonical). Set `False` for the literal equation.

## 9. Weibo dataset is under-specified

Section 3.1 describes "Chinese rumor data, including repost and comment
information, scraped from the Sina Weibo misinformation reporting platform", and
the Data Availability Statement links `github.com/yeren66/ChineseRumorDataset` —
which is an **index of eight different corpora**, not a dataset. Only the CED
subset carries repost/comment structure, so that is what is used here
(`thunlp/Chinese_Rumor_Dataset`, `CED_Dataset`). Which subset the authors used,
how many cascades, and the resulting graph size are not reported, so Weibo numbers
are **not** directly comparable to the paper's Figures 10–11.

Two further adjustments were forced by the corpus we can actually obtain:

- **Window length.** Section 3.1 selects a 1-hour window because Weibo interactions
  arrive "1–N times per minute". But CED aggregates cascades spanning **1,100 days**,
  so a 1-hour window implies ~26,400 overwhelmingly empty snapshots — computationally
  infeasible and statistically vacuous. We use 30 days, giving 21 snapshots,
  comparable to CollegeMsg's 20. The authors' subset was evidently a much shorter
  window of real time than the full CED corpus.
- **Split rule.** Duration-based splitting is degenerate here: it leaves **0 validation
  and 4 test positives**. We split by edge count for Weibo (6,607 / 2,876 / 2,870
  positives).

Reconstruction detail: CED stores `parent` as a *message* id (`mid`), not a user id,
so the propagation graph is built by resolving `mid -> uid` within each cascade;
records with an empty `parent` attach to the cascade root, whose uid is encoded in
the filename (`<label>_<root_mid>_<root_uid>.json`). Two of 1,540 cascade files have
mixed encodings and fail to parse; they are skipped.

## 10. Not reproducible by construction

- **Figures 1–4** are hand-drawn schematics of the architecture and workflow, not
  plots of data. They are explained in the notebook, not regenerated.
- **Table 3** is a symbolic complexity comparison, not measured timings. The
  notebook reproduces the table and adds measured wall-clock times separately.
