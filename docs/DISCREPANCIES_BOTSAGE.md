# Discrepancies — Deshmukh (2025), *Bot Detection in Social Media using GraphSage and BERT*

SJSU Master's Project 1465. https://doi.org/10.31979/etd.wb6h-3yd6

Companion to `docs/DISCREPANCIES.md` (DTWRE) and `docs/DISCREPANCIES_SURVEY.md`.
Read this before changing `botsage/sage.py`, `botsage/data.py` or
`botsage/pipeline.py`.

The paper has no code release beyond one appendix listing, and is unusually
specific about some hyperparameters while silent about others. What follows is
split into three kinds of item: **findings** (things the replication established
that change how the results should be read), **ambiguities** (where the text
admits more than one implementation), and **data substitutions**.

---

# Findings

## 1. The GraphSAGE layer is never trained, and cannot add information

Section 3.5, in full:

> GraphSage is only used with the intention of generating embeddings for nodes.
> Hence, **training epochs and optimization tasks are not required due to the
> lack of a prediction head**... With the model in evaluation mode, embeddings
> are generated in a single forward pass.

So the "GraphSage embeddings" come from a `SAGEConv(5, 128)` at its **random
initialisation**. With PyTorch Geometric's defaults that layer computes

```
h_v = W_l · mean_{u ∈ N(v)} x_u + b + W_r · x_v
```

which is an **affine map of the 10-vector `[x_v ‖ mean of N(v)]`**. Three
consequences, each verified in `botsage/checks.py`:

| Claim | Check | Result |
|---|---|---|
| The layer is exactly affine in 10 numbers | `check_untrained_sage_is_linear` | reconstructed from `weight_matrix()` to 1e-4 |
| Its 128 columns have rank ≤ 10 | `check_embedding_rank_is_bounded` | **rank 10**, σ₁₀/σ₁₁ = 6.8 × 10⁶ |
| A linear SVM gains nothing from the projection | `check_linear_svm_gains_nothing_from_projection` | equal to 3 decimal places |

**118 of the 128 "network feature" dimensions are linearly dependent on the
other 10.** Section 3.6 describes the 896-vector as "a rich blend of network and
text features"; it is 768 text dimensions plus at most 10 usable network ones.

### The one place this needed more care

At the default `C=1` the 128-dimensional embedding scores **+0.0096** accuracy
over the 10 numbers it is computed from — which looks like a counterexample. It
is not, and the reason is worth stating because it is easy to get wrong:

**L2 regularisation is not invariant under a change of basis.** The random
projection spreads 10 informative directions across 128 coordinates and
per-coordinate standardisation then rescales them, which preconditions the SVM's
penalty differently. Weaken the penalty and the gap must vanish if — and only
if — the information is identical. It does (`suite_regularization_equivalence`):

| C | 10 dims | GraphSage[128] | Δ |
|---|---|---|---|
| 0.001 | 0.8249 | 0.8581 | +0.0332 |
| 0.01 | 0.8738 | 0.8885 | +0.0147 |
| 0.1 | 0.9055 | 0.9195 | +0.0140 |
| 1 | 0.9262 | 0.9359 | +0.0096 |
| 10 | 0.9408 | 0.9432 | +0.0025 |
| **100** | **0.9442** | **0.9445** | **+0.0004** |

Monotone to zero. The gain is regularisation geometry, not information.

A second, independent confirmation: varying **only** the untrained layer's
random seed over 10 seeds moves accuracy by **0.0019**, against a fold-to-fold
standard deviation of 0.0065. A random basis change cannot alter what a linear
model can express, so stability here is exactly what the analysis predicts.

## 2. Every accuracy in Table 5 is below the majority-class baseline

Verified from the released `label.csv` (Zenodo record 7012904):

```
TwiBot-22:  139,943 bot  /  860,057 human   →  majority baseline 86.01%
```

The paper evaluates by **5-fold cross-validation over the dataset** (Sects. 3.7,
4.1.2), so 86.01% is the figure its accuracies must clear. None do — and neither
do any of the baselines it cites:

| Method (Table 5) | Accuracy | vs 86.01% baseline |
|---|---|---|
| GraphSage+DistilBERT (this paper) | 74.62 | −11.39 |
| BotRGCN | 79.66 | −6.35 |
| RGT | 76.47 | −9.54 |
| HGT | 74.91 | −11.10 |
| BGSRD | 71.88 | −14.13 |
| GraphSage+SVM (this paper) | 67.21 | −18.80 |
| SVM | 49.30 | −36.71 |
| GCN | 47.72 | −38.29 |

**8 of 8 rows.** A classifier that outputs "human" unconditionally beats every
published number in the table on accuracy.

### It is worse than a simple oversight: the protocols do not match

TwiBot-22's official split is **not stratified**:

| Split | n | bot % | majority baseline |
|---|---|---|---|
| corpus | 1,000,000 | 13.99 | **86.01%** |
| train | 700,000 | 7.80 | 92.20% |
| val | 200,000 | 27.96 | 72.04% |
| **test** | 100,000 | 29.44 | **70.56%** |

The baselines in Table 5 come from the TwiBot-22 leaderboard, computed on the
**official test split** (baseline 70.56%). The paper's own rows come from
**5-fold CV over the whole corpus** (baseline 86.01%). The two columns therefore
describe tasks with class balances differing by a factor of two, and are placed
side by side without comment.

This does not mean the methods are worthless — on a 29.4%-positive test set,
74.62% is above the 70.56% floor, and F1 is the more informative metric
throughout. It means the **accuracy column of Table 5 cannot support the
comparisons drawn from it**, including the paper's own claim to "outperform some
state-of-the-art models".

## 3. On Cresci-15 the graph branch is close to vacuous

Cresci-15 has 5,301 users with metadata. The `follow`/`friend` relations
reference **1,292,763 distinct user ids**, so the graph is mostly made of
neighbours that appear nowhere else in the corpus and therefore have **no
features at all**. Measured:

- user–user edges: **6,994,858**
- edges joining two users that have metadata: **8,550 (0.12%)**
- labelled users whose neighbour-mean is **exactly zero**: **3,381 of 5,301 (64%)**
- median share of a labelled user's neighbours that have features: **0.0000**

So for two thirds of the labelled users the layer reduces to `W_r · x_v + b` — a
random linear map of *their own five features*, with no network contribution
whatsoever. For the remaining third the neighbour mean is diluted by a few
hundred zero vectors.

Combined with §1, the "GraphSage" stage on Cresci-15 is, for most users, a
random reparameterisation of five numbers.

## 4. The binding constraint is the linear kernel, not the untrained layer

Sections 3.5 and 3.7 are both suspect, but not equally. Decomposing the graph
branch into three arms — separating "the layer is untrained" from "the
classifier is linear" — gives:

| Arm | Accuracy | F1 |
|---|---|---|
| untrained SAGEConv[128] + **linear SVM** (the paper) | 0.9359 | 0.9511 |
| untrained SAGEConv[128] + **MLP head** | **0.9775** | 0.9822 |
| **trained** SAGEConv[128] + head | 0.9739 | 0.9795 |

Training the layer is worth **nothing** (arm 3 is below arm 2, inside the ±0.005
fold noise). Replacing the linear classifier is worth **+4.2 points**.

Section 3.7's justification does not hold up:

> The models performance was optimized by opting to not use a kernel trick, but
> rather relying on the inherent separability in this transformed space.

Per §1 the "transformed space" is an affine image of the input, so it cannot
make anything linearly separable that was not already. The five metadata
features are not linearly separable, and the linear kernel leaves ~4 points
unclaimed.

Note also where arm 2 lands: **0.9775 using no text at all**, against 0.9779 for
the full 896-dimensional pipeline. Five metadata numbers plus their neighbourhood
means, with a small MLP, match the entire method.

**Method.** Arm 3 exploits the fact that a *single* SAGEConv's aggregation is
parameter-free, so a trained 1-layer SAGEConv plus head is exactly an MLP on the
fixed 10-dimensional `[x_v ‖ mean N(v)]`. This is an identity, not an
approximation, and it turns 3,000 full-graph passes into seconds of work.

## 5. Table 4's protocol differs from its baselines' — worth ~9 points

The paper evaluates by 5-fold cross-validation; the Cresci-15 baselines it cites
(BotRGCN, RGT, BIC) use the corpus's official split. On our pipeline:

| Protocol | Accuracy | F1 |
|---|---|---|
| 5-fold CV (the paper) | 0.9779 | 0.9823 |
| official train/test split | **0.8860** | 0.9036 |

**9.2 points.** Larger than the entire spread between the methods in Table 4
(77.08 → 98.68).

The cause is not class imbalance — the balance is 63.2% bot in every split.
The official split is simply **not random**: its test users are systematically
less active (median tweet count 18 vs 42, followers 8 vs 18, account age 694 vs
921 days), and a classifier predicts test-set membership from the five features
alone at **AUC 0.79**. Random CV interpolates within one distribution; the
official split extrapolates to quieter accounts.

So the claim to "outperform all other models on the Cresci-15 dataset" is not
supported by a like-for-like comparison.

## 6. Cresci-15 is not "almost a 50/50 split"

Section 2.3.2 states:

> The dataset contains only about 5000 accounts with almost a 50/50 split
> between real users and bot accounts, this resulted in a well balanced dataset.

The released labels give **3,351 bot / 1,950 human = 63.2% / 36.8%**. This is
not a nitpick: 63.2% is the accuracy floor for Cresci-15, and it also explains a
pattern the paper never comments on — **F1 exceeds accuracy in every row of
Table 4**, which is what happens when F1 is computed on the *majority* class.

---

# Ambiguities

## 7. Feature scaling is never mentioned, and the method fails without it

The five features are raw counts: tweet count, followers, following, favourites,
account age. On Cresci-15 they span 0 to ~10⁷. The BERT half of the
concatenation has entries of order 0.1.

Concatenating them unscaled and handing the result to a **linear** SVM would let
the count features dominate the margin entirely, making the 768 BERT dimensions
almost invisible. The paper reports a large gain from adding BERT, so some
scaling must have been applied — but none is described.

**Resolution.** `Config.standardize_features=True` z-scores the five features
before the SAGEConv, and `make_classifier` puts a `StandardScaler` inside the
cross-validation pipeline so it is refitted per fold. Both are choices of this
replication.

## 8. The SVM's regularisation constant is never given

Section 3.7 says only that "the regularization term was appropriately adjusted
to avoid overfitting". No value is reported, and no held-out set is described on
which it could have been chosen — the paper's only protocol is the same 5-fold
cross-validation it reports results from. Tuning `C` on that CV and then
reporting that CV's score is optimistically biased.

`Config.svm_C=1.0` (scikit-learn's default) is used throughout. §1's table shows
the choice matters: accuracy on the graph branch alone moves from 0.825 to 0.944
across the range.

## 9. Two readings of the edge definition

Section 3.1.1 first says to clean the user data by "removing rows where values
in critical fields ... are missing", then says every relation is an edge and
"the type of relation does not matter". Whether an edge to a user who was
*removed* by the cleaning step survives is not stated.

`restrict_graph` implements both. `"all"` (default, the literal reading) keeps
6,994,858 edges; `"labelled"` keeps the 8,550 between users with metadata and
leaves 3,381 users isolated. Neither gives the graph branch much — see §3.

## 10. Account age depends on an unfixed reference date

Section 3.3 derives account age by subtracting the creation date from "the
current date". That is not a fixed quantity, so the feature is literally
irreproducible — a run in 2026 gets values ~2 years larger than the paper's 2024
run. Harmless for a linear model (a constant shift is absorbed by the bias) but
worth pinning down: `CRESCI_REFERENCE_DATE = "2015-01-01"`, the corpus's
collection year.

Also note the paper states the subtraction backwards ("The current date is
subtracted from the user's account creation date"), which would give negative
ages.

## 11. Token pooling is specified only in the appendix

The body of the paper does not say how a tweet's 768-vector is formed. Listing
A.1 does: `torch.mean(tweet_embeddings.squeeze(0), dim=0)` — the mean over
**token** vectors of the last hidden state, not the `[CLS]` vector. Then a
second mean over the user's tweets. `TweetEncoder(pooling="mean")` reproduces
this; `"cls"` is available for comparison.

One deviation, deliberate: Listing A.1 encodes one tweet at a time, so its mean
over the token axis has no padding to worry about. We batch for speed
(2.7M tweets), so the mean is taken over **real tokens only** via the attention
mask. Without masking, a tweet's embedding would depend on the longest tweet in
its batch — a batching artefact rather than a property of the text.

---

# Data substitutions

## 12. Neither evaluation dataset is openly downloadable

Cresci-15 requires an application to the Bot Repository; TwiBot-22 requires
emailing its authors from an institutional address. What `scripts/get_bot_data.sh`
fetches instead:

| Paper | Used here | Status |
|---|---|---|
| Cresci-15 | **cresci-2015**, from the TwiBot-22 authors' own conversion to the TwiBot-22 four-file schema | **Complete** — users, tweets, graph, labels, split |
| TwiBot-22 | **twibot-22** `user.json` + `label.csv` + `split.csv` (Zenodo 7012904, open) | **Partial** — no `edge.csv`, no tweets |
| TwiBot-22 | **twibot-20**, preprocessed BotRGCN format | **Substitute** — graph, 5 features, BERT embeddings, labels |

So Table 4 is directly testable and Table 5 is not. The TwiBot-22 analysis in §2
rests only on the released labels, which are exact.

## 13. `favourites_count` is missing from the Cresci-15 conversion

The TwiBot-22 schema exposes `public_metrics` = {followers, following, tweet,
listed}. There is no favourites count, which is the paper's fourth feature.
`listed_count` stands in. The feature *count* (5) and everything downstream are
unaffected, but the fourth column is not the paper's.

## 14. TwiBot-20's five features are not the paper's five

The preprocessed mirror ships BotRGCN's properties — followers, active days,
screen-name length, friends, statuses — already z-scored, so
`standardize_features` is a no-op there. Four of five overlap with the paper's
(screen-name length replaces favourites). Its tweet embeddings are BotRGCN's
pooling over all tweets, not the paper's mean-of-token-means over 15.

---

# Implementation notes

## 15. Memory: never materialise every tweet vector

The obvious implementation of Sect. 3.2 — encode all tweets, then segment-average
per user — costs `2.7M × 768 × 4 B = 8.4 GB` on Cresci-15 and would need ~250 GB
for TwiBot-22's 80M tweets. `user_text_embeddings` accumulates each batch
straight into the per-user running sum, keeping the peak at one batch. This was
found the way such things usually are: the first run reached 12.2 GB resident.

## 16. Why `LinearSVC` rather than `SVC(kernel="linear")`

Identical model, but `SVC` is quadratic in the number of samples and does not
finish on these sizes. `LinearSVC` solves the primal. Sect. 3.7's own rationale
("opting to not use a kernel trick") matches the primal solver.

## 17. PyTorch Geometric is not a dependency

Per the repository's tooling policy. `botsage.sage.SAGEConv` reproduces PyG's
layer including its initialisation (`nn.Linear` defaults: Kaiming uniform with
`a=√5`), which matters here more than usual — since the layer is never trained,
**the initialisation is the model**.
`checks.check_sage_matches_pyg_form` pins the closed form.
