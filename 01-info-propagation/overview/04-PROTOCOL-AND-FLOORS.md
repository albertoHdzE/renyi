# 04 — Protocol, baselines and the degenerate graph

Deep dive on **F5**. The question this answers: *how much of the variation in published
results in this area comes from the method, and how much from choices the papers treat as
incidental?*

Answer: the incidental choices dominate, by a factor of several.

---

## 1. Protocol — 9.2 points on the same pipeline

Cresci-2015, identical features, identical classifier, only the evaluation protocol
differing. Source: `results_protocol.csv`

| protocol | accuracy | F1 | majority baseline |
|---|---|---|---|
| 5-fold stratified CV (what the paper reports) | 0.9779 | 0.9823 | 0.6321 |
| the corpus's own official train/test split | **0.8860** | 0.9036 | 0.6318 |

**A 9.2-point gap.** For scale: the entire published spread between the competing methods
in the source paper's own comparison table is 77.08 → 98.68, and this single unreported
choice accounts for nearly half of it.

**It is not class imbalance.** The majority baseline is 0.6321 under CV and 0.6318 under
the official split — the balance is essentially identical.

**The split is simply not random.** From `docs/DISCREPANCIES_BOTSAGE.md` §5, the official
test users are systematically quieter: median tweet count 18 vs 42, followers 8 vs 18,
account age 694 vs 921 days. A classifier predicts **split membership** from the five
features alone at **AUC 0.79**.

So the two protocols are measuring different things. Random CV interpolates within one
distribution; the official split extrapolates to a quieter population. A method that
looks strong under one can be 9 points weaker under the other, and the source paper's
claim to "outperform all other models on the Cresci-15 dataset" is not supported by a
like-for-like comparison, because its own rows use CV and the baselines it cites use the
official split.

## 2. Majority baselines — the floor that several published tables do not clear

### TwiBot-22

Computed directly from the released `label.csv` (Zenodo 7012904). Source:
`twibot22_baselines.json`

| split | n | bot | bot % | majority baseline |
|---|---|---|---|---|
| corpus | 1,000,000 | 139,943 | 13.99 | **0.8601** |
| train | 700,000 | 54,586 | 7.80 | 0.9220 |
| val | 200,000 | 55,913 | 27.96 | 0.7204 |
| test | 100,000 | 29,444 | 29.44 | **0.7056** |

The source paper evaluates by 5-fold CV over the corpus, so **0.8601** is the number its
accuracies must clear. Its best row is 74.62 and the strongest baseline it cites is 79.66.
**All eight rows in its comparison table are below 86.01** — a classifier that outputs
"human" unconditionally beats every published number in the table on accuracy.

**And the two columns are not the same task.** The cited baselines come from the
TwiBot-22 leaderboard, computed on the **official test split**, whose floor is 0.7056.
The paper's own rows come from CV over the whole corpus, floor 0.8601. Two tasks whose
class balances differ by a factor of two are placed side by side without comment.

This does not make the methods worthless — on a 29.4%-positive test set, 74.62% is above
the 70.56% floor, and F1 is the more informative metric throughout. It means the
**accuracy column cannot support the comparisons drawn from it.**

### LIAR

The same pattern in the survey replication. Source:
`results/disinfo/results_gnn_comparison.csv`, 3 seeds, similarity graph:

| architecture | accuracy | majority | Δ | AUC |
|---|---|---|---|---|
| GCN | 0.2364 | 0.2056 | +0.0308 | 0.5739 |
| GAT | 0.2378 | 0.2056 | +0.0322 | 0.5895 |
| GATv2 | 0.2343 | 0.2056 | +0.0287 | 0.5875 |
| GraphSAGE | 0.2427 | 0.2056 | +0.0371 | 0.5930 |
| GIN | 0.2238 | 0.2056 | +0.0182 | 0.5695 |

Every architecture lands 2–4 points above a 6-class majority baseline, with AUC 0.57–0.59.
The architecture ranking here is not measuring architecture quality; it is measuring
almost nothing.

### Cresci-2015 is not "almost 50/50"

The source paper (Sect. 2.3.2) describes Cresci-2015 as having "almost a 50/50 split
between real users and bot accounts, this resulted in a well balanced dataset".

The released labels give **3,351 bot / 1,950 human = 63.2% / 36.8%**. The majority
baseline is **0.6321**, not 0.50.

This is not a nitpick. It explains a pattern the paper never comments on — **F1 exceeds
accuracy in every row of its results table**, which is what happens when F1 is computed
on the *majority* class.

## 3. The Cresci-2015 graph is 99.88% unusable

Source: `cresci_graph_degeneracy.json`

| quantity | value |
|---|---|
| user–user edges | 6,994,858 |
| edges joining two users that **both** have metadata | **8,550** |
| fraction | **0.12%** |
| labelled users isolated in that subgraph | **3,381 of 5,301 (63.8%)** |

Cresci-2015 ships 5,301 users with metadata, but its `follow`/`friend` relations reference
**1,292,763 distinct user ids** — the graph is mostly neighbours that appear nowhere else
in the corpus and therefore carry **no features at all**.

**For 63.8% of labelled users the neighbour mean is exactly zero**, so the graph layer
reduces to `W_r · x_v + b`: a random linear map of their own five features, with no
network contribution whatsoever. For the remaining third the neighbour mean is diluted by
several hundred zero vectors.

Measured both ways (`results_graph_scope.csv`):

| edge set | n edges | isolated labelled | median degree | accuracy | ±SD |
|---|---|---|---|---|---|
| all (the literal reading) | 6,994,858 | 0 | 317 | 0.9359 | 0.0063 |
| labelled-only | 8,550 | 3,381 | 0 | 0.9293 | 0.0063 |

A difference of **0.0066**, one fold σ. Whichever reading is taken, the graph branch is
doing almost nothing — which is consistent with F1, where it added +0.0049 to the text
branch.

**Consequence.** Any conclusion of the form "GNNs help / do not help for bot detection"
drawn from Cresci-2015 is a conclusion about a graph that is 99.88% dangling. This is a
property of the corpus conversion, not of graph neural networks.

## 4. The feature-ablation version of the same story

Third paper, different corpus, different architecture, same shape of result. Source:
`results/disinfo/results_feature_ablation.csv` — Twitter15, GCN, propagation graph,
3 seeds:

| feature set | accuracy | ±SD | macro-F1 |
|---|---|---|---|
| all features | 0.7539 | 0.0312 | 0.7552 |
| **without lexical** | **0.3702** | 0.0331 | 0.3596 |
| without syntactic | 0.7528 | 0.0312 | 0.7541 |
| without profile | 0.7539 | 0.0312 | 0.7552 |
| without propagation | 0.7517 | 0.0293 | 0.7525 |
| without temporal | 0.7562 | 0.0273 | 0.7583 |
| **lexical only** | **0.7685** | 0.0497 | 0.7704 |

- removing **lexical** features costs **38 points**;
- removing **anything else** costs nothing — *"without profile" is identical to "all
  features" to four decimal places*;
- **lexical-only scores above the full model** (0.7685 vs 0.7539).

The propagation graph — the structure the survey's whole taxonomy is organised around —
is worth **+0.0022**, well inside a ±0.03 error bar.

## 5. One place a published prose claim is contradicted by its own tables

The survey's Sect. 7 states that in the multiclass setting "existing algorithms suffer
from relatively low accuracy rates, typically below 50%".

Transcribing every accuracy in its own Tables 1–4 and filtering to the multiclass corpora
(LIAR, PHEME, Twitter15, Twitter16): **1 of 36** is below 0.5, and the median is
**0.881** (`results/disinfo/claim_verification.csv`).

Five of the six checkable prose claims in that survey hold. This one does not, and it is
contradicted by data the survey itself tabulates.

## 6. What this does and does not license

**Licensed:**

- Protocol choice is worth **9.2 points** on an otherwise identical pipeline, and is not
  reported.
- The TwiBot-22 accuracy column compares two different tasks and all eight rows sit below
  the relevant majority baseline.
- Cresci-2015's graph is 99.88% dangling and 63.8% of labelled users have no featured
  neighbours.
- On Twitter15, one feature family carries the result and the graph structure contributes
  inside noise.

**Not licensed:**

- *"These methods do not work."* They do clear their floors on the tasks where the floor
  is correctly identified — the point is that the floors and protocols are often not
  stated, so the published comparisons do not isolate the method.
- *"Graph structure never helps."* §3 shows a corpus where the graph is unusable and §4 a
  corpus where it is measured at +0.0022. Neither establishes a general claim.

## 7. Where to see it

- `01-info-propagation/bot-detection-paper/replication.ipynb` and
  `01-info-propagation/desinformation-paper/replication.ipynb`
- `01-info-propagation/bot-detection-paper/results/figures/` — `twibot22_splits.png`,
  `graph_degeneracy.png`, `table5_vs_baseline.png`, `table4_vs_baseline.png`
- `results/disinfo/figures/exp_feature_ablation.png`, `exp_vs_literature.png`
- `docs/DISCREPANCIES_BOTSAGE.md` §§2, 3, 5, 6 and `docs/DISCREPANCIES_SURVEY.md`
