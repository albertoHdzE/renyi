# renyi

Research on Rényi entropy and algorithmic information theory.

## Contents

| Path | What it is |
|---|---|
| `dtwre/` | Reusable implementation of the DTWRE method (see below) |
| `disinfo/` | Replication of the GNN disinformation-detection survey (see below) |
| `botsage/` | Replication of the GraphSage+BERT bot detector (see below) |
| `scripts/` | Data fetching, experiment runners, notebook builders |
| `docs/DISCREPANCIES.md` | Every ambiguity found in the DTWRE paper, and the choice made |
| `docs/DISCREPANCIES_SURVEY.md` | The same for the disinformation survey |
| `docs/DISCREPANCIES_BOTSAGE.md` | The same for the bot-detection project |
| `01-info-propagation/` | Research projects on information propagation |
| `RenyiPapers/`, `FirstCollection/`, `Causation/` | Reference library (one copy of each paper) |

## `dtwre` — replication of Tong et al. (2025)

Independent reimplementation of:

> Tong, Q.; Xu, X.; Zhang, J.; Xu, H. *Public Opinion Propagation Prediction Model
> Based on Dynamic Time-Weighted Rényi Entropy and Graph Neural Network.*
> **Entropy** 2025, 27, 516. <https://doi.org/10.3390/e27050516>

The paper has **no official code release**, so everything is built from the
published equations, Algorithm 1 and hyperparameter tables.

**The method.** Split a temporal network into time windows; compute a Rényi entropy
per node per window (Local Node Entropy, Eq. 1); decay those entropies
exponentially into the past so recent structure counts more (DTWRE, Eq. 2–4);
concatenate with Node2Vec embeddings; feed to a GraphSAGE encoder with an MLP
decoder that predicts which node pairs connect next.

### Quick start

```bash
# 1. data (CollegeMsg from SNAP + Chinese rumour cascades)
bash scripts/get_data.sh

# 2. environment
python3 -m venv 01-info-propagation/entropia-paper/.venv
01-info-propagation/entropia-paper/.venv/bin/pip install \
    torch numpy scipy networkx scikit-learn matplotlib pandas tqdm jupyter ipykernel
01-info-propagation/entropia-paper/.venv/bin/python -m ipykernel install \
    --user --name dtwre-entropia --display-name "Python (DTWRE entropia-paper)"

# 3. reproduce every table and figure (~11 min)
01-info-propagation/entropia-paper/.venv/bin/python scripts/run_all.py --quiet --weibo

# 4. paired significance test over 10 seeds (~20 min)
01-info-propagation/entropia-paper/.venv/bin/python scripts/significance.py --seeds 10 --quiet
```

### Library use

```python
from dtwre import Config, load_dataset, run_experiment

tg  = load_dataset("collegemsg")
res = run_experiment(tg, Config(alpha=0.6, lam=1.2), method="dtwre")
print(res["test"]["auc"])
```

### The notebook

`01-info-propagation/entropia-paper/replication.ipynb` is a didactic,
step-by-step walkthrough: it derives each equation, verifies it against a property
it must satisfy, then calls this package to reproduce the paper's results. All
scientific code lives in `dtwre/`; the notebook only orchestrates and explains.

Regenerate it with `python scripts/build_notebook.py`.

### Module map

| Module | Contents |
|---|---|
| `config.py` | Hyperparameters, sweep grids, published results for comparison |
| `data.py` | CollegeMsg + Weibo CED loaders, snapshots, splits, negative sampling |
| `entropy.py` | Eq. 1–4: Rényi entropy, LNE, global entropy, time weighting, DTWRE |
| `embeddings.py` | Node2Vec: biased random walks + Skip-Gram (pure PyTorch) |
| `features.py` | Feature fusion for the five compared methods |
| `models.py` | GraphSAGE encoder + MLP link predictor (Eq. 5–7) |
| `metrics.py` | AUC, precision, recall, F1, accuracy (Eq. 8–11) |
| `pipeline.py` | Split preparation and the train/evaluate loop |
| `experiments.py` | Drivers for every table and figure, with embedding caching |
| `plots.py` | Figures 5–11 |

### Replication outcome

**Partially replicated**, split cleanly by a 10-seed paired analysis
(`scripts/significance.py`):

| Claim | Result |
|---|---|
| DTWRE > Node2Vec | +0.051 AUC, 10/10 seeds, p = 0.002 ✅ |
| DTWRE > PageRank | +0.019 AUC, 8/10 seeds, p = 0.037 ✅ |
| DTWRE > node degree | +0.017 AUC, 8/10 seeds, p = 0.049 ✅ |
| DTWRE > **static** Rényi entropy | +0.004 AUC, 6/10 seeds, p = 0.63 ❌ |

So the **entropy** half of the argument holds: Rényi-entropy node features beat
degree, PageRank and embeddings with paired significance. The **time-weighting**
half — the paper's actual novelty, isolated by the static-entropy ablation — is not
distinguishable from noise in this setup. That is not a refutation (the test lacks
power at ~280 test pairs), but it is not a confirmation either.

Also not reproduced: absolute AUCs run ~0.11–0.19 below published; DTWRE is best on
1 of 5 metrics at a fixed 0.5 threshold rather than all 5 (it is conservative, and
regains the F1 lead under per-method thresholds); and neither the α = 0.6 nor the
λ = 1.2 optimum appears — α is flat within noise, and smaller λ is monotonically
better here. The train/test construction is under-specified in the paper, and
the paper is internally inconsistent about its own headline number (DTWRE AUC is
reported as 0.9742, 0.966 and 0.9680 in three places under nominally identical
settings, with no seeds or error bars anywhere).

Four implementation decisions determined whether this worked at all, and none is
recoverable from the paper alone — see `docs/DISCREPANCIES.md`.

---

## `disinfo` — replication of Lakzaei et al. (2024)

> Lakzaei, B.; Haghir Chehreghani, M.; Bagheri, A. *Disinformation detection
> using graph neural networks: a survey.* **Artificial Intelligence Review**
> (2024) 57:52. <https://doi.org/10.1007/s10462-024-10702-9>

This one is a **survey**: no experiments, no code release, and all six figures
are diagrams rather than data plots. So replication means three things instead
of rerunning a script.

**1. Reproduce Figs. 1–6 from encoded data.** Each taxonomy is a structure in
`disinfo/taxonomy.py` and is *rendered*, not traced. Fig. 3's computation tree is
unrolled from the adjacency, so it doubles as a check on the message-passing
semantics of Eq. 1.

**2. Test the survey's prose against its own tables.** Tables 1–2 are a
meta-analysis of 34 papers; Sect. 5.3.2 draws conclusions from them and plots
none. `verify_claims()` evaluates each as a predicate — **5 of 6 hold**.

**3. Run the framework it describes.** Eqs. 2–10 implemented from the printed
equations, the three graph constructions of Sect. 5.3, on LIAR, Twitter15/16,
PHEME and CED.

### Quick start

```bash
python3 -m venv 01-info-propagation/desinformation-paper/.venv
01-info-propagation/desinformation-paper/.venv/bin/pip install \
    torch numpy scipy networkx scikit-learn matplotlib pandas tqdm jupyter ipykernel
01-info-propagation/desinformation-paper/.venv/bin/python -m ipykernel install \
    --user --name disinfo-venv --display-name "Python (desinformation-paper)"

bash scripts/get_disinfo_data.sh                    # ~370 MB

P=01-info-propagation/desinformation-paper/.venv/bin/python
$P scripts/run_disinfo.py --quiet                   # figures + meta-analysis (~30 s)
$P scripts/run_disinfo.py --quiet --experiments     # + the live GNN runs (~1.8 h)
$P scripts/build_disinfo_notebook.py                # regenerate the notebook
```

`01-info-propagation/desinformation-paper/replication.ipynb` is the didactic
walkthrough (99 cells, 19 figures). It is a build artefact — edit the builder.

### What replication found

**Table 3 is exactly right.** All four obtainable datasets match their printed
sizes to the item (LIAR 12,836; PHEME 6,425; Twitter15 1,490; Twitter16 818).

**Five of six prose claims hold; one is contradicted by the paper's own tables.**
Sect. 7 says multiclass accuracy is "typically below 50%". Of the 36 accuracies
Tables 1–2 report on multiclass datasets, **1 is below 0.5** and the median is
**0.881**. The claim appears to generalise a fact about LIAR to all multiclass
work.

**Three printed equations differ from the works they cite.** Eq. 2 is not Kipf &
Welling's GCN and divides |N(v)|+1 terms by |N(v)|; Eq. 4 omits GAT's softmax;
Eq. 8 prints `mean` where the prose says "maximum pooling" and the source uses
`max`. Implementing from the survey alone would give three subtly wrong layers.

**Our runs land inside the published range on 3 of 5 datasets:**

| Dataset | Ours | Published range (n) |
|---|---|---|
| Twitter15 | 0.770 (GAT/propagation) | 0.690–0.946 (14) ✅ |
| Twitter16 | 0.770 (GCN/propagation) | 0.750–0.968 (14) ✅ |
| PHEME | 0.783 (GCN/similarity) | 0.694–0.887 (6) ✅ |
| CED | 0.851 (GCN/propagation) | 0.882 (1) — 3 pts below |
| LIAR (6-class) | 0.243 | 0.492–0.868 (2) — see below |

Three findings of our own:

- **Protocol beats method on PHEME.** Switching from a random split to
  leave-events-out costs 0.114 accuracy and inflates the seed s.d. **fifteen-fold**
  (0.009 → 0.135). The entire published PHEME spread is 0.193, and no paper
  states which protocol it used.
- **On Twitter15, text is nearly everything.** Lexical features alone score
  0.768 against 0.754 for all features; removing them collapses to 0.370.
  Propagation and temporal features are redundant with text here — and the
  survey's case for context features is about *adversarial robustness*, which a
  single-split accuracy table cannot measure.
- **The most expressive layer is the worst.** GIN loses by 0.10–0.14. Its sum
  aggregation makes the readout scale with cascade size (1 to 965 nodes here).

**The LIAR anomaly.** Table 1 reports 0.492, Table 2 reports 0.868, Sect. 7 says
"below 50%". We measure 0.236 on 6-class and 0.604 on binary — confirming the two
rows are different tasks, though binarisation alone does not reach 0.868.

Full detail, including what is deliberately *not* implemented and why, is in
`docs/DISCREPANCIES_SURVEY.md`.

---

## `botsage` — replication of Deshmukh (2025)

> Deshmukh, A. *Bot Detection in Social Media using GraphSage and BERT.*
> Master's Project 1465, San José State University, Spring 2025.
> <https://doi.org/10.31979/etd.wb6h-3yd6>

DistilBERT over a user's tweets (768-d) ‖ an **untrained** GraphSAGE layer over a
user graph (128-d) → 896 dimensions → linear SVM, 5-fold CV. Reported **98.68%**
on Cresci-15 and **74.62%** on TwiBot-22.

### Quick start

```bash
python3 -m venv 01-info-propagation/bot-detection-paper/.venv
01-info-propagation/bot-detection-paper/.venv/bin/pip install \
    torch numpy scipy networkx scikit-learn matplotlib pandas tqdm \
    jupyter ipykernel transformers
01-info-propagation/bot-detection-paper/.venv/bin/python -m ipykernel install \
    --user --name botsage-venv --display-name "Python (bot-detection-paper)"

bash scripts/get_bot_data.sh                     # ~2.5 GB

P=01-info-propagation/bot-detection-paper/.venv/bin/python
$P scripts/prepare_bot_embeddings.py --model distilbert-base-uncased  # ~28 min
$P scripts/prepare_bot_embeddings.py --model bert-base-uncased        # ~50 min
$P scripts/run_botsage.py --quiet --experiments
$P scripts/build_botsage_notebook.py
```

Everything paper-specific — environment, notebook, results, figures — lives in
`01-info-propagation/bot-detection-paper/`; the reusable code is `botsage/`.

### The replication succeeds

| Method | Ours | Paper | Δ |
|---|---|---|---|
| GraphSage+BERT | **97.89** | 98.68 | −0.79 |
| GraphSage+DistilBERT | **97.79** | 98.56 | −0.77 |

Within 0.8 points on both rows, with the paper's BERT > DistilBERT ordering
preserved, despite differing in tokenizer batching, stop-word list, one
substituted feature, and the SVM's unreported `C`.

### What it shows once it does

**1. The GraphSAGE layer is never trained** (Sect. 3.5, explicitly), and an
untrained `SAGEConv(5,128)` is an affine map of `[x_v ‖ mean N(v)]`. Its 128
columns have **rank exactly 10** (σ₁₀/σ₁₁ = 6.8 × 10⁶). The apparent +0.0096
advantage over those ten numbers is a regularisation-geometry artefact: it decays
monotonically to **+0.0004** as the SVM penalty weakens. Varying only the random
seed moves accuracy by 0.002 — a change of basis is invisible to a linear model.

**2. The binding constraint is the linear kernel, not the untrained layer.**

| Arm | Accuracy |
|---|---|
| untrained layer + linear SVM (the paper) | 0.9359 |
| untrained layer + **MLP head** | **0.9775** |
| **trained** layer + MLP head | 0.9739 |

Training the layer is worth nothing; replacing the linear classifier is worth
**+4.2 points** — and reaches 0.9775 **using no text at all**, matching the full
896-d pipeline's 0.9779.

**3. Every accuracy in Table 5 is below the majority baseline.** TwiBot-22 is
139,943 bot / 860,057 human, so predicting "human" scores **86.01%**. All eight
published rows fall below it, including BotRGCN's 79.66%. Worse, the protocols
differ: the paper uses 5-fold CV over the corpus (floor 86.01%) while the
baselines beside it use the official test split (floor 70.56%, since the split is
**not** stratified).

**4. Table 4 has the same problem, and it is measurable.** On Cresci-15's own
split we score **0.886** against **0.978** under 5-fold CV — a **9-point** gap,
because the official split holds out systematically quieter accounts
(test-membership is predictable from the five features at **AUC 0.79**). That is
larger than the entire spread between the methods in Table 4.

**5. On Cresci-15 the graph is nearly vacuous.** Only **0.12%** of its 7M edges
join two users that have any features, so the neighbourhood term is **exactly
zero for 3,381 of 5,301** labelled users.

Also: Sect. 2.3.2 calls Cresci-15 "almost a 50/50 split"; it is 63.2/36.8, which
is why F1 exceeds accuracy in every row of Table 4.

Full detail in `docs/DISCREPANCIES_BOTSAGE.md`.
