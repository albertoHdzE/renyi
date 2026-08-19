# Evidence index

Every number quoted in this folder, the file it comes from, and the command that
regenerates that file. Verified against the result files on **2026-08-19**.

**Read this first:** `.gitignore` line 237 ignores `results/` at every level, so **none of
the CSV/JSON files below are in git**. A fresh clone has none of them. They are build
artefacts and must be regenerated with the commands in §1.

Paths are relative to the repository root.

---

## 1. How to regenerate everything

Each replication has its own virtualenv, beside the paper it serves.

```bash
# ---- bot-detection (botsage) -> 01-info-propagation/bot-detection-paper/results/
B=01-info-propagation/bot-detection-paper/.venv/bin/python
bash scripts/get_bot_data.sh                                          # ~2.5 GB
$B scripts/prepare_bot_embeddings.py --model distilbert-base-uncased  # ~28 min
$B scripts/prepare_bot_embeddings.py --model bert-base-uncased        # ~50 min
$B scripts/run_botsage.py --quiet                                     # checks + figures (~5 s)
$B scripts/run_botsage.py --quiet --experiments                       # the ablation tables

# ---- entropia (dtwre) -> results/
P=01-info-propagation/entropia-paper/.venv/bin/python
bash scripts/get_data.sh                        # CollegeMsg + CED cascades
$P scripts/run_all.py --quiet --weibo           # all tables/figures (~15 min)
$P scripts/significance.py                      # the 10-seed paired tests

# ---- desinformation (disinfo) -> results/disinfo/
D=01-info-propagation/desinformation-paper/.venv/bin/python
bash scripts/get_disinfo_data.sh                # LIAR, Twitter15/16, PHEME
$D scripts/run_disinfo.py --quiet               # figures + meta-analysis (~30 s)
$D scripts/run_disinfo.py --quiet --experiments # live GNN runs (~1.8 h)
```

Committed and always available: the three `replication.ipynb` notebooks, the three
`docs/DISCREPANCIES*.md`, and the two teaching notebooks.

---

## 2. F1 — the text branch

**File:** `01-info-propagation/bot-detection-paper/results/results_ablation.csv`
**Protocol:** Cresci-2015, 5-fold stratified CV, majority baseline 0.6321448783248443

| variant | dim | accuracy | accuracy_std | f1 |
|---|---|---|---|---|
| raw 5 features | 5 | 0.8975686874255331 | 0.011615484670699 | 0.9235648857094263 |
| effective 10 dims `[x ‖ mean N(v)]` | 10 | 0.9262425977628794 | 0.011007964814346 | 0.9440511922522283 |
| GraphSage[128] | 128 | 0.9358622161364323 | 0.006348975042113 | 0.9510524686394088 |
| BERT[768] only | 768 | 0.9730236693756336 | 0.003926216036507 | 0.9783957516371924 |
| raw 5 + BERT[768] | 773 | 0.9764193622961607 | 0.003531303891298 | 0.9811317194846099 |
| GraphSage[128] + BERT[768] | 896 | 0.9779291519214697 | 0.005358977469398 | 0.9823354827266122 |

**File:** `.../results/results_replicate.csv`

| variant | dim | accuracy | accuracy_std | seconds |
|---|---|---|---|---|
| GraphSage+SVM | 128 | 0.9358622161364323 | 0.006348975042113 | 0.367 |
| GraphSage+BERT | 896 | 0.9788721924848399 | 0.001952170656719 | 17.33 |
| GraphSage+DistilBERT | 896 | 0.9779291519214697 | 0.005358977469398 | 19.54 |

**File:** `.../results/results_trained.csv`

| variant | accuracy | accuracy_std | f1 |
|---|---|---|---|
| untrained SAGEConv[128] + SVM (the paper) | 0.9358622161364323 | 0.006348975042113 | 0.9510524686394088 |
| untrained SAGEConv[128] + MLP head | 0.9774884854089235 | 0.005091014707280 | 0.9822218675170803 |
| trained SAGEConv[128] + head | 0.9739047652920291 | 0.004933452890169 | 0.9795153036136000 |

**Derived in this folder:** graph branch added to BERT = 0.9779 − 0.9730 = **+0.0049**;
added to raw5+BERT = 0.9779 − 0.9764 = **+0.0015**; MLP over linear SVM = 0.9775 − 0.9359
= **+0.0416**; trained over untrained = 0.9739 − 0.9775 = **−0.0036**.

## 3. F2 — rank 10, and the regularisation control

**File:** `.../results/checks.json` — 7 property checks. The three cited:

```
check_untrained_sage_is_linear: exactly affine in the 10 numbers [x_v || mean N(v)]
check_embedding_rank_is_bounded: rank 10 (bound 2*5=10); sigma_10/sigma_11 = 6.8e+06
check_linear_svm_gains_nothing_from_projection: 128-dim 0.880 vs 10 raw dims 0.880
check_sage_matches_pyg_form: SAGEConv == W_l·mean(N(v)) + b + W_r·x_v (err < 1e-05)
check_untrained_output_depends_on_seed: mean |delta| 0.798, rank unchanged (10)
```

**File:** `.../results/results_regularization.csv`

| C | acc_10dim | acc_sage128 | delta |
|---|---|---|---|
| 0.001 | 0.8249403375242295 | 0.8581407714331443 | 0.0332004339089148 |
| 0.01 | 0.8737988369818435 | 0.8885124393149930 | 0.0147136023331496 |
| 0.1 | 0.9054910817491508 | 0.9194508562587804 | 0.0139597745096296 |
| 1.0 | 0.9262425977628794 | 0.9358622161364323 | 0.0096196183735529 |
| 10.0 | 0.9407666316931339 | 0.9432189283872459 | 0.0024522966941120 |
| 100.0 | 0.9441616132875714 | 0.9445391496096598 | 0.0003775363220884 |

**File:** `.../results/results_seeds.csv` — `sage_seed` 0–9, accuracy min 0.9358622161364323
(seed 0), max 0.9377488307577402 (seed 6). **Spread 0.0018866146213079.**

## 4. F3 — the α sweep

**File:** `results/alpha_sweep.json` — CollegeMsg, 3 seeds

| α | auc | auc_std | accuracy | accuracy_std |
|---|---|---|---|---|
| 0.2 | 0.8409060244215071 | 0.0075607555040802 | 0.7934272300469484 | 0.0108845287505819 |
| 0.6 | 0.8348343790055995 | 0.0050061820933320 | 0.7417840375586854 | 0.0423837325945253 |
| 1.0 | 0.8359812453619376 | 0.0062849505214493 | 0.7687793427230046 | 0.0092417932793566 |
| 1.5 | 0.8390845307967348 | 0.0090756801810655 | 0.7500000000000000 | 0.0377051136249038 |
| 2.0 | 0.8329116912905620 | 0.0079738474594151 | 0.7417840375586854 | 0.0418933528112083 |
| 5.0 | 0.8311070633475005 | 0.0059577170191791 | 0.7464788732394366 | 0.0505376763887582 |

**Derived:** max − min = 0.8409060244215071 − 0.8311070633475005 = **0.0097989610740066**.
Argmax **α = 0.2**.

**Published comparison:** `dtwre/config.py`, `PAPER_ALPHA_AUC = {0.2: 0.950, 0.6: 0.966,
1.0: 0.959, 1.5: 0.955, 2.0: 0.952, 5.0: 0.944}`. Spread **0.022**.

**File:** `results/lambda_sweep.json`

| λ | auc | auc_std |
|---|---|---|
| 0.1 | 0.8659515617621264 | 0.0177312305625150 |
| 0.4 | 0.8711461917290696 | 0.0067583629771958 |
| 0.8 | 0.8410072185117722 | 0.0123704228059493 |
| 1.2 | 0.8354078121837686 | 0.0078837275687894 |
| 2.0 | 0.8142751130000674 | 0.0074254599520413 |

**Derived:** spread **0.0568710787290022**. Argmax **λ = 0.4** (paper states 1.2).

## 5. F4 — the significance tests

**File:** `results/significance_auc.json` — 10 seeds (42–51), paired Wilcoxon

Per-method mean AUC over the 10 seeds (computed from `matrix`):

| method | mean | SD (population) |
|---|---|---|
| node_degree | 0.8229659987856709 | 0.0191883 |
| node_pagerank | 0.8213266545233758 | 0.0194789 |
| node2vec | 0.7897035013155231 | 0.0201424 |
| renyi_static | 0.8369156041287189 | 0.0144662 |
| dtwre | 0.8403916211293261 | 0.0179144 |

`paired` block, verbatim:

| vs | mean diff | std diff | wins | wilcoxon p | significant |
|---|---|---|---|---|---|
| Node Degree | 0.0174 | 0.0251 | 8/10 | 0.0488 | true |
| Node PageRank | 0.0191 | 0.0244 | 8/10 | 0.0371 | true |
| Node2vec | 0.0507 | 0.0184 | 10/10 | 0.0020 | true |
| **Rényi Entropy** | **0.0035** | 0.0221 | **6/10** | **0.625** | **false** |

**The 0.0179 noise floor used throughout this folder** is the `dtwre` row SD above.

**Published comparison:** `dtwre/config.py`, `PAPER_TABLE1["dtwre"]["auc"] = 0.9742`,
`PAPER_TABLE1["renyi_static"]["auc"] = 0.9487`. Gap **0.0255**.

**File:** `results/table1_collegemsg.json` — the 3-seed replication of Table 1, for
comparison with the published values.

## 6. F5 — protocol, baselines, graph

**File:** `.../results/results_protocol.csv`

| protocol | accuracy | accuracy_std | f1 | majority_baseline |
|---|---|---|---|---|
| cv | 0.9779291519214697 | 0.005358977469398 | 0.9823354827266122 | 0.6321448783248443 |
| official_split | 0.8859813084112149 | 0.0 | 0.9036334913112164 | 0.6317757009345795 |

**Derived:** gap **0.0919478435102548**.

**File:** `.../results/twibot22_baselines.json`

| split | n | human | bot | bot_frac | majority_baseline |
|---|---|---|---|---|---|
| corpus | 1000000 | 860057 | 139943 | 0.139943 | 0.860057 |
| train | 700000 | 645414 | 54586 | 0.07798 | 0.92202 |
| val | 200000 | 144087 | 55913 | 0.279565 | 0.720435 |
| test | 100000 | 70556 | 29444 | 0.29444 | 0.70556 |

**File:** `.../results/cresci_graph_degeneracy.json`

```json
{"user-user edges": 6994858,
 "edges joining two users with metadata": 8550,
 "fraction of such edges": 0.001222326457520653,
 "labelled users isolated in that subgraph": 3381,
 "labelled users total": 5301}
```

**Derived:** 3381 / 5301 = **0.6378** isolated.

**File:** `.../results/results_graph_scope.csv`

| graph_scope | n_edges | isolated_labelled | median_degree_labelled | accuracy | accuracy_std |
|---|---|---|---|---|---|
| all | 6994858 | 0 | 317.0 | 0.9358622161364323 | 0.006348975042113 |
| labelled | 8550 | 3381 | 0.0 | 0.9292602208667509 | 0.006303481461649 |

**Cresci-2015 class balance:** `data/raw/bot/cresci-2015/label.csv` — 3,351 bot /
1,950 human, majority **0.6321448783248443**.

## 7. Survey replication

**File:** `results/disinfo/results_feature_ablation.csv` — twitter15, gcn, propagation,
stratified, 3 seeds

| variant | accuracy | accuracy_std | macro_f1 |
|---|---|---|---|
| all features | 0.7539149888143176 | 0.031179775983817 | 0.7551946594115867 |
| without lexical | 0.3702460850111857 | 0.033106596391384 | 0.3596154891799271 |
| without syntactic | 0.7527964205816554 | 0.031179775983817 | 0.7541211025155797 |
| without profile | 0.7539149888143176 | 0.031179775983817 | 0.7551946594115867 |
| without propagation | 0.7516778523489932 | 0.029254355325776 | 0.7524694920581899 |
| without temporal | 0.7561521252796420 | 0.027330630252507 | 0.7583068236951919 |
| lexical only | 0.7684563758389261 | 0.049659894587076 | 0.7704040396521806 |

**File:** `results/disinfo/results_gnn_comparison.csv` — LIAR rows, similarity graph,
3 seeds, majority baseline 0.205607476635514

| gnn | accuracy | auc |
|---|---|---|
| gcn | 0.2363707165109034 | 0.5739477613645518 |
| gat | 0.2377985462097612 | 0.5894855739322299 |
| gatv2 | 0.2342938733125649 | 0.5875013627756536 |
| sage | 0.2427310488058152 | 0.5930195628703391 |
| gin | 0.2237798546209761 | 0.5694652218539576 |

**File:** `results/disinfo/claim_verification.csv` — 6 prose claims, 5 supported. The
failure, verbatim:

> `multiclass_below_50`, Sect. 7 — *"…in this setting existing algorithms suffer from
> relatively low accuracy rates, typically below 50%."* → **False**. Evidence: "of 36
> accuracies on multiclass datasets (LIAR, PHEME, Twitter15, Twitter16), 1 are below 0.5;
> median is 0.881".

## 8. Committed artefacts (always present)

| Path | Contents |
|---|---|
| `01-info-propagation/bot-detection-paper/replication.ipynb` | F1, F2, F5 narrative + tables |
| `01-info-propagation/entropia-paper/replication.ipynb` | F3, F4 narrative + sweeps with error bands |
| `01-info-propagation/desinformation-paper/replication.ipynb` | survey replication, ablation, claim checks |
| `01-info-propagation/bot-detection-paper/core-concepts.ipynb` | GCN vs GraphSAGE from scratch, synthetic |
| `01-info-propagation/bot-detection-paper/one-training-step.ipynb` | one training cycle, every matrix printed |
| `docs/DISCREPANCIES_BOTSAGE.md` | 17 items — findings, ambiguities, data substitutions |
| `docs/DISCREPANCIES.md` | 10 items — the entropia paper |
| `docs/DISCREPANCIES_SURVEY.md` | the survey |

All three `replication.ipynb` are **build artefacts** of `scripts/build_*_notebook.py` —
edit the builder, not the notebook.

## 9. Figures

Regenerated by the commands in §1; not committed.

| Figure | Shows |
|---|---|
| `01-info-propagation/bot-detection-paper/results/figures/exp_ablation.png` | F1 |
| `…/singular_spectrum.png` | F2 — the rank-10 cliff |
| `…/exp_trained_vs_untrained.png` | F2 — training buys nothing |
| `…/exp_seed_sensitivity.png` | F2 — 0.0019 across 10 seeds |
| `…/twibot22_splits.png` | F5(b) |
| `…/graph_degeneracy.png` | F5(c) |
| `…/table4_vs_baseline.png`, `table5_vs_baseline.png` | F5(b) |
| `results/figures/figure6_alpha.png` | F3 — α sweep with error bands |
| `results/figures/figure7_lambda.png` | F4 — λ sweep |
| `results/figures/figure5_comparison.png` | F4 — Table 1 replicated vs published |
| `results/disinfo/figures/exp_feature_ablation.png` | §7 |
| `results/disinfo/figures/exp_vs_literature.png` | survey vs published |
