# Discrepancies — Lakzaei et al. (2024), *Disinformation detection using graph neural networks: a survey*

*Artificial Intelligence Review* **57**:52. https://doi.org/10.1007/s10462-024-10702-9

Companion to `docs/DISCREPANCIES.md` (which covers the entropia replication).
Read this before changing anything in `disinfo/layers.py`, `disinfo/survey_data.py`
or `disinfo/data.py`.

This paper is a **survey**. It reports no experiments, releases no code, and its
six figures are diagrams rather than data plots. So the items below are of three
kinds: equations printed differently from the work they cite, statements the
survey makes that its own tables contradict, and decisions this replication had
to make because a survey does not specify a method.

---

## 1. Eq. 2 is not the GCN of Welling & Kipf (2016)

The survey prints, attributing it to Kipf & Welling:

```
h_v^l = σ( W^{l-1} · Σ_{u ∈ N(v) ∪ {v}} h_u^{l-1} / |N(v)| )
```

Two departures from the cited paper:

1. **Normalisation.** Kipf & Welling use the symmetric `D̃^{-1/2} Ã D̃^{-1/2}`.
   Eq. 2 divides by the receiving node's degree alone — a random-walk style
   normalisation, not theirs.
2. **Off-by-one.** The sum runs over `N(v) ∪ {v}`, which is `|N(v)| + 1` terms,
   but the denominator is `|N(v)|`. So Eq. 2 is not an average: on a `k`-regular
   graph with constant features it scales the signal by `(k+1)/k` per layer,
   compounding to `((k+1)/k)^L`.

Also **an isolated node divides by zero**, which is not hypothetical — kNN
similarity graphs at a high threshold and PHEME threads with no replies both
produce them.

**Resolution.** `GCNLayer` implements both. `normalization="symmetric"` is the
default because it is what every method in Tables 1–2 that says "GCN" actually
ran. `normalization="paper"` reproduces Eq. 2 literally, with the degree clamped
at 1 so isolated nodes pass through instead of producing NaN.

`checks.check_gcn_normalization` pins the difference: on a 2-regular ring with
constant input, Eq. 2 returns **1.50** and Kipf & Welling return **1.00**.

## 2. Eq. 4's attention has no softmax

The survey prints `α_vu = σ(a(W h_v, W h_u))`, citing Velickovic et al. (2017) —
who normalise with a **softmax over N(v)**. Without it the coefficients do not
form a distribution, so the aggregated message grows with degree and hub nodes
dominate for being hubs rather than for being relevant.

**Resolution.** `GATLayer` applies the softmax and LeakyReLU(0.2) of the original.
`softmax=False` reproduces the printed form, kept only so the notebook can show
what it does. `checks.check_attention_is_a_distribution` asserts the
coefficients sum to 1 per node.

Numerically, `scatter_softmax` subtracts the per-node max before exponentiating;
raw logits on high-degree nodes reach magnitudes where `exp` overflows float32.

## 3. Eq. 8 prints `mean`, the prose says "maximum pooling"

Sect. 3.1 names the three GraphSAGE aggregators "average pooling, maximum
pooling, and long short-term memory", but Eq. 8 prints

```
AGG = mean(MLP(h_u^{l-1}), ∀u ∈ N(v))
```

Hamilton et al. (2017) use **max**. The prose and the equation disagree, and the
equation disagrees with the cited source.

**Resolution.** `SAGELayer(aggregator="pool")` takes `pool_reduce`, defaulting to
`"mean"` to follow the printed equation; `"max"` gives Hamilton et al.

## 4. Eq. 6 omits GraphSAGE's neighbour sampling

Sect. 3.1's prose is right — "GraphSAGE utilizes the sampled neighborhood... 
randomly selecting a subset of neighbors" — but Eq. 6 aggregates over all of
`N(v)`. Sampling is the only thing that distinguishes GraphSAGE from a
concatenating GCN at these graph sizes.

**Resolution.** `fanout=None` (full neighbourhood, exact) is the default because
these graphs are small; `fanout=k` enables sampling. It is implemented as
per-edge Bernoulli keeping with probability `fanout/deg(v)`, which is unbiased
and stays vectorised, rather than an exact top-`k` truncation.

## 5. Eq. 9 (LSTM aggregation) is order-dependent and the survey does not say so

An LSTM over a neighbourhood is not permutation-invariant, so `h_v` depends on
the order neighbours are visited — meaning it is not, strictly, a GNN layer.
Hamilton et al. handle this by applying the LSTM to a **random permutation** of
the neighbours.

**Resolution.** `SAGELayer(aggregator="lstm", shuffle=True)` permutes during
training. `checks.check_permutation_equivariance` deliberately **excludes** the
LSTM aggregator and covers the other six layer variants; this is the one place
where the property genuinely does not hold, and pretending otherwise would hide
a real limitation of Eq. 9.

## 6. GATv2 (Eq. 5) — the survey's own argument, confirmed

Here the survey is correct and worth recording as a positive. Sect. 3.1 argues
that in Eq. 4 the composed linear maps `a` and `W` collapse into one, making the
attention "a monotonic function of the neighbors of a node (rather than the node
itself)".

`checks.check_gat_v2_is_dynamic` confirms it on Brody et al.'s construction (a
shared sender set, so both receivers rank the same four neighbours): **GAT gives
every receiver an identical ranking (1 distinct ordering); GATv2 gives 2.**

---

## 7. Sect. 7 contradicts Tables 1–2 on multiclass accuracy

Sect. 7, "Multiclass classification":

> only a limited number of researchers have dedicated their efforts to
> multiclass classification and as a result, in this setting existing algorithms
> suffer from relatively low accuracy rates, **typically below 50%**.

The survey's own Tables 1–2 contain **36 accuracies on multiclass datasets**
(LIAR, PHEME, Twitter15, Twitter16 — all 4- or 6-class per Table 3). **One** is
below 0.5. The median is **0.881**.

This is the only one of the six checked claims that fails
(`survey_data.verify_claims`, reproduced by `scripts/run_disinfo.py`). The other
five hold:

| Claim (Sect. 5.3.2) | Verdict | Evidence from Tables 1–2 |
|---|---|---|
| First GNN work in 2019 | supported | earliest Year = 2019 (4 methods) |
| GCN and GAT most used | supported | GCN 20, GAT 15, GraphSage 4, GGNN 1 |
| Propagation graph is the majority | supported | 19/34 (56%) |
| Textual dominates; comments/semantic/temporal under-used | supported | Textual 32/34; Comments 5, Temporal 5, Semantic 2 |
| Mostly supervised | supported | 31/34 |
| Multiclass accuracy below 50% (Sect. 7) | **contradicted** | 1/36 below 0.5, median 0.881 |

The most likely reading is that Sect. 7 is describing **LIAR specifically** — the
one 6-class dataset, where Table 1's Hu et al. row does report 0.492 — and
over-generalises to all multiclass work.

## 8. Table 2's LIAR figure is irreconcilable with Table 1's

Table 1 reports Hu et al. (2019) on LIAR at **ACC 0.492**. Table 2 reports Cui et
al. (2023) on LIAR at **ACC 0.868**. Sect. 7 says multiclass accuracy is below
50%.

0.868 on 6-class LIAR would be far beyond the published state of the art (which
sits near 0.27–0.45); it is an entirely ordinary figure for **binary** LIAR.
Almost certainly the two rows describe different tasks and the survey does not
say so.

**Resolution.** `DisinfoDataset.binarised()` and `suite_liar_granularity` run
6-class and binary LIAR side by side, with and without the leaky credit-history
columns, so the size of the gap is measured rather than assumed.

## 9. Table 1's "Approach" column contradicts Sect. 4.2's own definitions

Ren et al. (2020) and Huang et al. (2020) are labelled **Content-based** while
building heterogeneous graphs out of user relations and source-tweet
propagation — which Sect. 4.2 defines as *context*. Transcribed as printed;
`survey_data.TRANSCRIPTION_NOTES` records it.

## 10. Smaller transcription issues

- **Fig. 4** spells a leaf "Commnet". Corrected to "Comment" in
  `taxonomy.FIG4_FEATURES`.
- **Year column vs citation year** disagree for Autef et al. (2020 cited /
  2019 in the table) and Bai et al. (2021 / 2020). Both are kept, as
  `cite_year` and `year`.
- **Table 1** prints "FakeNewsNe", truncated by the PDF column. Read as
  FakeNewsNet, matching the Table 2 row for the same authors.
- **Tables 1–2 mix metrics** (ACC, Macro-F1, AUC) in one Performance column and
  report **no standard deviations and no split protocol**. Cross-row comparison
  is therefore indicative only; every plot built from them says so.

---

## 11. Datasets: Table 3 verified, with one mismatch

`scripts/get_disinfo_data.sh` fetches four of the ten. Item counts reproduce
Table 3 **exactly**:

| Dataset | Table 3 | Downloaded |
|---|---|---|
| LIAR | 12,836 | 12,836 ✓ |
| PHEME | 6,425 | 6,425 ✓ |
| Twitter15 | 1,490 | 1,490 ✓ |
| Twitter16 | 818 | 818 ✓ |

**LIAR needs `csv.QUOTE_NONE`.** The released TSV is full of unbalanced double
quotes; the default csv dialect silently swallows **47 of the 12,836 rows**,
which is exactly the kind of quiet 0.4% loss that never shows up as an error.

**CED is not Table 3's "Sina Weibo".** Table 3's Weibo row cites Ma et al. (2016)
and gives 4,664; the CED corpus we load holds 3,387 (1,538 rumour + 1,849
non-rumour). Table 2 lists Weibo and CED as separate datasets for Xu et al.,
consistent with them being different corpora. `verify_table3` therefore does not
check CED against the Weibo row.

**CED reply text is empty.** Only the original microblog carries text; every
repost record has `"text": ""`. Likewise only the source author's profile is
released, so `_parse_ced_cascade` broadcasts it to every node. CED is a
**structure-only** benchmark here, and its numbers should be read that way.

**Twitter15/16 reply text is withheld** by the corpus README (Twitter terms of
service). Only source-tweet text exists — which is why every Table 1–2 method on
these datasets is source-text-only or structure-only.

## 12. What a survey cannot specify, and what we chose

`config.INFERRED_PARAMETERS` is the authoritative list. The items that most
affect comparability with Tables 1–2:

- **Split protocol.** *No surveyed method states its split.* This is the single
  largest source of incomparability. `suite_pheme_split` measures it directly by
  running PHEME under a random split and under leave-events-out.
- **Text encoder.** Surveyed methods use GloVe, BERT, RoBERTa or TF-IDF. We use
  TF-IDF + SVD: no download, and it works for Chinese (CED) as well as English.
  Expect this to cost accuracy against transformer-based rows.
- **Vocabulary fitting.** Fitted on training rows only. Fitting TF-IDF on all
  rows leaks test-set term statistics and is a common silent error.
- **LIAR credit-history columns leak the label** — each count is computed over
  the speaker's whole record including the statement being classified. Excluded
  by default (`use_credit_history=False`); papers reporting LIAR accuracy rarely
  say whether they used them.
- **No variance in the source.** Tables 1–2 are point estimates. We report mean
  ± s.d. over seeds, so any comparison must acknowledge the paper has no
  counterpart to our error bar.

## 13. What is deliberately not implemented

- **Visual features** — no images are redistributed with any of the four corpora.
- **Semantic features** — need an external knowledge graph (Wang et al.'s KMGCN).
- **Stance / comment features** — reply text is withheld by Twitter15/16 and CED.
- **Schema-level attention** for heterogeneous graphs (AA-HGNN, MFAN). Sect. 5.3.2
  describes it but **no equation in the survey defines it**. `heterogeneous_graph`
  builds the typed graph and returns `node_type`, but the layers are
  type-agnostic.
- **GGNN**, used by Cui et al. (Table 2) and Zhiyuan et al. — the survey names it
  without giving its update rule.

These absences are why this replication cannot reach the hybrid methods that
Tables 1–2 report at the top of the range.
