#!/usr/bin/env python
"""Generate the didactic replication notebook for the disinformation survey.

The notebook is a build artefact so it can be regenerated deterministically
(and reviewed as source). Edit this file, never the .ipynb. Run::

    01-info-propagation/desinformation-paper/.venv/bin/python \
        scripts/build_disinfo_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "01-info-propagation" / "desinformation-paper" / "replication.ipynb"

cells: list = []


def md(text: str):
    cells.append({"cell_type": "markdown", "id": f"md{len(cells):03d}",
                  "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str):
    cells.append({"cell_type": "code", "id": f"cd{len(cells):03d}",
                  "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ===========================================================================
# Front matter
# ===========================================================================

md(r"""
# Disinformation detection using graph neural networks: a survey — a working replication

Lakzaei, B.; Haghir Chehreghani, M.; Bagheri, A.
***Artificial Intelligence Review*** (2024) **57**:52.
<https://doi.org/10.1007/s10462-024-10702-9>

---

## Read this first: what it means to "replicate" a survey

This paper is a **survey**. That single fact shapes everything below, so let us
be honest about it up front:

- It reports **no experiments of its own**.
- It has **no code release**, and could not have one — nothing was run.
- Its **six figures are all diagrams**: a framework block diagram, a GNN
  schematic, and four taxonomy trees. There is not a single data plot in the
  paper.
- Its tables are a **meta-analysis**: Tables 1–2 record what 34 *other* papers
  reported; Table 3 characterises 10 datasets; Table 4 shows sample records.

So "reproduce Figure 3" cannot mean "rerun their script". This notebook takes
replication to mean three concrete, checkable things.

### 1. Reproduce the six figures — from data, not by tracing

Every taxonomy is encoded as a **structure** (`disinfo.taxonomy`) and *rendered*
(`disinfo.plots`). The figure is therefore a view of data we can query, not a
picture we copied. Figure 3 goes furthest: its computation tree is **unrolled
from the adjacency matrix**, so if our reading of message passing were wrong,
the picture would visibly disagree with the paper.

### 2. Test the survey's own claims against the survey's own tables

Section 5.3.2 draws four conclusions from Tables 1–2 **in prose**, and plots
none of them. Section 7 makes a fifth. We transcribe the tables, state each
conclusion as a predicate, and evaluate it.

> **Five of the six hold. One is contradicted by the paper's own tables.**
> We will find it in Part 2.

### 3. Actually run the framework the survey describes

Section 3.1 gives five GNN update rules as equations (2–10). Section 5.3 gives
three graph constructions. Figure 1 gives a four-stage pipeline. All of that
*is* specified precisely enough to implement — so we implement it from the
equations and run it on four real datasets.

---

## How this notebook is organised

| Part | What happens | Paper section |
|---|---|---|
| 0 | Setup; the package layout | — |
| 1 | Reproduce Figures 1–6 from encoded taxonomies | Figs. 1–6 |
| 2 | **Meta-analysis**: test the survey's prose against its tables | Tables 1–2, Sects. 5.3.2, 7 |
| 3 | **The equations**: Eqs. 1–10 derived, implemented, property-checked | Sect. 3.1 |
| 4 | The datasets, and verifying Table 3 against reality | Sect. 6, Table 3 |
| 5 | Stage 1 — feature extraction | Sect. 4, Fig. 4 |
| 6 | Stage 2 — graph construction (the survey's novel axis) | Sect. 5.3 |
| 7 | Stages 3–4 — GNN and classification, end to end | Sects. 3, 5.1 |
| 8 | Five experiments testing five specific claims | Sects. 5.3.2, 7 |
| 9 | This replication against the published range | Tables 1–2 |
| 10 | What we learned; the open problems of Section 7 | Sect. 7 |

**Nothing scientific lives in this notebook.** Every computation imports from
the `disinfo` package at the repository root, so the same code is testable
outside Jupyter. The notebook explains and calls; the package computes.

---

## Running it

```bash
# 1. environment (already built if you are reading this from the repo)
python3 -m venv 01-info-propagation/desinformation-paper/.venv
01-info-propagation/desinformation-paper/.venv/bin/pip install \
    torch numpy scipy networkx scikit-learn matplotlib pandas tqdm jupyter ipykernel

# 2. data (~370 MB: LIAR, Twitter15/16, PHEME, and CED reused from the entropia paper)
bash scripts/get_disinfo_data.sh

# 3. everything, headless
01-info-propagation/desinformation-paper/.venv/bin/python \
    scripts/run_disinfo.py --quiet --experiments
```

Parts 0–7 run in a couple of minutes. Part 8 trains models and takes longer; it
is written so you can read the pre-computed results instead of re-running.
""")

# ===========================================================================
md(r"""
---
# Part 0 — Setup

The replication lives in `disinfo/` at the repository root, a sibling of
`dtwre/` (the earlier Rényi-entropy replication). Layering is strict and
one-directional:

```
config, taxonomy, survey_data     no dependencies on the rest
        ↓
layers  →  checks                 Eqs. 2–10 and their property tests
        ↓
data  →  features  →  graphs      Fig. 1 stages 1 and 2
        ↓
models  →  pipeline               Fig. 1 stages 3 and 4
        ↓
experiments  →  plots
```
""")

code(r"""
import sys, warnings
from pathlib import Path

# The notebook lives two levels below the repository root.
ROOT = Path.cwd()
while not (ROOT / "disinfo").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)

import disinfo
from disinfo import Config, plots

plots.setup_style()

print(f"repository root : {ROOT}")
print(f"disinfo version : {disinfo.__version__}")
print(f"torch           : {torch.__version__}")
for k, v in disinfo.PAPER.items():
    print(f"{k:9s}: {v}")
""")

# ===========================================================================
md(r"""
---
# Part 1 — Reproducing Figures 1–6

All six figures are diagrams. We encode each as a data structure and render it.
The benefit is not aesthetic: a structure can be **queried**, so the taxonomy
stops being a picture you squint at and becomes something you can ask questions
of.

## Figure 1 — the general framework

Section 1 breaks disinformation detection with GNNs into four stages. This is
the spine of the whole notebook: Parts 5, 6 and 7 implement stages 1, 2 and 3–4
respectively.
""")

code(r"""
fig = plots.fig1_framework()
plt.show()

for i, stage in enumerate(disinfo.taxonomy.FIG1_STAGES, 1):
    print(f"{i}. {stage['name']:20s} (Sect. {stage['section']})")
    print(f"   {stage['detail']}")
""")

md(r"""
## Figure 2 — types of false information

Section 2 is a glossary, and the distinctions it draws are not pedantic. The
split that matters most is **misinformation vs disinformation**, and it is
*intent*, not truth value:

- **Misinformation** — false, but an honest mistake. *Not intended to deceive.*
- **Disinformation** — false, and **spread deliberately to mislead**.

Note where **satire** sits: under misinformation, because it has no harmful
intention even though it can deceive. And note that **rumour** is *not
necessarily false* — it is merely unverified, and may later be confirmed. That
is why rumour datasets like PHEME and Twitter15/16 carry a four-way label
(`true / false / unverified / non-rumor`) rather than a binary one.
""")

code(r"""
fig = plots.fig2_false_information()
plt.show()

# Because it is a structure, we can print it as an accessible tree with the
# Section 2 definitions attached.
from disinfo.taxonomy import FIG2_FALSE_INFORMATION, FIG2_DEFINITIONS, describe, path_to

print(describe(FIG2_FALSE_INFORMATION))
print("\n--- Section 2 definitions ---")
for term, definition in FIG2_DEFINITIONS.items():
    print(f"\n{term}\n    {definition}")
""")

code(r"""
# Queries a picture cannot answer:
print("Where does 'Satire' sit?        ", " > ".join(path_to(FIG2_FALSE_INFORMATION, "Satire")))
print("Where does 'Clickbait' sit?     ", " > ".join(path_to(FIG2_FALSE_INFORMATION, "Clickbait")))
print("\nSo satire is a kind of MISinformation (no intent to harm),")
print("while clickbait is a kind of DISinformation (deliberate).")
""")

md(r"""
## Figure 3 — the general scheme of a GNN

This figure has two halves. On the left, a five-node graph with the
neighbourhood of node `a` circled. On the right, the **computation tree** for
`a` under a 3-layer GNN.

The right-hand side is the one worth dwelling on, because it is where people's
intuition usually breaks. A 3-layer GNN does not "look at the graph three
times". It builds a *tree of receptive fields*: to compute `a` at layer 3 you
need `a`, `b` and `c` at layer 2; to get `b` at layer 2 you need `b`, `a` and
`d` at layer 1; and so on. The tree grows as $O(\deg^L)$ and nodes reappear at
multiple positions.

**We generate this tree by unrolling the adjacency**, not by transcribing the
paper's picture. So it doubles as a check on our reading of Eq. 1.
""")

code(r"""
fig = plots.fig3_gnn_schematic()
plt.show()

# Verify the unrolling matches the paper's Figure 3 exactly.
from disinfo.taxonomy import FIG3_EXAMPLE

spec = FIG3_EXAMPLE
adj = {n: [] for n in spec["nodes"]}
for a, b in spec["edges"]:
    adj[a].append(b); adj[b].append(a)

levels = [[spec["root"]]]
for _ in range(spec["layers"] - 1):
    levels.append([m for v in levels[-1] for m in [v] + sorted(adj[v])])
levels.reverse()

for i, level in enumerate(levels, 1):
    print(f"Layer {i}: {' '.join(level)}")

assert levels[0] == list("abc") + list("bad") + list("cae"), "unrolling mismatch"
print("\nMatches the paper's Fig. 3 exactly (a b c | b a d | c a e).")
""")

md(r"""
## Figure 4 — types of features

Section 4's taxonomy of features. The top split is the one the whole field turns
on:

- **Content-based** — what the item *says* (text, images, semantics).
- **Context-based** — what the *network does with it* (who posted it, who
  reposted it, when, and what they said about it).

Section 4.1 gives the reason content alone is not enough, and it is an
adversarial one: *"writers of disinformation are aware of these features and
often attempt to mimic the structure and writing style of genuine
information."* Style is a cat-and-mouse game. Propagation structure is much
harder for an author to fake, because it is produced by other people.

The trade-off, stated in Section 5.2.3: context-based methods **cannot detect
early**, because there is no cascade yet. That tension drives the hybrid methods
at the top of Tables 1–2.

*(The published figure spells one leaf "Commnet"; we render the correction.)*
""")

code(r"""
fig = plots.fig4_features()
plt.show()

from disinfo.taxonomy import FIG4_FEATURES, leaves
print(describe(FIG4_FEATURES))
print("\nLeaf feature types:", ", ".join(n.label for n in leaves(FIG4_FEATURES)))
""")

md(r"""
## Figures 5 and 6 — the two categorisations

These are the survey's stated original contributions: *"For the first time, we
categorize disinformation detection methods from two perspectives: features and
algorithms."*

**Figure 5 (by strategy)** splits first into **detection** (find it after the
fact) versus **intervention** (stop the agents that spread it — bad sources, bad
users). Section 5.2.3 is candid about why detection dominates: intervention
"could result in substantial user dissatisfaction" if it misfires. Deleting an
innocent person's account is a worse failure than mislabelling a post.

**Figure 6 (by algorithm)** descends from ML/non-ML down to the GNN leaf, and it
is here that the survey's novel axis appears — GNN methods are split by **how
the graph is built** (similarity / propagation / heterogeneous), not by
architecture. Part 6 tests whether that axis actually predicts performance.

In both figures `[i]-j` means reference *i*, published in year *j*; a trailing
`...` is the authors' own truncation.
""")

code(r"""
fig = plots.fig5_approaches()
plt.show()
""")

code(r"""
fig = plots.fig6_algorithms()
plt.show()

from disinfo.taxonomy import FIG6_ALGORITHMS
print("Path to the GNN leaves of Fig. 6:")
for leaf in ["Similarity Graph", "Propagation Graph", "Heterogeneous Graph"]:
    print("  " + " > ".join(path_to(FIG6_ALGORITHMS, leaf)))
""")

# ===========================================================================
md(r"""
---
# Part 2 — The meta-analysis the survey implies but never draws

Here is the heart of what *can* be replicated in a survey.

Tables 1 and 2 record 34 GNN-based methods across ten columns. Section 5.3.2
then draws four conclusions from them, **in prose, with no plot**:

> - GNN is a novel technique... with its first research being presented in 2019.
> - GCN and GAT stand out as the most widely utilized graph neural networks.
> - The majority of methods employ the propagation graph...
> - The majority of methods rely on textual features... comments, semantic
>   characteristics and temporal aspects have been given less consideration.
> - Most methods are based on supervised learning...

Section 7 adds a fifth:

> - in this setting [multiclass] existing algorithms suffer from relatively low
>   accuracy rates, **typically below 50%**.

We transcribed both tables into `disinfo.survey_data` and stated each claim as a
predicate. Now we check them.
""")

code(r"""
methods = disinfo.methods_table()
print(f"Tables 1-2 transcribed: {len(methods)} methods, "
      f"{len(disinfo.long_results())} (method, dataset) results\n")
methods.head(10)
""")

code(r"""
# Transcription is not neutral. Everything ambiguous in the printed table is
# recorded rather than silently resolved.
for i, note in enumerate(disinfo.TRANSCRIPTION_NOTES, 1):
    print(f"{i}. {note}\n")
""")

md(r"""
## Testing the claims
""")

code(r"""
claims = disinfo.verify_claims()

for _, r in claims.iterrows():
    mark = "SUPPORTED   " if r["supported"] else "CONTRADICTED"
    print(f"[{mark}] {r['claim']}  (Sect. {r['section']})")
    print(f"               {r['evidence']}\n")

print(f"=> {int(claims['supported'].sum())} of {len(claims)} claims supported "
      f"by the survey's own tables.")
""")

code(r"""
fig = plots.plot_claims(claims)
plt.show()
""")

md(r"""
### The four supported claims, plotted

These are the figures Section 5.3.2 describes but does not draw.
""")

code(r"""
from disinfo.survey_data import METHODS
long_df = disinfo.long_results()

fig = plots.plot_gnn_usage(METHODS); plt.show()
fig = plots.plot_graph_type_usage(METHODS); plt.show()
""")

code(r"""
fig = plots.plot_feature_usage(METHODS); plt.show()
fig = plots.plot_year_trend(METHODS); plt.show()
""")

md(r"""
Two things stand out that the prose passes over:

- **Visual features appear in only 2 of 34 methods**, even though Section 7
  lists them as an open problem and Section 4.1 argues images are how authors
  "elicit anger or other emotional responses". The field says images matter and
  then does not use them.
- **The 2023 bar is a partial year** — the survey was accepted 4 January 2024 —
  so the apparent decline after 2022 is a censoring artefact, not a trend.
""")

md(r"""
### The contradicted claim

Section 7 says multiclass accuracy is "typically below 50%". The survey's own
Tables 1–2 contain 36 accuracies on datasets that Table 3 lists as 4- or
6-class. Let us look at every one of them.
""")

code(r"""
multiclass = {d["name"] for d in disinfo.DATASETS if len(d["labels"]) > 2}
mc = long_df[long_df["dataset"].isin(multiclass) & (long_df["metric"] == "ACC")]

print(f"multiclass datasets in Table 3 : {sorted(multiclass)}")
print(f"accuracies reported on them    : {len(mc)}")
print(f"how many are below 0.50        : {(mc['value'] < 0.5).sum()}")
print(f"median                         : {mc['value'].median():.3f}")
print(f"range                          : {mc['value'].min():.3f} - {mc['value'].max():.3f}\n")
print(mc.nsmallest(5, "value")[["ref", "year", "dataset", "value"]].to_string(index=False))
""")

md(r"""
**One of 36.** The median is 0.881.

The single value below 0.5 is Hu et al. (2019) on **LIAR** — the only 6-class
dataset in the collection. The most charitable reading is that Section 7 is
describing LIAR specifically and over-generalises to all multiclass work. It is
an easy slip to make, and it matters, because that sentence is offered as
motivation for an entire open research direction.

We test this directly in **Part 8.4** by running 6-class and binary LIAR
side by side.

### A second internal contradiction

While we are here: Table 1 reports LIAR at **ACC 0.492** (Hu et al.), and Table 2
reports LIAR at **ACC 0.868** (Cui et al.). Both are printed as plain accuracy on
the same named dataset.
""")

code(r"""
print(long_df[long_df["dataset"] == "LIAR"][["ref", "year", "gnn", "metric", "value"]]
      .to_string(index=False))
""")

md(r"""
0.868 on 6-class LIAR would be far past the published state of the art (roughly
0.27–0.45). It is an unremarkable number for **binary** LIAR. Almost certainly
the two rows describe different tasks, and the survey does not say so — which is
exactly the "lack of a standardised benchmark" problem Section 7 complains about,
reproduced inside the survey itself.

### Which datasets carry the field's evidence, and how much they disagree
""")

code(r"""
fig = plots.plot_dataset_usage(long_df); plt.show()
fig = plots.plot_reported_performance(long_df); plt.show()
""")

md(r"""
Look at the **within-dataset spread**. On PHEME, published accuracies run from
0.694 to 0.887 — a range of 0.19. That is far wider than the difference between
any two architectures. Since **no paper in Tables 1–2 states its split protocol
or reports a standard deviation**, that spread cannot be attributed to method.

This is the central methodological finding of the meta-analysis, and Part 8.5
tests one candidate explanation directly.
""")

code(r"""
fig = plots.plot_performance_heatmap(long_df); plt.show()
""")

md(r"""
The blank cells are the point. The field has covered a small fraction of the
(dataset × architecture) grid, and most cells that *are* filled rest on one or
two papers.
""")

# ===========================================================================
md(r"""
---
# Part 3 — The equations of Section 3.1

Now we leave meta-analysis and start building. Section 3.1 is the part of the
survey that is specified precisely enough to implement: five update rules,
Eqs. 2–10.

We implement each **from the printed equation**, then check it against a
property the mathematics must satisfy. The repository has no test suite by
convention — correctness is asserted against properties instead, which catches
the errors that "it ran without crashing" does not.

## Eq. 1 — the message-passing skeleton

$$h_v^{l} = \mathrm{UPDATE}\Big(h_v^{l-1},\ \mathrm{AGG}\big(\{h_u^{l-1} : u \in N(v)\}\big)\Big)$$

Every layer below is a choice of `AGG` and `UPDATE`. Three steps, per the paper:
gather neighbours' vectors, aggregate them, update your own.

**The one constraint that matters:** `AGG` takes a *set*, so it must be
permutation-invariant. If it is not, the layer is reading the node numbering —
an artefact of how the file was written — rather than the graph. We test this
first, and it is the check that catches the most bugs.
""")

code(r"""
from disinfo.layers import (GCNLayer, GATLayer, GATv2Layer, SAGELayer, GINLayer,
                            scatter_sum, scatter_mean, scatter_softmax, degree)

# Graphs are (2, E) edge_index tensors: column (u, v) means "u sends to v",
# so edge_index[1] is always the receiver index used by every scatter.
edge_index = torch.tensor([[0, 1, 1, 2, 3],
                           [1, 0, 2, 1, 1]])
print("edge_index:\n", edge_index)
print("senders  :", edge_index[0].tolist())
print("receivers:", edge_index[1].tolist())
print("degrees  :", degree(edge_index, 4).tolist())
""")

md(r"""
## Eq. 2 — GCN, and a discrepancy

The survey prints, citing Welling & Kipf (2016):

$$h_v^{l} = \sigma\Big( W^{l-1} \sum_{u \in N(v) \cup \{v\}} \frac{h_u^{l-1}}{|N(v)|} \Big)$$

Read it carefully. The sum runs over $N(v) \cup \{v\}$ — that is $|N(v)|+1$
terms — but the denominator is $|N(v)|$. **This is not an average.** On a
$k$-regular graph with constant features it multiplies the signal by
$(k{+}1)/k$ every layer.

And it is not Kipf & Welling's rule either: they normalise symmetrically by
$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$, which divides by
$\sqrt{\deg(u)\deg(v)}$, not by $\deg(v)$ alone.

There is also a division by zero waiting for any isolated node — and isolated
nodes are routine here (a kNN graph at a high threshold, a PHEME thread with no
replies).

We implement **both** and let the notebook show the difference.
""")

code(r"""
from disinfo.checks import check_gcn_normalization
print(check_gcn_normalization())
""")

md(r"""
On a 2-regular ring with constant input, the printed Eq. 2 returns **1.50** and
Kipf & Welling return **1.00** — the $(k{+}1)/k = 3/2$ factor, made concrete.

`normalization="symmetric"` is our default, because it is what every method in
Tables 1–2 that says "GCN" actually ran. `normalization="paper"` reproduces
Eq. 2 literally.

## Eqs. 3–4 — GAT, and a missing softmax

$$h_v^{l} = \Big\|_{k=1}^{K} \sigma\Big(\sum_{u \in N(v)} \alpha_{vu}^{k}\, W^{l-1} h_u^{l-1}\Big),
\qquad \alpha_{vu} = \sigma\big(a(W^{l-1}h_v^{l-1},\, W^{l-1}h_u^{l-1})\big)$$

The idea is right and important: not all neighbours matter equally, so learn a
weight per edge. But **Eq. 4 as printed has no softmax**, while Velickovic et
al. (2017) — cited on the same line — normalise over $N(v)$.

Without normalisation, $\alpha$ is not a distribution, so the aggregated message
**grows with degree**. A node with 300 repliers gets a vector 300× larger than a
node with one. That is a scale artefact masquerading as importance.
""")

code(r"""
# A star plus a chain: degrees 4, 1, 1, 1, 2.
ei = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4, 4],
                   [1, 2, 3, 4, 0, 0, 0, 0, 3]])
logits = torch.randn(ei.size(1), generator=torch.Generator().manual_seed(0)) * 10

alpha = scatter_softmax(logits, ei[1], 5)
print("per-node sums of alpha:", scatter_sum(alpha, ei[1], 5).tolist())

from disinfo.checks import check_attention_is_a_distribution
print(check_attention_is_a_distribution())
""")

md(r"""
Note the logits were scaled by 10 on purpose. Raw attention logits on
high-degree nodes routinely reach magnitudes where `exp` overflows float32, so
`scatter_softmax` subtracts the per-node maximum first — a detail that is
invisible until it silently produces NaNs on your largest cascade.

## Eq. 5 — GATv2, and the survey's best argument

Here the survey is not only correct but genuinely illuminating. Its objection to
Eq. 4 (Sect. 3.1):

> the parameters $W$ and $a$, which are linear transformations, are successively
> multiplied by the embedding vectors... when two linear transformations are
> composed, the result can be represented as an equivalent single linear
> transformation... it makes the attention weights a monotonic function of the
> neighbors of a node (rather than the node itself).

In plain terms: in GAT, the *ranking* of neighbours is the **same for every
receiving node**. The receiver contributes an additive constant, and adding a
constant does not reorder. So GAT's "attention" is really a global popularity
score.

GATv2 (Brody et al. 2021) fixes it by swapping two operations —
$\alpha_{vu} = a \cdot \sigma(W[h_v, h_u])$, nonlinearity **before** the linear
form instead of after.

We can demonstrate this rather than assert it.
""")

code(r"""
from disinfo.checks import check_gat_v2_is_dynamic
print(check_gat_v2_is_dynamic())
""")

md(r"""
Two receivers, the same four senders. **GAT produces one distinct ranking —
identical for both receivers. GATv2 produces two.** That is static versus
dynamic attention, made visible in six lines.

## Eqs. 6–9 — GraphSAGE

$$h_v^{l} = \sigma\big(W^{l-1}\,[\,h_v^{l-1} \,\|\, \mathrm{AGG}(h_u^{l-1},\ \forall u \in N(v))\,]\big)$$

The **concatenation** is the design point: the node's own previous state is kept
separate from its neighbourhood summary rather than averaged into it, so the
layer cannot wash out a node's identity.

Three aggregators are offered — and here the prose and the equation disagree.
Section 3.1 names them "average pooling, maximum pooling, and LSTM", but Eq. 8
prints `mean(MLP(·))`, while Hamilton et al. use `max`. We default to the
printed equation and expose the alternative.

Eq. 9's LSTM aggregator deserves a warning: **an LSTM is order-dependent**, so
it is not permutation-invariant and, strictly, not a GNN layer. Hamilton et al.
paper over this by feeding a random permutation. We do the same, and we exclude
it from the equivariance check — because it genuinely fails, and hiding that
would be dishonest.
""")

code(r"""
x = torch.randn(6, 5)
ring = torch.tensor([[0,1,2,3,4,5, 1,2,3,4,5,0],
                     [1,2,3,4,5,0, 0,1,2,3,4,5]])

for agg in ["mean", "pool", "lstm"]:
    layer = SAGELayer(5, 4, aggregator=agg).eval()
    print(f"SAGE-{agg:5s} output {tuple(layer(x, ring).shape)}")
""")

md(r"""
## Eq. 10 — GIN

$$h_v^{l} = \mathrm{MLP}^{l}\Big((1 + \epsilon^{l})\,h_v^{l-1} + \sum_{u \in N(v)} h_u^{l-1}\Big)$$

The **sum** is the entire point, and it is worth understanding why. Xu et al.
(2018) prove that mean and max aggregation cannot distinguish neighbourhood
*multisets* that differ only in multiplicity — a node with neighbours
$\{x, x\}$ and one with $\{x\}$ have the same mean and the same max.

Sum tells them apart. That injectivity is what makes GIN as discriminative as
the Weisfeiler–Lehman test the survey invokes.
""")

code(r"""
from disinfo.checks import check_gin_injectivity
print(check_gin_injectivity())
""")

md(r"""
## All checks at once

Before we run anything on real data, every property must hold.
""")

code(r"""
results = disinfo.run_all_checks()
print(f"\n{len(results)}/6 property checks passed.")
""")

md(r"""
> **Why this matters.** Each of these caught something real while this
> replication was being written. The permutation check catches sender/receiver
> index swaps — the single most common GNN bug, and one that still trains to a
> plausible-looking accuracy. The isolated-node check catches the Eq. 2 division
> by zero. The GATv2 check initially *failed*, and the fault was the test, not
> the layer: a complete graph gives each receiver a different sender set, so the
> rankings differ trivially. Brody et al.'s construction needs a **shared**
> sender set.
""")

# ===========================================================================
md(r"""
---
# Part 4 — The datasets, and checking Table 3 against reality

Section 6 introduces ten datasets and Table 3 characterises them. Four are still
publicly obtainable and are what we use:

| Dataset | Why it is here |
|---|---|
| **LIAR** | The only 6-class corpus; text-only, so it can *only* support a similarity graph. This is the dataset behind both LIAR anomalies from Part 2. |
| **Twitter15 / Twitter16** | The most-used benchmarks in Tables 1–2. Real propagation trees with timing. |
| **PHEME** | Nine breaking-news events, which makes leave-one-event-out evaluation possible. |
| **CED** | Chinese rumour cascades, reused from the entropia replication. Deep trees (median 327 nodes). |

A survey's most checkable factual claim is its dataset table, so let us check it.
""")

code(r"""
disinfo.datasets_table()
""")

code(r"""
sizes = disinfo.dataset_sizes()
t3 = disinfo.verify_table3(sizes)
print(t3[t3["match"].notna()].to_string(index=False))
""")

md(r"""
**Four for four, exact.** Table 3's sizes are correct.

Getting there required one non-obvious fix, recorded in
`docs/DISCREPANCIES_SURVEY.md`: the released LIAR TSV is full of unbalanced
double quotes, and Python's default `csv` dialect silently swallows **47 of the
12,836 rows**. A 0.4% loss that raises no error — precisely the kind of thing
that never gets noticed.

One dataset does *not* match, and shouldn't: Table 3's "Sina Weibo" row cites Ma
et al. (2016) with 4,664 claims, while the CED corpus holds 3,387. Table 2 lists
Weibo and CED as separate datasets, so these are different corpora and we do not
compare them.
""")

code(r"""
datasets = {}
for name in ["twitter15", "twitter16", "pheme", "ced", "liar"]:
    datasets[name] = disinfo.load_dataset(name)
    print(datasets[name].summary())
    print()
""")

md(r"""
Read the cascade statistics closely — they explain a lot of what follows.

- **Twitter15/16**: median 20 and 18 nodes, mean depth 1.7. These are *shallow*
  and *wide* — a source tweet with a fan of direct replies.
- **PHEME**: median 12 nodes, depth 3.2. Smaller but deeper conversations.
- **CED**: median 327 nodes, depth 5.9. An order of magnitude larger.

A 2-layer GNN sees 2 hops. On a depth-1.7 tree that is the whole cascade; on a
depth-5.9 tree it is not. This is exactly why `add_root_edges` (Bian et al.'s
"root feature enhancement") is on by default.

### Table 4 — what the data actually looks like
""")

code(r"""
disinfo.examples_table()
""")

code(r"""
# The real records, alongside the paper's samples.
tw = datasets["twitter15"]
for item in tw.items[:3]:
    c = item.cascade
    print(f"[{item.label:10s}] {item.text[:80]}")
    print(f"             cascade: {c.size} nodes, depth {c.depth}, "
          f"span {c.times.max():.0f} min\n")
""")

md(r"""
### A caveat that constrains everything downstream

Twitter15/16 ship **only the source tweet's text** — the corpus README cites the
Twitter terms of service. Reply nodes carry structure and timing but no words.
CED is worse: every repost record has an empty `text` field, and only the source
author's profile is released.

So on these corpora, "content" means *the source post only*. This is not a
limitation of our loaders; it is why every Table 1–2 method on these datasets is
either source-text-only or structure-only, and why no purely stance-based method
can be reproduced on them at all.
""")

# ===========================================================================
md(r"""
---
# Part 5 — Stage 1: feature extraction (Section 4)

Now we walk Figure 1 from left to right. Stage 1 turns items into vectors.

We implement the leaves of Figure 4 that the data can support:

| Fig. 4 leaf | Implementation |
|---|---|
| Content / Linguistic / **Lexical** | TF-IDF over unigrams+bigrams → truncated SVD |
| Content / Linguistic / **Syntactic** | punctuation, capitalisation, pronoun rates |
| Context / User / **Profile** | followers, friends, ratio, statuses, verified |
| Context / Network / **Propagation** | root degree, node count, avg degree, depth |
| Context / Network / **Temporal** | span, median delay, reposts/hour, first-hour share |

Not implemented, for lack of data rather than lack of interest: **visual**
(no images redistributed), **semantic** (needs an external knowledge graph),
**stance/comments** (reply text withheld). Those absences are why we cannot
reach the hybrid methods at the top of Tables 1–2 — and they are worth naming,
because a replication that quietly skips them looks better than it is.
""")

code(r"""
cfg = Config(graph="propagation", gnn="gcn")
ds = datasets["twitter15"]

train_idx, val_idx, test_idx = disinfo.make_splits(ds, cfg)
print(f"split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

bundle = disinfo.build_features(ds, cfg, train_idx)
print(bundle.summary())
""")

md(r"""
Two details in `build_features` that are easy to get wrong and expensive to get
wrong:

**1. The vocabulary is fitted on training rows only.** Fitting TF-IDF on all
rows leaks test-set term statistics into the representation. It inflates
accuracy by a point or two and it is a common silent error in this literature.
Note `fit_on=train_idx` being threaded through.

**2. Everything is standardised, also on training rows only.** Without it, the
raw `depth` count sits orders of magnitude above the SVD components and
dominates both the cosine distances of the kNN graph and the first layer's
gradients — a feature wins on units rather than on information.
""")

code(r"""
# What each block actually measures.
from disinfo.features import propagation_features, temporal_features

prop, prop_names = propagation_features(ds)
y = ds.y

print("Propagation features by class (Sect. 4.2's four, plus two):\n")
rows = []
for cls, name in enumerate(ds.label_names):
    rows.append([name] + list(prop[y == cls].mean(axis=0).round(3)))
print(pd.DataFrame(rows, columns=["label"] + prop_names).to_string(index=False))
""")

md(r"""
This is the empirical core of Section 4.2's claim that *"the propagation pattern
of disinformation on social networks differs from that of true information"* —
the class means genuinely separate on structure alone, before any text is read.
""")

code(r"""
temp, temp_names = temporal_features(ds)
rows = []
for cls, name in enumerate(ds.label_names):
    rows.append([name] + list(temp[y == cls].mean(axis=0).round(3)))
print("Temporal features by class:\n")
print(pd.DataFrame(rows, columns=["label"] + temp_names).to_string(index=False))
""")

# ===========================================================================
md(r"""
---
# Part 6 — Stage 2: graph construction (Section 5.3)

**This is the survey's own novel contribution**, so it deserves the most care:

> For the first time, we examine GNNs-based methods in details from the
> perspective of graph type and graph construction approach.

Figure 6 gives three leaves, and the choice between them decides *what kind of
learning problem you have* — a point the survey makes but which is easy to read
past:

| Construction | Graphs | Task | Setting |
|---|---|---|---|
| **Similarity** | one, over all items | **node** classification | can be semi-supervised |
| **Propagation** | one per item | **graph** classification | supervised |
| **Heterogeneous** | one per item, typed nodes | **graph** classification | supervised |

That is the mechanism behind Table 1's "Setting" column: **all three
semi-supervised rows are similarity-graph methods**, and necessarily so. If your
graph joins labelled and unlabelled items, unlabelled nodes still pass messages.
If each item is its own graph, an unlabelled item is simply absent.

## 6.1 Similarity graphs

Two variants, both taken from named methods in Table 1.
""")

code(r"""
from disinfo.graphs import (knn_similarity_graph, threshold_similarity_graph,
                            attribute_graph, graph_stats)

# Benamira et al. (2019), Table 1 row 1: "the k-nearest-neighbor method
# (where k is set to 4)". This is one of the very few hyperparameters the
# survey states outright, so we use their value.
e_knn = knn_similarity_graph(bundle.X, k=4)
print(f"kNN (k=4)      : {e_knn.shape[1]} directed edges")

# Yuan et al. (2021): join items whose cosine similarity exceeds a threshold.
e_thr = threshold_similarity_graph(bundle.X, threshold=0.8)
print(f"threshold (0.8): {e_thr.shape[1]} directed edges")
""")

md(r"""
The threshold variant needs a guard the survey does not mention. Taken
literally, a threshold slightly too low produces a near-complete graph —
quadratic memory, and a GNN that averages the entire dataset into every node. We
cap each node's neighbours at the most similar `max_degree`.
""")

code(r"""
for t in [0.5, 0.7, 0.9]:
    e = threshold_similarity_graph(bundle.X, threshold=t, max_degree=64)
    print(f"threshold {t}: {e.shape[1]:>7} edges")
print("\n(all capped at max_degree=64; without the cap, threshold 0.5 is quadratic)")
""")

md(r"""
### The M-GCN attribute graph (Hu et al. 2019)

The LIAR row of Table 1 builds its graph a third way, and Section 5.3.2
describes it precisely enough to implement:

> a speaker's profile, including attributes such as their political party, the
> news topic, or their home state, is considered. **If two nodes share the same
> value for a given profile attribute, they are connected via an edge.**

Taken literally that is a union of cliques — and LIAR's `party` field has two
values covering most of the corpus, i.e. a clique of ~5,000 nodes and 12 million
edges carrying almost no information. We subsample within oversized groups,
preserving "these items share an attribute" while keeping degree finite.
""")

code(r"""
liar = datasets["liar"]
cfg_liar = Config(graph="attribute")
liar_idx = disinfo.make_splits(liar, cfg_liar)[0]
liar_feats = disinfo.build_features(liar, cfg_liar, liar_idx)

e_attr = attribute_graph(liar, seed=0)
print(f"LIAR attribute graph: {e_attr.shape[1]} directed edges "
      f"over {len(liar)} items")

e_liar_knn = knn_similarity_graph(liar_feats.X, k=4)
print(f"LIAR kNN graph      : {e_liar_knn.shape[1]} directed edges")
""")

md(r"""
## 6.2 Propagation graphs

One graph per news item, built from its repost tree. Node features concatenate
three blocks, each traceable to a named method:

1. **The source item's own features, broadcast to every node** — this is Bian et
   al.'s (2020) *root feature enhancement*: *"the source post information is
   integrated into each layer of GCN to enhance the influence from the roots of
   rumors."*
2. **Per-node user profile features**, where the corpus provides them (Malhotra
   et al. 2020 build the tree at user level with 12 profile features).
3. **The node's position in the cascade** — log delay and normalised index.
""")

code(r"""
graphs = disinfo.build_graphs(ds, bundle.X, cfg)
print("propagation graphs:", graph_stats(graphs))

g = graphs[0]
print(f"\nfirst cascade: {g.num_nodes} nodes, {g.num_edges} edges, "
      f"features {tuple(g.x.shape)}, label {ds.label_names[int(g.y)]}")
""")

md(r"""
## 6.3 Heterogeneous graphs

Section 5.3: *"Social networks are characterized by heterogeneous graphs,
encompassing various types of nodes (users, posts, comments, etc)."*

We add a **user node** per distinct author, joined to every post it wrote. That
is the tweet–user subgraph of Huang et al. (2020), and it does something the
propagation tree alone cannot: if a user reposts twice in one cascade, the two
branches become connected through them.

An honest limit: AA-HGNN and MFAN use *schema-level attention* over node types,
which Section 5.3.2 describes but **no equation in the survey defines**. We
build the typed graph and return `node_type`, but our layers are type-agnostic.
""")

code(r"""
cfg_het = cfg.with_(graph="heterogeneous")
het = disinfo.build_graphs(ds, bundle.X, cfg_het)
print("heterogeneous graphs:", graph_stats(het))

h0 = het[0]
n_post = h0.meta["n_post"]; n_user = h0.meta["n_user"]
print(f"\nfirst cascade: {n_post} post nodes + {n_user} user nodes "
      f"= {h0.num_nodes} total")
print(f"a user posted more than once in {sum(1 for x in het if x.meta['n_user'] < x.meta['n_post'])} "
      f"of {len(het)} cascades")
""")

# ===========================================================================
md(r"""
---
# Part 7 — Stages 3 and 4: the GNN and the classifier

Stage 3 stacks the layers of Part 3 into an encoder. Stage 4 reads the
embeddings out into a label.

For **graph** classification we must pool node embeddings into one vector per
graph. The survey does not specify how — Zhiyuan et al. (2020) contrast a global
embedding (GLO-PGNN) against ensembling per-node predictions (ENS-PGNN). We
default to mean pooling and offer `root`, which uses only the source post's
embedding, since *"the root node holds significant importance in rumor
detection"* (Sect. 5.3.2).

Let us run the whole of Figure 1, once, end to end.
""")

code(r"""
res = disinfo.run_experiment(ds, cfg.with_(gnn="gat", epochs=150))

m = res["metrics"]
print(f"dataset        : {res['dataset']}")
print(f"graph          : {cfg.graph}, {res['graph']['n_nodes']} nodes total")
print(f"feature dim    : {res['graph']['feature_dim']}")
print(f"train/val/test : {res['n_train']}/{res['n_val']}/{res['n_test']}")
print()
print(f"accuracy       : {m['accuracy']:.3f}")
print(f"macro-F1       : {m['macro_f1']:.3f}")
print(f"majority class : {m['majority_baseline']:.3f}   <- the floor")
print(f"AUC (macro OvR): {m['auc']:.3f}")
print()
print("per-class F1:")
for k, v in m["per_class_f1"].items():
    print(f"  {k:12s} {v:.3f}")
""")

code(r"""
fig = plots.plot_confusion(m["confusion"], ds.label_names,
                           title="Twitter15, GAT on the propagation graph")
plt.show()
""")

md(r"""
Always compare against the **majority baseline**, not against zero. Twitter15 is
balanced four ways, so the floor is 0.25 and an accuracy of 0.77 is a real
result. On PHEME the floor is 0.63, and we will see accuracies near 0.75 that
look similar but mean far less — which is exactly why we report macro-F1
alongside.
""")

# ===========================================================================
md(r"""
---
# Part 8 — Five experiments, each testing one claim

Each suite is aimed at a specific statement in the survey. All results are
pre-computed by `scripts/run_disinfo.py --experiments`; the cells below load
them, and the commented line re-runs from scratch.
""")

code(r"""
from disinfo.config import RESULTS

def load_or_none(name):
    p = RESULTS / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None

df_gnn   = load_or_none("results_gnn_comparison")
df_graph = load_or_none("results_graph_comparison")
df_abl   = load_or_none("results_feature_ablation")
df_liar  = load_or_none("results_liar_granularity")
df_split = load_or_none("results_pheme_split")

for n, d in [("architectures", df_gnn), ("graph construction", df_graph),
             ("feature ablation", df_abl), ("LIAR granularity", df_liar),
             ("PHEME split", df_split)]:
    print(f"{n:20s} {'loaded, ' + str(len(d)) + ' rows' if d is not None else 'MISSING - run scripts/run_disinfo.py --experiments'}")
""")

md(r"""
## 8.1 — Do GCN and GAT deserve their popularity?

Section 5.3.2 establishes that GCN and GAT are the **most used** architectures.
It does not claim they are the **best** — that inference is left to the reader,
and it is worth testing, because "most used" and "best" come apart routinely in
ML through path dependence.

We run all five of Section 3.1's architectures on identical graphs and features.
""")

code(r"""
if df_gnn is not None:
    print(df_gnn[["dataset", "graph", "gnn", "accuracy", "accuracy_std",
                  "macro_f1", "majority_baseline"]].to_string(index=False))
    fig = plots.plot_gnn_comparison(df_gnn); plt.show()
""")

code(r"""
if df_gnn is not None:
    best = df_gnn.loc[df_gnn.groupby("dataset")["accuracy"].idxmax()]
    print("best architecture per dataset:")
    print(best[["dataset", "gnn", "accuracy", "accuracy_std"]].to_string(index=False))
    print("\nmean accuracy across datasets, by architecture:")
    print(df_gnn.groupby("gnn")["accuracy"].agg(["mean", "std"]).round(3)
          .sort_values("mean", ascending=False).to_string())
""")

md(r"""
**Findings.**

- **GCN and GAT do win**, but narrowly — GCN takes CED, PHEME and Twitter16,
  GAT takes Twitter15, and the gap among the top four architectures is
  0.008–0.047, often within one standard deviation across seeds. So the
  survey's popularity ranking is not misleading, but neither is it a strong
  performance ranking, and Tables 1–2 could not have told us either way.
- **GIN is consistently worst**, by a wide and consistent margin — 0.64 on
  Twitter15 against GAT's 0.77, and 0.71 on CED against GCN's 0.85. This is a
  genuinely interesting result, because GIN is the *most theoretically
  expressive* of the five (Part 3, Eq. 10). The reason is its sum aggregation:
  cascade sizes here span 1 to 965 nodes, so a sum readout scales with graph
  size and the model spends its capacity encoding *how big* the cascade is
  rather than *what shape* it has. Maximum expressiveness on multisets is the
  wrong objective when multiset **size** is a nuisance variable.

That is a concrete, useful caution the survey's architecture taxonomy cannot
surface — it categorises by what a method *is*, never by what it costs.
""")

md(r"""
## 8.2 — Does graph construction matter more than architecture?

This tests the survey's own central organising claim (Sect. 5.3.1):

> the precise and effective modeling of information as a graph can profoundly
> impact the performance of these models. **Incorrect modeling can result in a
> significant deterioration of their performance.**

We hold the architecture fixed and vary only the construction.
""")

code(r"""
if df_graph is not None:
    print(df_graph[["dataset", "graph", "gnn", "accuracy", "accuracy_std",
                    "macro_f1"]].to_string(index=False))
    fig = plots.plot_graph_comparison(df_graph); plt.show()
""")

code(r"""
if df_gnn is not None and df_graph is not None:
    arch_spread = (df_gnn.groupby("dataset")["accuracy"]
                   .agg(lambda s: s.max() - s.min()).rename("architecture"))
    graph_spread = (df_graph.groupby("dataset")["accuracy"]
                    .agg(lambda s: s.max() - s.min()).rename("construction"))
    cmp = pd.concat([arch_spread, graph_spread], axis=1).round(3)
    cmp["construction wins"] = cmp["construction"] > cmp["architecture"]
    print("accuracy spread induced by each choice:\n")
    print(cmp.to_string())
""")

md(r"""
**This does not come out the way the survey predicts.**

Changing the *architecture* moves accuracy more than changing the *construction*
on **all five datasets** — and it still does after excluding GIN, the outlier
from 8.1. Construction shifts accuracy by 0.004–0.029; architecture by
0.008–0.047 excluding GIN, and up to 0.139 including it.

Before reading this as a refutation, note a real limitation of our own design.
All three of our constructions give every node the **source item's feature
vector**, broadcast (Bian et al.'s root feature enhancement, Part 6.2). So even
the propagation graph carries everything the similarity graph knows, and the
three differ only in what they *add*. That is faithful to how the surveyed
methods build node features, but it means our comparison measures the *marginal*
value of structure, not construction versus content from scratch.

Read that way, the finding is sharper and more useful: **once the source
content is in the node features, the choice of graph construction adds
remarkably little.** Which sets up the next experiment.

## 8.3 — Which of Figure 4's features carry the signal?

Section 5.3.2 observes that most methods use textual features. Is that a
finding, or a habit? We zero one feature block at a time — keeping the input
dimension and parameter count fixed, so a drop is attributable to information
removed rather than to a smaller model.
""")

code(r"""
if df_abl is not None:
    print(df_abl[["variant", "accuracy", "accuracy_std", "macro_f1"]]
          .sort_values("accuracy", ascending=False).to_string(index=False))
    fig = plots.plot_ablation(df_abl); plt.show()
""")

md(r"""
**The result is blunt.** Remove the lexical block and accuracy collapses from
0.754 to 0.370. Remove *any other* block — syntactic, profile, propagation,
temporal — and accuracy does not move outside one standard deviation. Keep
**only** the lexical block and accuracy is 0.768, marginally *better* than using
everything.

On Twitter15, the propagation and temporal features contribute nothing once the
source text is present.

This deserves care rather than over-claiming. It does **not** mean propagation
structure is uninformative — Part 5 showed the class means separate clearly on
structure alone, and a structure-only model would beat the 0.25 majority
baseline comfortably. It means the structural signal here is **redundant** with
the textual signal: both encode the same distinction, and text encodes it more
sharply.

But it does sit awkwardly beside the survey's framing. Section 4.1's argument
for context-based features is *adversarial* — authors can mimic writing style,
so text-based methods decay. That argument is about **robustness over time**,
not accuracy on a fixed benchmark. On a static test split, the feature an
adversary could game is the one that wins. Tables 1–2, which report exactly such
single-split accuracies, are structurally incapable of measuring the property
that motivates half the methods they catalogue.

## 8.4 — The LIAR contradiction

From Part 2 we have three mutually inconsistent statements about LIAR:

| Source | Claim |
|---|---|
| Table 1 (Hu et al. 2019) | ACC **0.492** |
| Table 2 (Cui et al. 2023) | ACC **0.868** |
| Section 7 | multiclass accuracy "typically below 50%" |

Our hypothesis: 0.868 is **binary** LIAR, and the survey does not say so. We
also test the credit-history columns, which leak the label — each count is
computed over the speaker's full record *including the statement being
classified* — and which papers reporting LIAR accuracy rarely mention either way.
""")

code(r"""
if df_liar is not None:
    print(df_liar[["variant", "accuracy", "accuracy_std", "macro_f1",
                   "majority_baseline"]].to_string(index=False))
""")

md(r"""
**6-class: 0.236. Binary: 0.604.** Collapsing the label space is worth roughly
0.37 accuracy, which confirms the direction of our hypothesis — the two LIAR
rows in Tables 1–2 cannot be the same task.

But it does **not** fully close the gap: 0.604 is still well short of 0.868, and
against a binary majority baseline of 0.558 our binary model is only modestly
above chance. So binarisation alone does not explain Cui et al.'s number. The
remainder is presumably their GGNN over semantic graphs versus our TF-IDF, plus
whatever their class-collapsing rule was. We cannot tell, because neither the
survey nor its summary of the method states either.

The credit-history columns turn out to matter **almost not at all** here
(0.236 → 0.237; 0.604 → 0.605), which surprised us — the leakage concern is real
in principle but our SVD-compressed representation evidently does not exploit
it. Worth reporting precisely because it contradicts what we expected when we
built the flag.

**Conclusion:** Section 7's "below 50%" is right about LIAR — we measure 0.236 —
and wrong as a general statement about multiclass work, since Part 2 found 35 of
36 multiclass accuracies above 0.5. Table 2's 0.868 is a different task from
Table 1's 0.492, printed in the same column, under the same dataset name, with
no distinction drawn.

Our 6-class 0.236 also sits below Hu et al.'s 0.492, and we should be explicit
about why: they model the speaker profile richly (that is the whole point of
M-GCN), we use TF-IDF plus a party one-hot. A replication that closed that gap
by switching leakage on would be a worse replication, not a better one.

## 8.5 — How much of PHEME's reported spread is the split?

Part 2 found published PHEME accuracies from 0.694 to 0.887 with **no paper
stating its protocol**. Here is one candidate explanation.

PHEME has nine breaking-news events. Threads within an event share entities,
phrasing and timing. Under a **random** split, threads from the same event
appear in train and test — so a model can succeed by recognising *the event*
rather than judging *veracity*. Under a **leave-events-out** split it cannot.
""")

code(r"""
if df_split is not None:
    print(df_split[["variant", "accuracy", "accuracy_std", "macro_f1",
                    "majority_baseline"]].to_string(index=False))
    if len(df_split) == 2:
        gap = df_split["accuracy"].max() - df_split["accuracy"].min()
        print(f"\ngap between protocols: {gap:.3f} accuracy")
        print(f"published PHEME spread in Tables 1-2: "
              f"{long_df[(long_df.dataset=='PHEME') & (long_df.metric=='ACC')]['value'].max() - long_df[(long_df.dataset=='PHEME') & (long_df.metric=='ACC')]['value'].min():.3f}")
""")

md(r"""
**This is the strongest result in the notebook.**

| Protocol | Accuracy | s.d. over seeds |
|---|---|---|
| random (stratified) | 0.754 | **0.009** |
| leave-events-out | 0.640 | **0.135** |

Two things happen at once, and the second matters more than the first.

1. **Accuracy falls by 0.114** — a large drop for a change that touches no model
   code whatsoever.
2. **The standard deviation grows fifteen-fold**, from 0.009 to 0.135. Under a
   random split, PHEME looks like a stable benchmark. Under an event split it is
   revealed to be a *nine-sample* problem: which events land in the test set
   swamps everything else.

Now compare with the literature. Published PHEME accuracies in Tables 1–2 span
**0.694 to 0.887, a spread of 0.193**. Our protocol change alone accounts for
0.114 of that, and the per-seed spread under the event split (±0.135) covers the
rest several times over.

So a large part of what Tables 1–2 present as *methodological progress on PHEME*
is plausibly **evaluation protocol and event luck** — invisible, because no
paper reports either. A method could improve its PHEME number by 0.1 without
improving anything at all.

This sharpens Section 7's "gold standard datasets" open problem in a way the
survey does not. The datasets are not the bottleneck: all four downloaded
cleanly and matched Table 3 exactly (Part 4). What is missing is a **shared
protocol** — and a survey, uniquely positioned to see all 34 papers at once, is
exactly where that demand should have been made.
""")

# ===========================================================================
md(r"""
---
# Part 9 — This replication against the published range

The honest comparison is against a **range**, not a point. Tables 1–2 give one
number per (method, dataset) with no variance and no protocol, drawn from
different splits, encoders and preprocessing. Matching any single row would be a
coincidence; landing inside the range is the strongest claim available.
""")

code(r"""
frames = [d for d in (df_gnn, df_graph) if d is not None]
if frames:
    ours = pd.concat(frames, ignore_index=True)
    name_map = {"twitter15": "Twitter15", "twitter16": "Twitter16"}
    ours["dataset"] = ours["dataset"].map(lambda d: name_map.get(d, d))
    fig = plots.plot_vs_literature(ours, long_df); plt.show()
""")

code(r"""
if frames:
    acc = long_df[long_df["metric"] == "ACC"]
    rows = []
    for d in ours["dataset"].unique():
        lit = acc.loc[acc["dataset"] == d, "value"]
        if lit.empty:
            continue
        sub = ours[ours["dataset"] == d]
        best = sub.loc[sub["accuracy"].idxmax()]
        rows.append({
            "dataset": d,
            "ours": round(best["accuracy"], 3),
            "config": f"{best['gnn']}/{best['graph']}",
            "published min": round(lit.min(), 3),
            "published median": round(lit.median(), 3),
            "published max": round(lit.max(), 3),
            "n published": len(lit),
            "inside range": bool(lit.min() <= best["accuracy"] <= lit.max()),
        })
    print(pd.DataFrame(rows).to_string(index=False))
""")

md(r"""
### How to read this

Where we land **inside** the published range, the framework of Figure 1 —
implemented from the survey's equations, with no tuning — reproduces the
literature's operating point. That is the substantive replication result.

Where we land **below** it, the causes are known and listed in
`config.INFERRED_PARAMETERS`. The three that dominate:

1. **Text encoding.** We use TF-IDF+SVD; the strongest rows use BERT, RoBERTa or
   GloVe. On corpora where only the source tweet has text, the encoder is doing
   most of the work.
2. **No tuning.** One configuration for all datasets. Published numbers are
   tuned per dataset.
3. **Missing feature types.** No visual, semantic or stance features (Part 5) —
   which is precisely what the top hybrid methods combine.

None of these is a defect of the survey. They are the consequence of replicating
a *survey*: it describes 34 methods in prose and specifies none of them
completely enough to reproduce exactly. That is worth stating plainly, because
it is the strongest practical finding of the whole exercise.
""")

# ===========================================================================
md(r"""
---
# Part 10 — What we learned

## On the survey

The paper is a good, useful survey. Its taxonomies are coherent, its coverage of
2019–2023 GNN work is thorough, and its organising insight — that GNN
disinformation methods are best distinguished by **how the graph is built**
rather than by which layer they use — is a genuinely clarifying way to read the
literature.

As a *predictor of accuracy*, though, that axis did not hold up in Part 8.2:
architecture moved our numbers more than construction did, on every dataset. We
would not push that far, given the caveat there (all our constructions share the
broadcast source features). But it is worth saying plainly, because it is the
survey's own central organising claim and it is testable.

What replication surfaced:

**1. Five of six prose claims are supported by the paper's own tables; one is
not.** Section 7's "multiclass accuracy typically below 50%" is contradicted by
35 of 36 multiclass accuracies in Tables 1–2 (median 0.881). It appears to
generalise a fact about LIAR to all multiclass work, and it is offered as
motivation for a research direction.

**2. Two internal contradictions about LIAR.** Table 1 says 0.492, Table 2 says
0.868, Section 7 says "below 50%". Part 8.4 shows the gap is the 6-class/binary
distinction, which is never stated.

**3. Three of the printed equations differ from the works they cite.** Eq. 2 is
not Kipf & Welling's GCN and divides $|N(v)|{+}1$ terms by $|N(v)|$; Eq. 4 omits
GAT's softmax; Eq. 8 prints `mean` where the prose says "maximum pooling" and
the source uses `max`. Anyone implementing from the survey alone would build
three subtly wrong layers. All are documented in
`docs/DISCREPANCIES_SURVEY.md`.

**4. The survey's own strongest argument checks out.** Its explanation of why
GATv2 supersedes GAT — composed linear maps collapse, making attention
independent of the receiving node — is correct, and Part 3 demonstrates it in
six lines.

**5. Table 3 is exactly right.** All four obtainable datasets match their
printed sizes to the item.

## What our own experiments found

**6. Protocol beats method on PHEME (8.5).** Switching from a random split to
leave-events-out costs 0.114 accuracy and inflates the seed-to-seed standard
deviation fifteen-fold (0.009 → 0.135). The entire published PHEME spread in
Tables 1–2 is 0.193. No paper states which protocol it used.

**7. On Twitter15, text is nearly everything (8.3).** Lexical features alone
score 0.768; all features together score 0.754; removing lexical collapses it to
0.370. Propagation and temporal features are *redundant* with text here, not
uninformative — and the survey's case for context features is about
**robustness to adversaries**, a property single-split accuracy tables cannot
measure at all.

**8. The most theoretically expressive layer is the worst one (8.1).** GIN loses
by 0.10–0.14. Its sum aggregation makes the readout scale with cascade size,
which ranges from 1 to 965 nodes here — maximum expressiveness on multisets is
the wrong objective when multiset size is a nuisance variable.

## On the open problems of Section 7

Our runs speak to several of them:

- **Gold standard datasets.** The datasets are not the bottleneck — all four
  downloaded cleanly and matched Table 3 exactly. The **protocol** is, and
  finding 6 quantifies it.
- **Multiclass classification.** Genuinely hard — we measure 0.236 on 6-class
  LIAR against a 0.206 baseline — but hard on LIAR specifically, not in general.
- **Visual features.** Section 7 calls them under-used; Part 2 shows only 2 of
  34 methods use them. We could not test this at all: **no corpus in Table 3
  redistributes its images.** The open problem is partly a data-availability
  problem, which Section 7 does not say.
- **Early detection.** Directly visible in our temporal features: the signal
  that makes propagation graphs work is cascade shape, which does not exist at
  $t=0$. The content/context trade-off of Section 5.2.3 is real and structural.

## Reproducing everything

```bash
P=01-info-propagation/desinformation-paper/.venv/bin/python

bash scripts/get_disinfo_data.sh              # ~370 MB
$P scripts/run_disinfo.py --quiet             # figures + meta-analysis (~30 s)
$P scripts/run_disinfo.py --quiet --experiments   # + the live runs
$P scripts/build_disinfo_notebook.py          # regenerate this notebook
```

`replication.ipynb` is a **build artefact** of `scripts/build_disinfo_notebook.py`.
Edit the builder, not the notebook, or your changes are lost on the next
regeneration.

## Files

| Path | Contents |
|---|---|
| `disinfo/layers.py` | Eqs. 2–10, from the printed equations |
| `disinfo/checks.py` | The six property checks of Part 3 |
| `disinfo/survey_data.py` | Tables 1–4 as data; `verify_claims` |
| `disinfo/taxonomy.py` | Figs. 1–6 as structures |
| `disinfo/data.py` | Loaders for LIAR, Twitter15/16, PHEME, CED |
| `disinfo/features.py` | Stage 1 — the leaves of Fig. 4 |
| `disinfo/graphs.py` | Stage 2 — the three constructions of Sect. 5.3 |
| `disinfo/models.py`, `pipeline.py` | Stages 3–4 |
| `disinfo/experiments.py` | The five suites of Part 8 |
| `disinfo/plots.py` | Every figure |
| `docs/DISCREPANCIES_SURVEY.md` | **Read before changing `layers.py`** |
""")

# ===========================================================================

nb = {
    "cells": cells,
    "metadata": {
        # The venv beside this paper, registered with:
        #   .venv/bin/python -m ipykernel install --user --name disinfo-venv \
        #       --display-name "Python (desinformation-paper)"
        "kernelspec": {"display_name": "Python (desinformation-paper)",
                       "language": "python", "name": "disinfo-venv"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {OUT}  ({len(cells)} cells)")
