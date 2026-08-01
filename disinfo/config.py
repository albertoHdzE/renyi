"""Paths and hyperparameters for the survey replication.

``Config`` is the single source of hyperparameters, matching the convention
``dtwre.config`` sets for the entropia replication. The difference is that a
survey states almost no hyperparameters: it describes 34 other papers' methods
in prose. So ``INFERRED_PARAMETERS`` here is long, and everything in it is a
choice made by this replication rather than a value read off the paper.

Where the survey *does* pin a number down -- Benamira et al.'s k=4 for the kNN
similarity graph, Wang et al.'s 2-layer GCN, Malhotra et al.'s and Thota et
al.'s 2-layer GCN -- the value is used and attributed in the comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Config", "ROOT", "DATA_RAW", "DATA_PROCESSED", "RESULTS", "FIGURES",
           "INFERRED_PARAMETERS", "GNN_GRID", "GRAPH_GRID"]

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "disinfo"
FIGURES = RESULTS / "figures"


@dataclass
class Config:
    """Hyperparameters for one run of the Fig. 1 pipeline."""

    # ---- Stage 1: feature extraction (Sect. 4.1) ----
    text_features: str = "tfidf"      # tfidf | hashing
    max_features: int = 5000
    ngram_max: int = 2                # Sect. 4.1 names unigrams and bigrams
    min_df: int = 2
    svd_dim: int = 256                # TF-IDF is too sparse/wide for a dense GNN
    use_profile: bool = True          # Sect. 4.2 user-based features
    use_credit_history: bool = False  # LIAR leakage; see data.load_liar

    # ---- Stage 2: graph construction (Sect. 5.3) ----
    graph: str = "similarity"         # similarity | attribute | propagation
    knn_k: int = 4                    # Benamira et al. (2019): "k is set to 4"
    similarity: str = "cosine"
    sim_threshold: float | None = None  # Yuan et al. (2021) threshold variant
    undirected: bool = True           # Jiang et al. (2021) treat edges as bi-directional
    add_root_edges: bool = True       # Bian et al. (2020) root feature enhancement

    # ---- Stage 3: GNN (Sect. 3.1) ----
    gnn: str = "gcn"                  # gcn | gat | gatv2 | sage | gin
    hidden_dim: int = 64
    num_layers: int = 2               # Wang et al. (2020), Malhotra et al. (2020)
    heads: int = 4                    # GAT/GATv2 only
    dropout: float = 0.5
    sage_aggregator: str = "mean"
    gcn_normalization: str = "symmetric"
    gin_train_eps: bool = False
    readout: str = "mean"             # graph classification pooling

    # ---- Stage 4: classification (Sect. 5.1) ----
    classifier: str = "linear"        # linear | mlp
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 30
    batch_size: int = 64              # graph classification only

    # ---- protocol ----
    seed: int = 0
    n_seeds: int = 5
    test_frac: float = 0.2
    val_frac: float = 0.1
    split: str = "stratified"         # stratified | event (PHEME only)
    device: str = "cpu"

    def with_(self, **kw) -> "Config":
        """A copy with fields overridden -- for sweeps."""
        from dataclasses import replace
        return replace(self, **kw)


# Values the survey does not state, chosen here. Anything in this list is a
# decision of the replication and a candidate explanation for a gap against
# Tables 1-2. Keep accurate when adding parameters.
INFERRED_PARAMETERS = [
    ("text_features=tfidf", "Table 1-2 methods use GloVe (Benamira), BERT "
     "(Autef), RoBERTa (Malhotra) or TF-IDF (Wei). TF-IDF is chosen because it "
     "needs no download and works for Chinese (CED) as well as English."),
    ("svd_dim=256", "No method states an input dimension. Truncated SVD keeps "
     "the GNN input dense and comparable across datasets."),
    ("hidden_dim=64", "Not stated by any surveyed method."),
    ("num_layers=2", "Stated by Wang et al. (2020), Malhotra et al. (2020) and "
     "Thota et al. (2023); assumed for the rest."),
    ("heads=4", "GAT head count is never stated in the survey."),
    ("dropout=0.5", "Not stated; the GCN/GAT default."),
    ("lr=0.01, weight_decay=5e-4", "Not stated; the Kipf & Welling defaults."),
    ("epochs=200, patience=30", "Not stated. Early stopping on validation loss."),
    ("test_frac=0.2, val_frac=0.1", "No surveyed method states its split. This "
     "is the single largest source of incomparability with Tables 1-2."),
    ("n_seeds=5", "Tables 1-2 report point estimates with no variance, so the "
     "spread reported here has no counterpart in the paper."),
    ("readout=mean", "Graph-level pooling is unstated; Zhiyuan et al. (2020) "
     "contrast global embedding against per-node ensembling."),
    ("use_credit_history=False", "LIAR's credit-history columns leak the label. "
     "Papers reporting LIAR accuracy rarely say whether they used them."),
]

# Sweeps used by scripts/run_disinfo.py.
GNN_GRID = ["gcn", "gat", "gatv2", "sage", "gin"]
GRAPH_GRID = ["similarity", "attribute", "propagation"]
