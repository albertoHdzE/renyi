"""Configuration for the DTWRE replication.

Every value here is traceable to the paper (Tong et al., *Entropy* 2025, 27, 516).
Values the paper does not state are marked ``INFERRED`` in the field comment and
are collected in :data:`INFERRED_PARAMETERS` so the notebook can display them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# Repository root (this file lives at <root>/dtwre/config.py)
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


@dataclass
class Config:
    """Hyperparameters for one DTWRE experiment."""

    # ---- Rényi entropy (Section 2.2, Eq. 1-4) ----
    alpha: float = 0.6          # Section 4.2.1: optimal value
    lam: float = 1.2            # Section 4.2.2: optimal value ("lambda")
    prob_metric: str = "degree"  # "degree" or "pagerank" (Section 2.2)
    # Eq. 1 sums p_u^alpha over neighbours of v, where p comes from Algorithm 1
    # as a GLOBAL normalisation over all nodes. "global" reproduces Algorithm 1
    # literally but makes sum(p) < 1, which diverges at alpha -> 1; "local"
    # renormalises within the neighbourhood so Eq. 1 is a true Renyi entropy.
    # Default is "local". See docs/DISCREPANCIES.md.
    prob_normalisation: str = "local"

    # ---- Temporal segmentation (Section 3.1) ----
    window_seconds: int = 604_800   # 7 days, best in Table 2
    # Eq. 4 weight exp(-lam * (t - t_k)). Units of (t - t_k) are TIME-STEP
    # INDICES, not seconds -- with seconds the weight underflows to 0. INFERRED.
    time_unit: str = "steps"

    # ---- Node2Vec (Section 3.2.2) ----
    embedding_dim: int = 64     # stated: "embedding dimension set to 64"
    walk_length: int = 80       # INFERRED (Grover & Leskovec defaults)
    num_walks: int = 10         # INFERRED
    context_window: int = 10    # INFERRED
    p: float = 1.0              # INFERRED
    q: float = 1.0              # INFERRED
    n2v_epochs: int = 5         # INFERRED
    # INFERRED; selected by edge-reconstruction AUC on the training graph
    # (0.84 vs 0.75 at the word2vec default of 0.025 under Adam).
    n2v_lr: float = 0.005
    n2v_negative: int = 5       # INFERRED

    # ---- GraphSAGE + MLP (Section 2.3, 3.3) ----
    hidden_dim: int = 64        # INFERRED
    num_layers: int = 2         # INFERRED (standard GraphSAGE depth)
    fanout: int = 25            # neighbour sample size; INFERRED
    dropout: float = 0.2        # INFERRED
    concat_self: bool = True    # standard GraphSAGE; Eq. 5 omits the self term

    # ---- Training (Section 3.3) ----
    epochs: int = 100           # stated: "trained over 100 epochs"
    lr: float = 0.005           # INFERRED; selected on validation AUC
    weight_decay: float = 5e-4  # INFERRED
    neg_ratio: float = 0.75     # Section 4.2.3 optimum; n_neg / n_pos

    # ---- Data preprocessing (Section 3.1) ----
    # Augmentation is introduced "given the scale of real-world social network
    # datasets", i.e. for Weibo; it is off by default for the CollegeMsg
    # benchmark and applied to entropy snapshots only, never to supervision.
    augment: bool = False
    add_frac: float = 0.05      # "randomly add 5% of nodes and edges"
    del_frac: float = 0.02      # "delete 2% of nodes and edges"
    drop_isolated: bool = True
    train_frac: float = 0.80    # chronological 80/10/10
    val_frac: float = 0.10
    # "80% of the total duration and 90% of the total duration" (Section 3.1)
    # => cut points on the timeline. "count" splits by edge rank instead,
    # which is the more common protocol and yields far larger test sets.
    split_by: str = "duration"
    # If True, a val/test positive must be a pair the encoder has never
    # propagated over (true link *formation*). If False, any pair interacting
    # in the window counts, including repeats -- the paper's literal wording
    # ("connected node pairs extracted from the actual network").
    new_links_only: bool = True
    # Fraction of the *training period* used for message passing; the tail
    # supplies supervision positives. Keeps the edges the encoder propagates
    # over disjoint from the edges it is trained to predict. INFERRED -- the
    # paper does not describe how training positives are held out.
    # Selected on validation AUC over {0.5, 0.7, 0.9}: 0.7 gives 644 training
    # positives, enough to train the 67-dim model without collapsing.
    mp_frac: float = 0.70

    seed: int = 42              # INFERRED

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Parameter sweeps used by Sections 4.2.1-4.2.4 ----
ALPHA_GRID = [0.2, 0.6, 1.0, 1.5, 2.0, 5.0]      # Figure 6
LAMBDA_GRID = [0.1, 0.4, 0.8, 1.2, 2.0]          # Figure 7
NEG_RATIO_GRID = [0.5, 0.75, 0.8, 1.0, 2.0]      # Figure 8
WINDOW_GRID = [151_200, 302_400, 604_800]        # Table 2 / Figure 9

BASELINES = ["node_degree", "node_pagerank", "node2vec", "renyi_static", "dtwre"]

BASELINE_LABELS = {
    "node_degree": "Node Degree",
    "node_pagerank": "Node PageRank",
    "node2vec": "Node2vec",
    "renyi_static": "Renyi Entropy",
    "dtwre": "DTWRE",
}

METRICS = ["auc", "precision", "recall", "f1", "accuracy"]

# ---- Published results, for fidelity comparison ----
# Table 1: CollegeMsg comparison with baselines.
PAPER_TABLE1 = {
    "node_degree":   {"auc": 0.9323, "precision": 0.8522, "recall": 0.8618,
                      "f1": 0.8570, "accuracy": 0.8562},
    "node_pagerank": {"auc": 0.9285, "precision": 0.8509, "recall": 0.8613,
                      "f1": 0.8561, "accuracy": 0.8552},
    "node2vec":      {"auc": 0.9649, "precision": 0.9221, "recall": 0.8909,
                      "f1": 0.9062, "accuracy": 0.9078},
    "renyi_static":  {"auc": 0.9487, "precision": 0.8677, "recall": 0.8970,
                      "f1": 0.8821, "accuracy": 0.8802},
    "dtwre":         {"auc": 0.9742, "precision": 0.9259, "recall": 0.9144,
                      "f1": 0.9201, "accuracy": 0.9207},
}

# Table 2: effect of the temporal window length.
PAPER_TABLE2 = {
    604_800: {"auc": 0.9680, "precision": 0.9159, "recall": 0.9044,
              "f1": 0.9101, "accuracy": 0.9107},
    302_400: {"auc": 0.9567, "precision": 0.9016, "recall": 0.8972,
              "f1": 0.8994, "accuracy": 0.8996},
    151_200: {"auc": 0.9579, "precision": 0.8963, "recall": 0.8882,
              "f1": 0.8922, "accuracy": 0.8927},
}

# Section 4.2.1 quotes AUC and accuracy for the alpha sweep in the text.
PAPER_ALPHA_AUC = {0.2: 0.950, 0.6: 0.966, 1.0: 0.959,
                   1.5: 0.955, 2.0: 0.952, 5.0: 0.944}
PAPER_ALPHA_ACC = {0.2: 0.888, 0.6: 0.916, 5.0: 0.904}

# Table 3: time complexity (symbolic, not measured).
PAPER_TABLE3 = {
    "This research": ("O(NTk + rNl + Nk^d)", "Entropy calculation, random walks, GNN"),
    "GCN":           ("O(mdD)",              "Full-graph convolution (m: edges)"),
    "Node2Vec":      ("O(rNl)",              "Random walks, Skip-Gram"),
    "DeepWalk":      ("O(rNl)",              "Random walks, hierarchical softmax"),
}

INFERRED_PARAMETERS = [
    ("walk_length, num_walks, context_window, p, q",
     "Node2Vec walk settings", "Grover & Leskovec defaults"),
    ("hidden_dim, num_layers, fanout, dropout",
     "GraphSAGE architecture", "standard 2-layer mean-aggregator SAGE"),
    ("lr, weight_decay", "Adam settings",
     "paper says only 'refined via cross-validation'"),
    ("seed", "random seed", "not reported"),
    ("time_unit", "units of (t - t_k) in Eq. 4",
     "must be time-step indices or exp(-lam*dt) underflows"),
    ("concat_self", "self term in Eq. 5",
     "Eq. 5 shows only Aggregate(); standard GraphSAGE concatenates self"),
]
