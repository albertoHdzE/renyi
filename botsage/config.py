"""Paths, hyperparameters and the published results of Deshmukh (2025).

Deshmukh, A. *Bot Detection in Social Media using GraphSage and BERT.*
Master's Project 1465, San José State University, Spring 2025.
https://doi.org/10.31979/etd.wb6h-3yd6

``Config`` is the single source of hyperparameters, following the convention
``dtwre.config`` and ``disinfo.config`` set. This paper is unusually specific
about some values (5 in-channels, 128 out-channels, 768-dim BERT, 896-dim
concatenation, linear-kernel SVM, 5-fold CV, 15 tweets per user, tokenizer
``max_length=50``) and silent about others (the SVM's C, the random seed of the
untrained GraphSAGE layer, any feature scaling). ``INFERRED_PARAMETERS`` lists
the silences.

Artefacts live beside the paper, not at the repository root, so each replication
is self-contained:

    01-info-propagation/bot-detection-paper/
        .venv/            environment
        replication.ipynb build artefact of scripts/build_botsage_notebook.py
        results/          tables, figures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Config", "ROOT", "PAPER_DIR", "DATA_RAW", "RESULTS", "FIGURES",
           "PAPER_TABLE4", "PAPER_TABLE5", "PAPER_TABLE1", "PAPER_TABLE2",
           "PAPER_TABLE3", "STATED_PARAMETERS", "INFERRED_PARAMETERS",
           "PAPER", "TWIBOT22_LABEL_COUNTS"]

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "01-info-propagation" / "bot-detection-paper"
DATA_RAW = ROOT / "data" / "raw"
RESULTS = PAPER_DIR / "results"
FIGURES = RESULTS / "figures"


PAPER = {
    "title": "Bot Detection in Social Media using GraphSage and BERT",
    "author": "Abhishek Deshmukh",
    "venue": "Master's Project 1465, San Jose State University, Spring 2025",
    "doi": "10.31979/etd.wb6h-3yd6",
    "advisor": "Dr. Teng Moh",
    "kind": "Master's project (no code release; appendix lists one script)",
}


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class Config:
    """Hyperparameters for one run of the Figure 2 workflow."""

    # ---- text branch (Sects. 3.1.2, 3.2) ----
    bert_model: str = "distilbert-base-uncased"
    text_dim: int = 768               # "it provides 768-size vectors"
    max_length: int = 50              # appendix listing A.1, tokenizer
    max_tweets: int | None = 15       # "only the 15 most recent tweets"; None = all
    token_pooling: str = "mean"       # A.1 means over tokens, then over tweets
    clean_text: bool = True           # Sect. 3.1.2 normalisation
    batch_size: int = 64

    # ---- node features (Sect. 3.3) ----
    n_node_features: int = 5          # "which in this case is 5"
    standardize_features: bool = True  # NOT stated by the paper; see DISCREPANCIES

    # ---- graph (Sects. 3.1.1, 3.4) ----
    undirected: bool = True           # "the goal is to construct a homogeneous graph"

    # ---- GraphSAGE (Sect. 3.5) ----
    sage_out: int = 128               # "which is set to 128"
    sage_layers: int = 1              # "A single sageConv layer is used"
    sage_trained: bool = False        # Sect. 3.5: no prediction head, no epochs
    sage_seed: int = 0                # NOT stated; an untrained layer depends on it
    sage_aggr: str = "mean"           # PyG SAGEConv default
    sage_root_weight: bool = True     # PyG SAGEConv default
    sage_normalize: bool = False      # PyG SAGEConv default

    # ---- classifier (Sects. 3.7, 3.8) ----
    svm_kernel: str = "linear"        # "initialized using a linear kernel"
    svm_C: float = 1.0                # "appropriately adjusted"; value never given
    svm_max_iter: int = 5000
    class_weight: str | None = None   # NOT stated; matters hugely on TwiBot-22
    n_folds: int = 5                  # "5 fold cross-validation"
    protocol: str = "cv"              # cv | official_split
    seed: int = 0

    # ---- ablation switches (not the paper's; used by experiments) ----
    use_text: bool = True
    use_graph: bool = True

    def with_(self, **kw) -> "Config":
        from dataclasses import replace
        return replace(self, **kw)

    @property
    def embedding_dim(self) -> int:
        """896 = 128 + 768 when both branches are on (Sect. 3.6)."""
        return (self.sage_out if self.use_graph else 0) + \
               (self.text_dim if self.use_text else 0)


# --------------------------------------------------------------------------
# Published results
# --------------------------------------------------------------------------
# Type (F/G) is the paper's own column: F = feature-based, G = graph-based.
# `is_this_paper` marks the two rows the author contributes.

PAPER_TABLE4 = [
    # method, type, accuracy %, F1 %, is_this_paper
    ("SATAR", "F", 93.42, 95.05, False),
    ("GCN", "G", 77.08, 77.91, False),
    ("BotRGCN", "G", 96.52, 97.30, False),
    ("RGT", "G", 97.15, 97.78, False),
    ("BIC", "G", 98.35, 98.71, False),
    ("GraphSage+BERT", "G", 98.68, 98.92, True),
    ("GraphSage+DistilBERT", "G", 98.56, 98.88, True),
]

PAPER_TABLE5 = [
    ("SVM", "F", 49.30, None, False),
    ("GCN", "G", 47.72, 38.10, False),
    ("HGT", "G", 74.91, 39.60, False),
    ("BotRGCN", "G", 79.66, 57.50, False),
    ("RGT", "G", 76.47, 49.24, False),
    ("BGSRD", "G", 71.88, 21.14, False),
    ("GraphSage+SVM", "G", 67.21, 48.32, True),
    ("GraphSage+DistilBERT", "G", 74.62, 51.69, True),
]

# Tables 1-3: wall-clock, on "a MacBook Pro 2019, 2.4 GHz Quad-Core i5, 8 GB".
PAPER_TABLE1 = {"BERT": 320 + 23 / 60, "DistilBERT": 100 + 55 / 60}
PAPER_TABLE2 = {"BERT": 75 + 12 / 60, "DistilBERT": 13 + 40 / 60}
PAPER_TABLE3 = [
    ("BERT+GraphSage", "Cresci-15", 3 + 47 / 60),
    ("DistilBERT+GraphSage", "Cresci-15", 2 + 32 / 60),
    ("GraphSage+SVM", "Twibot-22", 10 + 55 / 60),
    ("DistilBERT+GraphSage", "Twibot-22", 14 + 29 / 60),
    ("BERT+GraphSage*", "Twibot-22", 300.0),      # ">300"
]

# Verified directly from the open TwiBot-22 label.csv (Zenodo record 7012904).
# The trivial "always predict human" classifier scores 86.01% on the full
# corpus. Every accuracy in PAPER_TABLE5 is below it. See DISCREPANCIES §2.
TWIBOT22_LABEL_COUNTS = {"bot": 139_943, "human": 860_057}


# --------------------------------------------------------------------------
# What the paper does and does not pin down
# --------------------------------------------------------------------------

STATED_PARAMETERS = [
    ("in_channels=5", "Sect. 3.5: 'the number of node features used, which in "
     "this case is 5'."),
    ("out_channels=128", "Sect. 3.5: 'the size of the embeddings generated, "
     "which is set to 128'."),
    ("single SAGEConv layer", "Sect. 3.5: 'A single sageConv layer is used'."),
    ("GraphSAGE is not trained", "Sect. 3.5: 'training epochs and optimization "
     "tasks are not required due to the lack of a prediction head'. The layer "
     "runs in eval mode, in a single forward pass, at its random initialisation."),
    ("text_dim=768", "Sect. 3.2, and 896 = 128 + 768 in Sect. 3.6."),
    ("max_tweets=15", "Sect. 2.4.1, for Twibot-22 only; Cresci-15 uses all."),
    ("max_length=50", "Appendix Listing A.1, the tokenizer call."),
    ("mean over tokens, then mean over tweets", "Listing A.1 takes "
     "torch.mean(..., dim=0) over the token axis, then over the tweet axis."),
    ("zeros(768) for users with no tweets", "Listing A.1."),
    ("SVM, linear kernel", "Sect. 3.7."),
    ("5-fold cross-validation", "Sects. 3.7 and 4.1.2."),
]

INFERRED_PARAMETERS = [
    ("sage_seed", "An untrained layer's output is entirely determined by its "
     "random initialisation, which the paper never fixes or reports. "
     "`experiments.suite_seed_sensitivity` measures how much this alone moves "
     "the result."),
    ("svm_C=1.0", "Sect. 3.7 says only that 'the regularization term was "
     "appropriately adjusted to avoid overfitting'. No value is given, and no "
     "held-out set is described on which it could have been tuned."),
    ("standardize_features=True", "Never mentioned. The five raw features are "
     "counts spanning six orders of magnitude; concatenated with BERT "
     "embeddings of order 0.1 they would dominate a linear SVM entirely. See "
     "DISCREPANCIES §7."),
    ("class_weight=None", "Never mentioned. On TwiBot-22 (14% bots) this is the "
     "difference between a usable model and one that predicts the majority."),
    ("train/test protocol", "'5 fold cross-validation' with no held-out test "
     "set. The baselines it is compared against use TwiBot-22's official "
     "test split, which has a different class balance (29.4% vs 14.0% bots)."),
    ("edge definition", "Sect. 3.1.1 says followers, friends, retweets and "
     "mentions are all collapsed into one undirected relation, but not which "
     "relations exist in each corpus."),
    ("account age reference date", "Sect. 3.3 derives account age from 'the "
     "current date', which is not a fixed quantity. Harmless for a linear "
     "model (a constant shift is absorbed by the bias) but it makes the "
     "feature literally irreproducible."),
]
