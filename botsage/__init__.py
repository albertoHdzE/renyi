"""Replication of Deshmukh (2025), *Bot Detection in Social Media using
GraphSage and BERT* (SJSU Master's Project 1465, DOI 10.31979/etd.wb6h-3yd6).

The method is a three-stage pipeline: DistilBERT/BERT over a user's tweets gives
a 768-vector, an untrained GraphSAGE layer over a user graph gives a 128-vector,
the two are concatenated into 896 dimensions and classified by a linear SVM
under 5-fold cross-validation. Reported: **98.68%** on Cresci-15 and **74.62%**
on TwiBot-22.

There is no code release beyond one appendix listing, so everything here is
built from the text. Three things this replication establishes:

1. **The GraphSAGE layer is never trained** (Sect. 3.5, explicitly). An
   untrained ``SAGEConv(5, 128)`` is an affine map of ``[x_v || mean of N(v)]``,
   so its 128 columns span a subspace of dimension at most 10. ``checks.py``
   verifies this numerically, and
   ``experiments.suite_regularization_equivalence`` shows the apparent gain over
   those 10 numbers is a regularisation-geometry artefact that vanishes as the
   SVM penalty weakens.

2. **Every accuracy in Table 5 is below the majority-class baseline.**
   TwiBot-22 is 139,943 bots and 860,057 humans, so predicting "human" for
   everyone scores 86.01%; the paper reports 74.62% under 5-fold CV over that
   corpus. Verified from the released ``label.csv``.

3. **On Cresci-15 the graph branch carries almost nothing**, because only 0.12%
   of edges join two users that have metadata, leaving the neighbour mean
   exactly zero for 3,381 of 5,301 labelled users.

Deviations from the paper, and the ambiguities that forced a choice, are in
``docs/DISCREPANCIES_BOTSAGE.md``.

Typical use::

    from botsage import Config, load_dataset, run_experiment

    ds = load_dataset("cresci-15")
    res = run_experiment(ds, Config(), text_embeddings=te)
    print(res["metrics"]["accuracy"]["mean"])
"""

from .config import (Config, ROOT, PAPER_DIR, DATA_RAW, RESULTS, FIGURES,
                     PAPER, PAPER_TABLE1, PAPER_TABLE2, PAPER_TABLE3,
                     PAPER_TABLE4, PAPER_TABLE5, STATED_PARAMETERS,
                     INFERRED_PARAMETERS, TWIBOT22_LABEL_COUNTS)
from .data import (BotDataset, load_dataset, load_cresci15, load_twibot20,
                   load_twibot22, twibot22_baseline_report, FEATURE_NAMES)
from .sage import (SAGEConv, TrainedSAGE, sage_embeddings, mean_aggregate,
                   effective_input)
from .checks import run_all_checks
from .text import (TweetEncoder, clean_tweet, user_text_embeddings,
                   embedding_diagnostics, STOPWORDS)
from .pipeline import (build_embeddings, evaluate, run_experiment,
                       make_classifier, restrict_graph, expand_text)
from .experiments import (suite_replicate, suite_ablation,
                          suite_regularization_equivalence,
                          suite_seed_sensitivity, suite_trained_vs_untrained,
                          suite_graph_scope, suite_protocol,
                          results_to_frame, save_results, load_results)
from . import plots

__version__ = "1.0.0"

__all__ = [
    "Config", "ROOT", "PAPER_DIR", "DATA_RAW", "RESULTS", "FIGURES", "PAPER",
    "PAPER_TABLE1", "PAPER_TABLE2", "PAPER_TABLE3", "PAPER_TABLE4",
    "PAPER_TABLE5", "STATED_PARAMETERS", "INFERRED_PARAMETERS",
    "TWIBOT22_LABEL_COUNTS",
    "BotDataset", "load_dataset", "load_cresci15", "load_twibot20",
    "load_twibot22", "twibot22_baseline_report", "FEATURE_NAMES",
    "SAGEConv", "TrainedSAGE", "sage_embeddings", "mean_aggregate",
    "effective_input", "run_all_checks",
    "TweetEncoder", "clean_tweet", "user_text_embeddings",
    "embedding_diagnostics", "STOPWORDS",
    "build_embeddings", "evaluate", "run_experiment", "make_classifier",
    "restrict_graph", "expand_text",
    "suite_replicate", "suite_ablation", "suite_regularization_equivalence",
    "suite_seed_sensitivity", "suite_trained_vs_untrained",
    "suite_graph_scope", "suite_protocol", "results_to_frame", "save_results",
    "load_results", "plots", "__version__",
]
