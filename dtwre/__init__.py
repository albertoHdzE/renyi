"""Replication of Tong, Xu, Zhang & Xu (2025), *Entropy* 27(5), 516.

"Public Opinion Propagation Prediction Model Based on Dynamic Time-Weighted
Rényi Entropy and Graph Neural Network" -- https://doi.org/10.3390/e27050516

The paper has no official code release; this package is an independent
reimplementation from the published equations, Algorithm 1 and hyperparameters.
Points where the paper is ambiguous or internally inconsistent are documented
in ``docs/DISCREPANCIES.md`` and flagged in ``config.INFERRED_PARAMETERS``.

Typical use::

    from dtwre import Config, load_dataset, run_experiment

    tg = load_dataset("collegemsg")
    res = run_experiment(tg, Config(), method="dtwre")
    print(res["test"]["auc"])
"""

from .config import (Config, ALPHA_GRID, LAMBDA_GRID, NEG_RATIO_GRID,
                     WINDOW_GRID, BASELINES, BASELINE_LABELS, METRICS,
                     PAPER_TABLE1, PAPER_TABLE2, PAPER_TABLE3,
                     PAPER_ALPHA_AUC, PAPER_ALPHA_ACC, INFERRED_PARAMETERS,
                     ROOT, DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES)
from .data import (TemporalGraph, load_dataset, load_collegemsg,
                   load_weibo_ced, chronological_split, build_snapshots,
                   augment_graph, sample_negatives)
from .entropy import (renyi_entropy, local_node_entropy,
                      global_timestep_entropy, dtwre_series, time_weight,
                      static_renyi_entropy, node_probabilities)
from .embeddings import node2vec, random_walks
from .features import build_features, entropy_features, FeatureBundle
from .models import LinkPredictor, GraphSAGE, MLPPredictor
from .metrics import binary_metrics
from .pipeline import run_experiment, prepare_split, set_seed

__version__ = "1.0.0"

__all__ = [
    "Config", "ALPHA_GRID", "LAMBDA_GRID", "NEG_RATIO_GRID", "WINDOW_GRID",
    "BASELINES", "BASELINE_LABELS", "METRICS", "PAPER_TABLE1", "PAPER_TABLE2",
    "PAPER_TABLE3", "PAPER_ALPHA_AUC", "PAPER_ALPHA_ACC",
    "INFERRED_PARAMETERS", "ROOT", "DATA_RAW", "DATA_PROCESSED", "RESULTS",
    "FIGURES", "TemporalGraph", "load_dataset", "load_collegemsg",
    "load_weibo_ced", "chronological_split", "build_snapshots",
    "augment_graph", "sample_negatives", "renyi_entropy",
    "local_node_entropy", "global_timestep_entropy", "dtwre_series",
    "time_weight", "static_renyi_entropy", "node_probabilities", "node2vec",
    "random_walks", "build_features", "entropy_features", "FeatureBundle",
    "LinkPredictor", "GraphSAGE", "MLPPredictor", "binary_metrics",
    "run_experiment", "prepare_split", "set_seed", "__version__",
]
