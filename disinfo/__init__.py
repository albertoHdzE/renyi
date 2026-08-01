"""Replication of Lakzaei, Haghir Chehreghani & Bagheri (2024).

"Disinformation detection using graph neural networks: a survey",
*Artificial Intelligence Review* 57:52 -- https://doi.org/10.1007/s10462-024-10702-9

This is a survey, so it has no experiments and no code release, and "replicate
the results" has to mean something other than rerunning the authors' scripts.
This package takes it to mean three things:

1. **Reproduce the six figures.** All six are diagrams. They are encoded as data
   in ``taxonomy`` and rendered by ``plots``, so each figure is a rendering of a
   structure rather than a redrawing of a picture.

2. **Test the survey's own empirical claims.** Tables 1-2 are a meta-analysis of
   34 papers, and Sect. 5.3.2 draws four conclusions from them in prose without
   plotting anything. ``survey_data`` transcribes the tables and states those
   conclusions as predicates; ``survey_data.verify_claims`` evaluates them.
   Five of six hold. The sixth -- Sect. 7's claim that multiclass accuracy is
   "typically below 50%" -- is contradicted by the survey's own tables.

3. **Run the framework it describes.** Sect. 3.1 gives five GNN update rules
   (Eqs. 2-10) and Sect. 5.3 gives three graph constructions. ``layers``
   implements the equations, ``graphs`` the constructions, and ``pipeline``
   wires them into the four stages of Fig. 1, on LIAR, Twitter15, Twitter16,
   PHEME and CED.

Deviations from the printed equations, and contradictions inside the survey, are
recorded in ``docs/DISCREPANCIES_SURVEY.md``.

Typical use::

    from disinfo import Config, load_dataset, run_experiment

    ds = load_dataset("twitter15")
    res = run_experiment(ds, Config(graph="propagation", gnn="gat"))
    print(res["metrics"]["accuracy"])
"""

from .config import (Config, ROOT, DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES,
                     INFERRED_PARAMETERS, GNN_GRID, GRAPH_GRID)
from .data import (Cascade, Item, DisinfoDataset, load_dataset, load_liar,
                   load_twitter, load_pheme, load_ced, dataset_sizes,
                   CANONICAL_LABELS)
from .layers import (GCNLayer, GATLayer, GATv2Layer, SAGELayer, GINLayer,
                     LAYERS, add_self_loops, degree, scatter_sum, scatter_mean,
                     scatter_max, scatter_softmax)
from .checks import run_all_checks
from .features import (FeatureBundle, build_features, text_features,
                       syntactic_features, profile_features,
                       propagation_features, temporal_features)
from .graphs import (Graph, build_graph, build_graphs, knn_similarity_graph,
                     threshold_similarity_graph, attribute_graph,
                     propagation_graph, heterogeneous_graph, graph_stats)
from .models import GNNEncoder, NodeClassifier, GraphClassifier, build_model
from .metrics import classification_metrics
from .pipeline import run_experiment, run_seeds, make_splits, set_seed, collate
from .experiments import (FeatureCache, suite_gnn_comparison,
                          suite_graph_comparison, suite_feature_ablation,
                          suite_liar_granularity, suite_pheme_split,
                          results_to_frame, save_results, load_results)
from .survey_data import (METHODS, DATASETS, EXAMPLES, PAPER_CLAIMS,
                          TRANSCRIPTION_NOTES, methods_table, datasets_table,
                          examples_table, long_results, verify_claims,
                          verify_table3)
from . import taxonomy, plots

__version__ = "1.0.0"

PAPER = {
    "title": "Disinformation detection using graph neural networks: a survey",
    "authors": "Batool Lakzaei, Mostafa Haghir Chehreghani, Alireza Bagheri",
    "venue": "Artificial Intelligence Review (2024) 57:52",
    "doi": "10.1007/s10462-024-10702-9",
    "accepted": "2024-01-04",
    "kind": "survey (no original experiments, no code release)",
}

__all__ = [
    "Config", "ROOT", "DATA_RAW", "DATA_PROCESSED", "RESULTS", "FIGURES",
    "INFERRED_PARAMETERS", "GNN_GRID", "GRAPH_GRID", "PAPER",
    "Cascade", "Item", "DisinfoDataset", "load_dataset", "load_liar",
    "load_twitter", "load_pheme", "load_ced", "dataset_sizes",
    "CANONICAL_LABELS",
    "GCNLayer", "GATLayer", "GATv2Layer", "SAGELayer", "GINLayer", "LAYERS",
    "add_self_loops", "degree", "scatter_sum", "scatter_mean", "scatter_max",
    "scatter_softmax", "run_all_checks",
    "FeatureBundle", "build_features", "text_features", "syntactic_features",
    "profile_features", "propagation_features", "temporal_features",
    "Graph", "build_graph", "build_graphs", "knn_similarity_graph",
    "threshold_similarity_graph", "attribute_graph", "propagation_graph",
    "heterogeneous_graph", "graph_stats",
    "GNNEncoder", "NodeClassifier", "GraphClassifier", "build_model",
    "classification_metrics",
    "run_experiment", "run_seeds", "make_splits", "set_seed", "collate",
    "FeatureCache", "suite_gnn_comparison", "suite_graph_comparison",
    "suite_feature_ablation", "suite_liar_granularity", "suite_pheme_split",
    "results_to_frame", "save_results", "load_results",
    "METHODS", "DATASETS", "EXAMPLES", "PAPER_CLAIMS", "TRANSCRIPTION_NOTES",
    "methods_table", "datasets_table", "examples_table", "long_results",
    "verify_claims", "verify_table3",
    "taxonomy", "plots", "__version__",
]
