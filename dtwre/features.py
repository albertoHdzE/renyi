"""Feature construction for each method compared in Table 1 (Section 3.2).

Every method feeds the *same* GraphSAGE + MLP pipeline; only the input node
feature matrix changes. That isolates the contribution of each feature family,
which is what Section 3.5 calls "a balanced comparison of the contributions of
various features to overall model performance".

    (1) node_degree    -- degree scalar
    (2) node_pagerank  -- PageRank scalar
    (3) node2vec       -- 64-dim embedding
    (4) renyi_static   -- Renyi entropy on the aggregated graph, no time weight
    (5) dtwre          -- LNE + time-weighted entropy + global DTWRE + Node2Vec
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from .entropy import (local_node_entropy, global_timestep_entropy,
                      dtwre_series, time_weight)
from .embeddings import node2vec

__all__ = ["entropy_features", "build_features", "FeatureBundle"]


class FeatureBundle:
    """Feature matrix plus the intermediate quantities the notebook plots."""

    def __init__(self, X, columns, extras=None):
        self.X = X
        self.columns = columns
        self.extras = extras or {}

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    def __repr__(self):
        return f"FeatureBundle(n={self.X.shape[0]}, d={self.X.shape[1]}, cols={self.columns})"


def entropy_features(snapshots, num_nodes, alpha=0.6, lam=1.2,
                     metric="degree", normalisation="local"):
    """Per-node entropy features across the temporal snapshots.

    Returns ``(lne_matrix, global_series, dtwre_global, lne_last, lne_tw)``:

    * ``lne_matrix``   -- (T, N) Local Node Entropy, Eq. 1
    * ``global_series``-- (T,)   sum over nodes per snapshot, Eq. 2
    * ``dtwre_global`` -- (T,)   time-weighted global entropy, Eq. 3
    * ``lne_last``     -- (N,)   LNE at the most recent training snapshot
    * ``lne_tw``       -- (N,)   node-level time-weighted entropy,
      ``sum_k w(T - t_k) * H_a(v, t_k)``

    The paper concatenates "LNE, DTWRE, Node2Vec_embeddings" (Algorithm 1
    line 3). ``dtwre_global`` is a single scalar per time step, so as a node
    feature it is constant across nodes and can only shift the bias; the
    node-level ``lne_tw`` is the quantity that actually carries the temporal
    signal per node. Both are included -- see docs/DISCREPANCIES.md.
    """
    T = len(snapshots)
    lne_matrix = np.zeros((T, num_nodes), dtype=np.float64)
    for t, g in enumerate(snapshots):
        lne = local_node_entropy(g, alpha=alpha, metric=metric,
                                 normalisation=normalisation)
        for v, h in lne.items():
            if v < num_nodes:
                lne_matrix[t, v] = h

    global_series = np.array([global_timestep_entropy(
        {i: lne_matrix[t, i] for i in range(num_nodes)}) for t in range(T)])
    dtwre_global = dtwre_series(global_series, lam=lam)

    lne_last = lne_matrix[-1] if T else np.zeros(num_nodes)

    # Node-level DTWRE: same exponential decay as Eq. 4, applied per node.
    if T:
        deltas = (T - 1) - np.arange(T)          # age of each snapshot
        w = time_weight(deltas, lam)             # (T,)
        lne_tw = (w[:, None] * lne_matrix).sum(axis=0)
    else:
        lne_tw = np.zeros(num_nodes)

    return lne_matrix, global_series, dtwre_global, lne_last, lne_tw


def build_features(method, train_graph, snapshots, num_nodes, cfg,
                   cached_embedding=None, seed=None):
    """Assemble the node feature matrix for one method.

    ``train_graph`` is the aggregated graph over the training period (used for
    message passing and for the static baselines); ``snapshots`` are the
    per-window subgraphs of the same period.
    """
    seed = cfg.seed if seed is None else seed
    extras = {}

    if method == "node_degree":
        deg = np.zeros(num_nodes)
        for n, d in train_graph.degree():
            if n < num_nodes:
                deg[n] = d
        X, cols = deg.reshape(-1, 1), ["degree"]

    elif method == "node_pagerank":
        pr = np.zeros(num_nodes)
        if train_graph.number_of_edges():
            for n, v in nx.pagerank(train_graph).items():
                if n < num_nodes:
                    pr[n] = v
        X, cols = pr.reshape(-1, 1), ["pagerank"]

    elif method == "node2vec":
        emb = (cached_embedding if cached_embedding is not None else
               node2vec(train_graph, num_nodes, dim=cfg.embedding_dim,
                        num_walks=cfg.num_walks, walk_length=cfg.walk_length,
                        window=cfg.context_window, p=cfg.p, q=cfg.q,
                        epochs=cfg.n2v_epochs, lr=cfg.n2v_lr,
                        negative=cfg.n2v_negative, seed=seed))
        X = emb
        cols = [f"n2v_{i}" for i in range(emb.shape[1])]

    elif method == "renyi_static":
        lne = local_node_entropy(train_graph, alpha=cfg.alpha,
                                 metric=cfg.prob_metric,
                                 normalisation=cfg.prob_normalisation)
        vec = np.zeros(num_nodes)
        for v, h in lne.items():
            if v < num_nodes:
                vec[v] = h
        X, cols = vec.reshape(-1, 1), ["renyi_static"]
        extras["lne_static"] = vec

    elif method == "dtwre":
        lne_m, gser, dglob, lne_last, lne_tw = entropy_features(
            snapshots, num_nodes, alpha=cfg.alpha, lam=cfg.lam,
            metric=cfg.prob_metric, normalisation=cfg.prob_normalisation)
        emb = (cached_embedding if cached_embedding is not None else
               node2vec(train_graph, num_nodes, dim=cfg.embedding_dim,
                        num_walks=cfg.num_walks, walk_length=cfg.walk_length,
                        window=cfg.context_window, p=cfg.p, q=cfg.q,
                        epochs=cfg.n2v_epochs, lr=cfg.n2v_lr,
                        negative=cfg.n2v_negative, seed=seed))
        dtwre_col = np.full(num_nodes, dglob[-1] if len(dglob) else 0.0)
        X = np.column_stack([lne_last, lne_tw, dtwre_col, emb])
        cols = (["lne_last", "lne_time_weighted", "dtwre_global"] +
                [f"n2v_{i}" for i in range(emb.shape[1])])
        extras.update({"lne_matrix": lne_m, "global_series": gser,
                       "dtwre_global": dglob, "lne_last": lne_last,
                       "lne_tw": lne_tw})

    else:
        raise ValueError(f"unknown method {method!r}")

    # Section 3.2.2: "normalize ... using MinMaxScaler" (applied to all
    # feature blocks so entropy and embedding scales are commensurate).
    X = MinMaxScaler().fit_transform(np.asarray(X, dtype=np.float64))
    return FeatureBundle(X.astype(np.float32), cols, extras)
