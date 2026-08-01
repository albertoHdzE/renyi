"""End-to-end experiment pipeline (Algorithm 1 + Sections 3.1-3.4).

One call to :func:`run_experiment` performs the whole Figure 4 workflow:
data processing -> feature engineering -> GraphSAGE training -> evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

from .config import Config
from .data import (chronological_split, build_snapshots, augment_graph,
                   sample_negatives)
from .features import build_features
from .models import LinkPredictor, mean_aggregation_matrix
from .metrics import binary_metrics

__all__ = ["prepare_split", "run_experiment", "set_seed"]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _unique_pairs(edges: np.ndarray) -> set:
    """Undirected, de-duplicated node pairs from a timestamped edge array."""
    if len(edges) == 0:
        return set()
    a = np.minimum(edges[:, 0], edges[:, 1])
    b = np.maximum(edges[:, 0], edges[:, 1])
    return set(zip(a.tolist(), b.tolist()))


def prepare_split(tg, cfg: Config):
    """Chronological split, snapshots, and positive/negative sample sets.

    The task is link *formation*: predicting pairs that become connected in a
    future window. Two properties are required for that to be measured
    honestly, and both are enforced here.

    1. **Disjoint message passing and supervision.** The encoder never sees an
       edge it is asked to predict. The training period is itself cut at
       ``mp_frac`` of its duration: earlier edges build the graph the model
       propagates over, later edges become the supervision positives. Training
       on the same edges used for message passing instead teaches the model to
       recognise *existing* adjacency, which inverts on unseen future links
       (measured: test AUC 0.36, i.e. below chance).

    2. **Positives are genuinely new.** Any pair already present in the
       message-passing graph is removed from the val/test positives, so the
       model cannot score by recalling a link it has already propagated over.

    All three evaluations share the same message-passing graph and feature
    matrix, so the only thing that differs across splits is *which future*
    is being predicted.
    """
    rng = np.random.default_rng(cfg.seed)
    train_e, val_e, test_e, cuts = chronological_split(
        tg, cfg.train_frac, cfg.val_frac, cfg.split_by)

    # Cut the training period again: [t0, mp_cut] builds the graph,
    # (mp_cut, cut_train] supervises it.
    ts = tg.timestamps
    t0 = float(ts.min())
    mp_cut = t0 + cfg.mp_frac * (cuts[0] - t0)
    mp_e = train_e[train_e[:, 2] <= mp_cut]
    sup_e = train_e[train_e[:, 2] > mp_cut]

    mp_pairs = _unique_pairs(mp_e)
    if cfg.new_links_only:
        sup_pairs = _unique_pairs(sup_e) - mp_pairs
        val_pairs = _unique_pairs(val_e) - mp_pairs - sup_pairs
        test_pairs = _unique_pairs(test_e) - mp_pairs - sup_pairs
    else:
        sup_pairs = _unique_pairs(sup_e)
        val_pairs = _unique_pairs(val_e)
        test_pairs = _unique_pairs(test_e)
    all_pairs = (mp_pairs | _unique_pairs(sup_e) | _unique_pairs(val_e)
                 | _unique_pairs(test_e))

    # Message-passing graph: only edges observed before the supervision cut.
    train_graph = nx.Graph()
    train_graph.add_nodes_from(range(tg.num_nodes))
    if len(mp_e):
        train_graph.add_edges_from(mp_e[:, :2].tolist())
    train_graph.remove_edges_from(nx.selfloop_edges(train_graph))

    # Temporal snapshots of the message-passing period drive entropy features.
    snapshots = build_snapshots(mp_e, cfg.window_seconds, tg.num_nodes)
    if cfg.augment:
        snapshots = [augment_graph(g, cfg.add_frac, cfg.del_frac,
                                   cfg.drop_isolated, rng) for g in snapshots]

    def with_negatives(pos_set, seed_offset):
        pos = np.array(sorted(pos_set), dtype=np.int64) if pos_set else \
            np.empty((0, 2), dtype=np.int64)
        neg = sample_negatives(pos, tg.num_nodes, all_pairs, cfg.neg_ratio,
                               np.random.default_rng(cfg.seed + seed_offset))
        pairs = np.vstack([pos, neg]) if len(neg) else pos
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return pairs, labels

    return {
        "train_graph": train_graph,
        "snapshots": snapshots,
        "cuts": cuts,
        "mp_cut": mp_cut,
        "n_mp_edges": int(len(mp_e)),
        "n_train_edges": int(len(train_e)),
        "n_val_edges": int(len(val_e)),
        "n_test_edges": int(len(test_e)),
        "n_pos": {"train": len(sup_pairs), "val": len(val_pairs),
                  "test": len(test_pairs)},
        "train": with_negatives(sup_pairs, 1),
        "val": with_negatives(val_pairs, 2),
        "test": with_negatives(test_pairs, 3),
    }


def run_experiment(tg, cfg: Config, method: str = "dtwre", device: str = "cpu",
                   track_history: bool = False, cached_embedding=None,
                   split=None, verbose: bool = False) -> dict:
    """Train and evaluate one (dataset, method, hyperparameter) combination.

    Returns a dict with ``test`` / ``val`` metric dicts, the per-epoch
    ``history`` when requested, and the feature ``extras`` used for plotting.
    """
    set_seed(cfg.seed)
    split = split or prepare_split(tg, cfg)

    bundle = build_features(method, split["train_graph"], split["snapshots"],
                            tg.num_nodes, cfg, cached_embedding=cached_embedding)

    x = torch.as_tensor(bundle.X, dtype=torch.float, device=device)
    adj = mean_aggregation_matrix(split["train_graph"], tg.num_nodes,
                                  fanout=cfg.fanout,
                                  rng=np.random.default_rng(cfg.seed),
                                  device=device)

    def as_tensors(key):
        pairs, labels = split[key]
        return (torch.as_tensor(pairs, dtype=torch.long, device=device),
                torch.as_tensor(labels, dtype=torch.float, device=device))

    tr_pairs, tr_y = as_tensors("train")
    va_pairs, va_y = as_tensors("val")
    te_pairs, te_y = as_tensors("test")

    model = LinkPredictor(bundle.dim, cfg.hidden_dim, cfg.num_layers,
                          cfg.dropout, cfg.concat_self).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()          # Eq. 7

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, adj, tr_pairs)
        loss = criterion(logits, tr_y)
        loss.backward()
        opt.step()

        if track_history or verbose:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(x, adj, te_pairs)).cpu().numpy()
            m = binary_metrics(te_y.cpu().numpy(), probs)
            m["epoch"] = epoch
            m["loss"] = float(loss.item())
            history.append(m)
            if verbose and epoch % 20 == 0:
                print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  "
                      f"AUC {m['auc']:.4f}")

    model.eval()
    with torch.no_grad():
        emb = model.encoder(x, adj)
        te_prob = torch.sigmoid(model.decoder(emb, te_pairs)).cpu().numpy()
        va_prob = torch.sigmoid(model.decoder(emb, va_pairs)).cpu().numpy()

    return {
        "method": method,
        "test": binary_metrics(te_y.cpu().numpy(), te_prob),
        "val": binary_metrics(va_y.cpu().numpy(), va_prob),
        "history": history,
        "extras": bundle.extras,
        "columns": bundle.columns,
        "feature_dim": bundle.dim,
        "test_pairs": split["test"][0],
        "test_scores": te_prob,
        "test_labels": te_y.cpu().numpy(),
        "node_embeddings": emb.detach().cpu().numpy(),
        "split_sizes": {k: split[k] for k in
                        ("n_train_edges", "n_val_edges", "n_test_edges")},
        "config": cfg.to_dict(),
    }
