"""The four stages of Fig. 1, wired together.

``run_experiment`` executes the general framework end to end:

    feature extraction  ->  graph construction  ->  GNN  ->  classification

and returns test metrics comparable with the Performance column of Tables 1-2.
``run_seeds`` repeats it and reports mean +/- std, which the survey's tables do
not carry -- they are point estimates with no variance, so any comparison to
them must state the spread on our side.
"""

from __future__ import annotations

import random
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from .config import Config
from .data import DisinfoDataset
from .features import build_features
from .graphs import build_graph, build_graphs
from .metrics import classification_metrics
from .models import build_model

__all__ = ["set_seed", "make_splits", "collate", "run_experiment", "run_seeds"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def make_splits(ds: DisinfoDataset, cfg: Config):
    """Train/val/test indices.

    ``cfg.split="event"`` holds out whole PHEME events. This is the harder and
    more honest protocol for that corpus: threads within one breaking-news event
    share entities and phrasing, so a random split lets the model recognise the
    event rather than the veracity. None of the surveyed papers state which they
    used, which is a plausible source of PHEME's 0.694-0.887 spread in Tables
    1-2.
    """
    y = ds.y
    idx = np.arange(len(ds))

    if cfg.split == "event":
        events = np.array([it.meta.get("event", "?") for it in ds.items])
        uniq = sorted(set(events))
        if len(uniq) < 3:
            raise ValueError(f"event split needs >=3 events, found {len(uniq)}")
        rng = np.random.default_rng(cfg.seed)
        order = list(rng.permutation(uniq))
        n_test = max(1, round(len(order) * cfg.test_frac))
        n_val = max(1, round(len(order) * cfg.val_frac))
        test_ev, val_ev = set(order[:n_test]), set(order[n_test:n_test + n_val])
        test = idx[np.isin(events, list(test_ev))]
        val = idx[np.isin(events, list(val_ev))]
        train = idx[~np.isin(events, list(test_ev | val_ev))]
        return train, val, test

    train, test = train_test_split(idx, test_size=cfg.test_frac,
                                   stratify=y, random_state=cfg.seed)
    rel_val = cfg.val_frac / (1.0 - cfg.test_frac)
    train, val = train_test_split(train, test_size=rel_val,
                                  stratify=y[train], random_state=cfg.seed)
    return train, val, test


# --------------------------------------------------------------------------
# batching for graph classification
# --------------------------------------------------------------------------

def collate(graphs: list, device: str = "cpu"):
    """Merge graphs into one disconnected graph, the standard GNN batching trick.

    Node indices are offset per graph so no message can cross a graph boundary;
    ``batch`` records the origin of each node for the readout.
    """
    xs, eis, batch, ys = [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g.x)
        if g.num_edges:
            eis.append(g.edge_index + offset)
        batch.append(torch.full((g.num_nodes,), i, dtype=torch.long))
        if g.y is not None:
            ys.append(g.y)
        offset += g.num_nodes

    x = torch.cat(xs).to(device)
    ei = (torch.cat(eis, dim=1) if eis
          else torch.zeros((2, 0), dtype=torch.long)).to(device)
    return (x, ei, torch.cat(batch).to(device), len(graphs),
            torch.stack(ys).to(device) if ys else None)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def _train_node(graph, splits, cfg, n_classes, quiet):
    train, val, test = splits
    dev = cfg.device
    x, ei, y = graph.x.to(dev), graph.edge_index.to(dev), graph.y.to(dev)
    tr = torch.as_tensor(train, device=dev)
    va = torch.as_tensor(val, device=dev)
    te = torch.as_tensor(test, device=dev)

    model = build_model(x.size(1), n_classes, cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    best, best_state, waited = np.inf, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, ei)[tr], y[tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = F.cross_entropy(model(x, ei)[va], y[va]).item()
        if vloss < best - 1e-5:
            best, waited = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, ei)
    return model, logits[te].cpu(), y[te].cpu()


def _train_graph(graphs, splits, cfg, n_classes, quiet):
    train, val, test = splits
    dev = cfg.device
    tr_g = [graphs[i] for i in train]
    va_batch = collate([graphs[i] for i in val], dev)
    te_batch = collate([graphs[i] for i in test], dev)

    model = build_model(graphs[0].x.size(1), n_classes, cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)

    best, best_state, waited = np.inf, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(tr_g))
        for s in range(0, len(order), cfg.batch_size):
            chunk = [tr_g[i] for i in order[s:s + cfg.batch_size]]
            x, ei, batch, n, y = collate(chunk, dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(x, ei, batch, n), y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            x, ei, batch, n, y = va_batch
            vloss = F.cross_entropy(model(x, ei, batch, n), y).item()
        if vloss < best - 1e-5:
            best, waited = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x, ei, batch, n, y = te_batch
        logits = model(x, ei, batch, n)
    return model, logits.cpu(), y.cpu()


def run_experiment(ds: DisinfoDataset, cfg: Config, quiet: bool = True,
                   features=None) -> dict:
    """One end-to-end pass of Fig. 1. Returns metrics and provenance.

    ``features`` lets a caller reuse a ``FeatureBundle`` across configurations
    that do not change stage 1 -- every GNN and graph-type sweep, in practice.
    TF-IDF plus SVD over PHEME or CED costs more than the GNN training does, so
    recomputing it per configuration is pure waste.
    """
    set_seed(cfg.seed)
    splits = make_splits(ds, cfg)
    train_idx = splits[0]

    # Stage 1
    bundle = features if features is not None else build_features(ds, cfg, train_idx)

    # Stage 2
    node_level = cfg.graph in ("similarity", "attribute")
    if node_level:
        graph = build_graph(ds, bundle.X, cfg)
        n_nodes, n_edges = graph.num_nodes, graph.num_edges
    else:
        graph = build_graphs(ds, bundle.X, cfg)
        n_nodes = sum(g.num_nodes for g in graph)
        n_edges = sum(g.num_edges for g in graph)

    # Stages 3-4
    n_classes = len(ds.label_names)
    trainer = _train_node if node_level else _train_graph
    model, logits, y_true = trainer(graph, splits, cfg, n_classes, quiet)

    metrics = classification_metrics(y_true.numpy(), logits.numpy(),
                                     ds.label_names)
    return {
        "dataset": ds.name, "config": asdict(cfg), "metrics": metrics,
        "n_train": len(splits[0]), "n_val": len(splits[1]),
        "n_test": len(splits[2]),
        "graph": {"n_nodes": n_nodes, "n_edges": n_edges,
                  "feature_dim": bundle.dim},
        "feature_blocks": {k: v.stop - v.start for k, v in bundle.blocks.items()},
    }


def run_seeds(ds: DisinfoDataset, cfg: Config, n_seeds: int | None = None,
              quiet: bool = True) -> dict:
    """Repeat ``run_experiment`` over seeds and summarise.

    Stage 1 is recomputed per seed on purpose: the split changes, and the
    vocabulary must be fitted on that seed's training rows to avoid leakage.
    """
    n_seeds = n_seeds or cfg.n_seeds
    runs = [run_experiment(ds, cfg.with_(seed=s), quiet=quiet)
            for s in range(n_seeds)]

    keys = [k for k, v in runs[0]["metrics"].items() if isinstance(v, float)]
    summary = {}
    for k in keys:
        vals = np.array([r["metrics"][k] for r in runs], dtype=float)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                      "values": vals.tolist()}
    if not quiet:
        acc = summary["accuracy"]
        print(f"{ds.name:10s} {cfg.graph:12s} {cfg.gnn:6s} "
              f"ACC {acc['mean']:.3f} +/- {acc['std']:.3f}")
    return {"dataset": ds.name, "config": asdict(cfg), "n_seeds": n_seeds,
            "summary": summary, "runs": runs}
