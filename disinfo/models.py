"""Stages 3 and 4 of Fig. 1: the GNN encoder and the classifier.

``GNNEncoder`` stacks any of the five layers of ``layers.py`` and produces the
per-node embedding of Fig. 1's third stage. ``NodeClassifier`` and
``GraphClassifier`` add the fourth stage for the two task shapes the survey's
graph taxonomy implies (see ``graphs.py``): node-level for similarity graphs,
graph-level for propagation and heterogeneous graphs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .layers import (GATLayer, GATv2Layer, GCNLayer, GINLayer, SAGELayer,
                     scatter_max, scatter_mean, scatter_sum)

__all__ = ["GNNEncoder", "NodeClassifier", "GraphClassifier", "build_model",
           "readout"]


def _make_layer(name: str, in_dim: int, out_dim: int, cfg: Config,
                last: bool) -> nn.Module:
    if name == "gcn":
        return GCNLayer(in_dim, out_dim, normalization=cfg.gcn_normalization)
    if name in ("gat", "gatv2"):
        cls = GATLayer if name == "gat" else GATv2Layer
        # The original GAT concatenates heads on hidden layers and averages on
        # the output layer; `out_dim` is per-head so total width stays hidden_dim.
        if last:
            return cls(in_dim, out_dim, heads=cfg.heads, concat=False,
                       dropout=cfg.dropout)
        per_head = max(out_dim // cfg.heads, 1)
        return cls(in_dim, per_head, heads=cfg.heads, concat=True,
                   dropout=cfg.dropout)
    if name == "sage":
        return SAGELayer(in_dim, out_dim, aggregator=cfg.sage_aggregator)
    if name == "gin":
        return GINLayer(in_dim, out_dim, train_eps=cfg.gin_train_eps)
    raise ValueError(f"unknown gnn {name!r}")


def _out_dim(name: str, hidden: int, cfg: Config, last: bool) -> int:
    if name in ("gat", "gatv2") and not last:
        return max(hidden // cfg.heads, 1) * cfg.heads
    return hidden


class GNNEncoder(nn.Module):
    """Stage 3: L message-passing layers producing one embedding per node.

    ReLU and dropout sit between layers, not after the last one -- the final
    embedding is what the classifier consumes and what a reader would want to
    inspect, so it is left un-rectified.
    """

    def __init__(self, in_dim: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        d = in_dim
        for i in range(cfg.num_layers):
            last = i == cfg.num_layers - 1
            self.layers.append(_make_layer(cfg.gnn, d, cfg.hidden_dim, cfg, last))
            d = _out_dim(cfg.gnn, cfg.hidden_dim, cfg, last)
        self.out_dim = d

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.cfg.dropout, training=self.training)
        return h


def readout(h: torch.Tensor, batch: torch.Tensor, n_graphs: int,
            mode: str = "mean") -> torch.Tensor:
    """Pool node embeddings into one vector per graph.

    ``batch[i]`` gives the graph index of node ``i``. ``"root"`` returns the
    source post's embedding only -- the survey notes that "the root node holds
    significant importance in rumor detection" (Sect. 5.3.2, on L-GAT), and node
    0 of every propagation graph is the source post by construction.
    """
    if mode == "mean":
        return scatter_mean(h, batch, n_graphs)
    if mode == "sum":
        return scatter_sum(h, batch, n_graphs)
    if mode == "max":
        return scatter_max(h, batch, n_graphs)
    if mode == "root":
        first = torch.zeros(n_graphs, dtype=torch.long, device=h.device)
        order = torch.argsort(batch, stable=True)
        counts = torch.bincount(batch, minlength=n_graphs)
        first[counts > 0] = order[(torch.cumsum(counts, 0) - counts)[counts > 0]]
        return h.index_select(0, first)
    raise ValueError(f"unknown readout {mode!r}")


def _head(in_dim: int, n_classes: int, cfg: Config) -> nn.Module:
    if cfg.classifier == "linear":
        return nn.Linear(in_dim, n_classes)
    return nn.Sequential(
        nn.Linear(in_dim, cfg.hidden_dim), nn.ReLU(),
        nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, n_classes))


class NodeClassifier(nn.Module):
    """Similarity/attribute graph: classify every item as a node in one graph."""

    def __init__(self, in_dim: int, n_classes: int, cfg: Config):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, cfg)
        self.head = _head(self.encoder.out_dim, n_classes, cfg)

    def forward(self, x, edge_index):
        return self.head(self.encoder(x, edge_index))


class GraphClassifier(nn.Module):
    """Propagation/heterogeneous graph: classify each cascade as a whole."""

    def __init__(self, in_dim: int, n_classes: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = GNNEncoder(in_dim, cfg)
        self.head = _head(self.encoder.out_dim, n_classes, cfg)

    def forward(self, x, edge_index, batch, n_graphs):
        h = self.encoder(x, edge_index)
        return self.head(readout(h, batch, n_graphs, self.cfg.readout))


def build_model(in_dim: int, n_classes: int, cfg: Config) -> nn.Module:
    node_level = cfg.graph in ("similarity", "attribute")
    cls = NodeClassifier if node_level else GraphClassifier
    return cls(in_dim, n_classes, cfg)
