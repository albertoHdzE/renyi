"""GraphSAGE encoder and MLP link predictor (Sections 2.3 and 3.3).

    Eq. 5  h_u^(k) = sigma( W_k . Aggregate({ h_v^(k-1), v in N(u) }) )
    Eq. 6  h_v^(n) = [h_v,1^n, h_v,2^n, ..., h_v,d^n]
    Eq. 7  L = -sum_{(u,v) in E} y log(yhat) + (1-y) log(1-yhat)

The paper specifies mean aggregation and ReLU. Eq. 5 shows only the aggregated
neighbourhood term; canonical GraphSAGE also concatenates the node's own
previous representation, which ``concat_self`` controls (default ``True``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

__all__ = ["mean_aggregation_matrix", "GraphSAGE", "MLPPredictor", "LinkPredictor"]


def mean_aggregation_matrix(graph: nx.Graph, num_nodes: int, fanout=None,
                            rng=None, device="cpu") -> torch.Tensor:
    """Row-normalised sparse adjacency implementing mean aggregation.

    ``fanout`` reproduces GraphSAGE's neighbour sampling: each node keeps at
    most ``fanout`` uniformly sampled neighbours. ``None`` uses every
    neighbour, which is exact and affordable at the scale of these datasets.
    """
    rng = rng or np.random.default_rng(0)
    rows, cols = [], []
    for u in graph.nodes():
        if u >= num_nodes:
            continue
        nbrs = [v for v in graph.neighbors(u) if v < num_nodes]
        if not nbrs:
            continue
        if fanout is not None and len(nbrs) > fanout:
            nbrs = rng.choice(nbrs, size=fanout, replace=False).tolist()
        rows.extend([u] * len(nbrs))
        cols.extend(nbrs)

    if not rows:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros(0),
            (num_nodes, num_nodes)).coalesce().to(device)

    rows_a = np.asarray(rows, dtype=np.int64)
    cols_a = np.asarray(cols, dtype=np.int64)
    deg = np.bincount(rows_a, minlength=num_nodes).astype(np.float32)
    vals = 1.0 / deg[rows_a]

    idx = torch.as_tensor(np.stack([rows_a, cols_a]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(vals, dtype=torch.float),
                                   (num_nodes, num_nodes)).coalesce().to(device)


class SAGELayer(nn.Module):
    """One mean-aggregation GraphSAGE layer (Eq. 5)."""

    def __init__(self, in_dim: int, out_dim: int, concat_self: bool = True):
        super().__init__()
        self.concat_self = concat_self
        self.lin = nn.Linear(in_dim * 2 if concat_self else in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = torch.sparse.mm(adj, x)
        h = torch.cat([x, agg], dim=1) if self.concat_self else agg
        return self.lin(h)


class GraphSAGE(nn.Module):
    """Stacked GraphSAGE layers producing the final node embedding (Eq. 6)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, concat_self: bool = True):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for k in range(num_layers):
            self.layers.append(SAGELayer(dims[k], dims[k + 1], concat_self))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < len(self.layers) - 1:
                h = torch.relu(h)          # sigma = ReLU (Section 2.3)
                h = self.dropout(h)
        return h


class MLPPredictor(nn.Module):
    """MLP scoring a node pair (Section 3.3).

    The paper states only that an MLP of several fully connected layers maps
    node features to a link probability. The pair is summarised by the
    element-wise (Hadamard) product, which keeps the score symmetric in
    ``(u, v)`` as an undirected link prediction requires.
    """

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(dim // 2, 8)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2 or 1), nn.ReLU(),
            nn.Linear(hidden // 2 or 1, 1),
        )

    def forward(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        z = h[pairs[:, 0]] * h[pairs[:, 1]]
        return self.net(z).squeeze(-1)      # logits


class LinkPredictor(nn.Module):
    """GraphSAGE encoder + MLP decoder, trained with BCE (Eq. 7)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, concat_self: bool = True):
        super().__init__()
        self.encoder = GraphSAGE(in_dim, hidden_dim, num_layers, dropout,
                                 concat_self)
        self.decoder = MLPPredictor(hidden_dim)

    def forward(self, x, adj, pairs):
        return self.decoder(self.encoder(x, adj), pairs)
