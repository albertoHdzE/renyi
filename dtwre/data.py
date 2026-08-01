"""Dataset loading and preprocessing (Section 3.1 of the paper).

Two datasets are supported:

* **CollegeMsg** -- SNAP temporal network of private messages, 1899 nodes /
  59835 timestamped edges spanning 193.7 days.
* **Weibo (CED)** -- Chinese rumour repost cascades from the Sina Weibo
  misinformation reporting platform. A propagation network is reconstructed by
  linking each reposting user to the user they reposted from.

Preprocessing follows Section 3.1: timestamp standardisation, temporal window
segmentation, node/edge perturbation augmentation, isolated-node removal and a
strictly chronological 80/10/10 split.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import networkx as nx

from .config import DATA_RAW

__all__ = [
    "TemporalGraph",
    "load_collegemsg",
    "load_weibo_ced",
    "load_dataset",
    "chronological_split",
    "build_snapshots",
    "augment_graph",
    "sample_negatives",
]


@dataclass
class TemporalGraph:
    """Timestamped edge list with contiguous integer node ids."""

    edges: np.ndarray          # (m, 3) int64 -- src, dst, unix timestamp
    num_nodes: int
    node_labels: list          # original identifier for each integer id
    name: str

    @property
    def timestamps(self) -> np.ndarray:
        return self.edges[:, 2]

    def span_days(self) -> float:
        ts = self.timestamps
        return float((ts.max() - ts.min()) / 86400.0)

    def summary(self) -> dict:
        return {
            "dataset": self.name,
            "nodes": self.num_nodes,
            "edges": int(len(self.edges)),
            "span_days": round(self.span_days(), 2),
            "t_min": int(self.timestamps.min()),
            "t_max": int(self.timestamps.max()),
        }


def _reindex(raw_edges) -> tuple:
    """Map arbitrary node identifiers onto 0..n-1, preserving edge order."""
    labels: list = []
    index: dict = {}
    out = np.empty((len(raw_edges), 3), dtype=np.int64)
    for i, (u, v, t) in enumerate(raw_edges):
        for node in (u, v):
            if node not in index:
                index[node] = len(labels)
                labels.append(node)
        out[i] = (index[u], index[v], t)
    return out, labels


def load_collegemsg(path: Path | None = None) -> TemporalGraph:
    """Load ``CollegeMsg.txt`` (``src dst unix_ts`` per line)."""
    path = Path(path) if path else DATA_RAW / "CollegeMsg.txt"
    raw = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                continue
            raw.append((parts[0], parts[1], int(parts[2])))
    raw.sort(key=lambda r: r[2])
    edges, labels = _reindex(raw)
    return TemporalGraph(edges, len(labels), labels, "CollegeMsg")


def _read_json_lenient(path: Path):
    """CED files are mostly UTF-8 with a couple of mixed-encoding strays."""
    blob = path.read_bytes()
    for enc in ("utf-8", "gb18030"):
        try:
            return json.loads(blob.decode(enc))
        except Exception:
            continue
    return None


def load_weibo_ced(
    root: Path | None = None,
    subset: str = "rumor-repost",
    max_cascades: int | None = None,
) -> TemporalGraph:
    """Reconstruct the Weibo repost network from the CED dataset.

    Each cascade file is named ``<label>_<root_mid>_<root_uid>.json`` and holds
    repost records with ``uid``, ``mid``, ``parent`` (the ``mid`` reposted
    from) and ``date``. An edge ``parent_uid -> uid`` is created per repost;
    records with an empty ``parent`` are direct reposts of the cascade root.
    """
    base = Path(root) if root else DATA_RAW / "thunlp_rumor" / "CED_Dataset"
    folder = base / subset
    files = sorted(os.listdir(folder))
    if max_cascades is not None:
        files = files[:max_cascades]

    raw = []
    skipped = 0
    for fname in files:
        records = _read_json_lenient(folder / fname)
        if not records:
            skipped += 1
            continue
        parts = fname.replace(".json", "").split("_")
        root_uid = parts[-1]

        mid_to_uid = {r["mid"]: r["uid"] for r in records if r.get("mid")}
        for r in records:
            uid, date = r.get("uid"), r.get("date")
            if not uid or not date:
                continue
            try:
                ts = int(datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                         .replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                continue
            parent = r.get("parent") or ""
            src = mid_to_uid.get(parent, root_uid) if parent else root_uid
            if src == uid:
                continue                      # ignore self-loops
            raw.append((src, uid, ts))

    raw.sort(key=lambda r: r[2])
    edges, labels = _reindex(raw)
    tg = TemporalGraph(edges, len(labels), labels, "Weibo-CED")
    tg.skipped_files = skipped                # noqa: attribute assignment
    return tg


def load_dataset(name: str, **kwargs) -> TemporalGraph:
    if name.lower() in {"collegemsg", "college"}:
        return load_collegemsg(**kwargs)
    if name.lower() in {"weibo", "ced", "weibo-ced"}:
        return load_weibo_ced(**kwargs)
    raise ValueError(f"unknown dataset {name!r}")


def chronological_split(tg: TemporalGraph, train_frac=0.80, val_frac=0.10,
                        split_by="duration"):
    """Split edges chronologically (Section 3.1).

    ``split_by="duration"`` follows the paper: "80% of the total duration and
    90% of the total duration" are the demarcation points, i.e. cut points on
    the *timeline*. On a bursty, front-loaded network this concentrates almost
    every edge in the training period.

    ``split_by="count"`` cuts at edge *rank* instead, giving balanced splits.
    """
    ts = tg.timestamps
    t0, t1 = float(ts.min()), float(ts.max())

    if split_by == "count":
        order = np.sort(ts)
        cut_train = float(order[min(int(train_frac * len(order)), len(order) - 1)])
        cut_val = float(order[min(int((train_frac + val_frac) * len(order)),
                                  len(order) - 1)])
    elif split_by == "duration":
        span = t1 - t0
        cut_train = t0 + train_frac * span
        cut_val = t0 + (train_frac + val_frac) * span
    else:
        raise ValueError(f"unknown split_by {split_by!r}")

    train = tg.edges[ts <= cut_train]
    val = tg.edges[(ts > cut_train) & (ts <= cut_val)]
    test = tg.edges[ts > cut_val]
    return train, val, test, (cut_train, cut_val)


def build_snapshots(edges: np.ndarray, window_seconds: int, num_nodes: int):
    """Partition ``edges`` into consecutive temporal windows (Section 3.1).

    Returns a list of undirected ``networkx`` graphs, one per window. Every
    snapshot carries all ``num_nodes`` nodes so entropy vectors stay aligned;
    isolated nodes are filtered later by :func:`augment_graph` if requested.
    """
    if len(edges) == 0:
        return []
    ts = edges[:, 2]
    t0 = ts.min()
    bucket = ((ts - t0) // window_seconds).astype(int)
    n_windows = int(bucket.max()) + 1

    snapshots = []
    for w in range(n_windows):
        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        sel = edges[bucket == w]
        if len(sel):
            g.add_edges_from(sel[:, :2].tolist())
        snapshots.append(g)
    return snapshots


def augment_graph(graph: nx.Graph, add_frac=0.05, del_frac=0.02,
                  drop_isolated=True, rng=None) -> nx.Graph:
    """Node/edge perturbation augmentation (Section 3.1).

    "randomly add 5% of nodes and edges and delete 2% of nodes and edges to
    simulate network uncertainty". Applied per temporal window.
    """
    rng = rng or np.random.default_rng(0)
    g = graph.copy()

    nodes = list(g.nodes())
    edges = list(g.edges())
    if not nodes:
        return g

    # --- deletions ---
    n_del_e = int(round(del_frac * len(edges)))
    if n_del_e and edges:
        idx = rng.choice(len(edges), size=min(n_del_e, len(edges)), replace=False)
        g.remove_edges_from([edges[i] for i in idx])

    n_del_n = int(round(del_frac * len(nodes)))
    if n_del_n:
        victims = rng.choice(nodes, size=min(n_del_n, len(nodes)), replace=False)
        g.remove_nodes_from(victims.tolist())

    # --- additions ---
    present = list(g.nodes())
    n_add_n = int(round(add_frac * len(nodes)))
    start = max(nodes) + 1
    new_nodes = list(range(start, start + n_add_n))
    g.add_nodes_from(new_nodes)

    n_add_e = int(round(add_frac * len(edges)))
    pool = present + new_nodes
    if n_add_e and len(pool) > 1:
        a = rng.choice(pool, size=n_add_e)
        b = rng.choice(pool, size=n_add_e)
        g.add_edges_from((int(x), int(y)) for x, y in zip(a, b) if x != y)

    if drop_isolated:
        g.remove_nodes_from([n for n, d in g.degree() if d == 0])
    return g


def sample_negatives(pos_pairs: np.ndarray, num_nodes: int,
                     forbidden: set, neg_ratio: float, rng=None) -> np.ndarray:
    """Draw non-existent node pairs as negative samples (Section 3.1).

    ``neg_ratio`` is ``n_neg / n_pos``. The paper's Figure 8 labels its x-axis
    "positive-to-negative sample ratio" but describes the value 2 as "an excess
    of negative samples", so larger values mean *more* negatives.

    Real networks are sparse, so uniform rejection sampling almost always
    lands on a true non-edge; ``forbidden`` guards the rare collision.
    """
    rng = rng or np.random.default_rng(0)
    n_neg = int(round(neg_ratio * len(pos_pairs)))
    if n_neg <= 0:
        return np.empty((0, 2), dtype=np.int64)

    out = np.empty((n_neg, 2), dtype=np.int64)
    filled = 0
    guard = 0
    while filled < n_neg and guard < 200:
        need = (n_neg - filled) * 2
        cand_u = rng.integers(0, num_nodes, size=need)
        cand_v = rng.integers(0, num_nodes, size=need)
        for u, v in zip(cand_u, cand_v):
            if filled >= n_neg:
                break
            if u == v:
                continue
            key = (int(min(u, v)), int(max(u, v)))
            if key in forbidden:
                continue
            out[filled] = (u, v)
            filled += 1
        guard += 1
    return out[:filled]
