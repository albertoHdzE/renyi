"""Stage 2 of Fig. 1: graph construction (Sect. 5.3).

The survey's taxonomy of GNN methods is *by how the graph is built*, not by
architecture -- that is its stated novel contribution ("For the first time, we
examine GNNs-based methods in details from the perspective of graph type and
graph construction approach"). Fig. 6 gives three leaves, all implemented here:

    Homogeneous / Similarity graph   ``knn_similarity_graph``,
                                     ``threshold_similarity_graph``,
                                     ``attribute_graph``
    Homogeneous / Propagation graph  ``propagation_graph``
    Heterogeneous graph              ``heterogeneous_graph``

The two homogeneous kinds produce structurally different learning problems, and
the survey is explicit about why (Sect. 5.3.1): a similarity graph joins *all*
items into one graph, so classification is **node**-level and can be
semi-supervised; a propagation graph is built per news item, so classification
is **graph**-level and is supervised. That is the mechanism behind Table 1's
"Setting" column -- all three semi-supervised rows are similarity-graph methods.

Graphs are returned as ``Graph``: an ``edge_index`` of shape ``(2, E)`` in the
``layers`` convention (column ``(u, v)`` means u sends to v), plus node
features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import Config
from .data import DisinfoDataset

__all__ = ["Graph", "knn_similarity_graph", "threshold_similarity_graph",
           "attribute_graph", "propagation_graph", "heterogeneous_graph",
           "build_graph", "build_graphs", "graph_stats"]


@dataclass
class Graph:
    x: torch.Tensor              # (n_nodes, d)
    edge_index: torch.Tensor     # (2, E) int64
    y: torch.Tensor | None = None
    node_type: torch.Tensor | None = None   # heterogeneous graphs only
    meta: dict | None = None

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)


def _to_undirected(edges: np.ndarray) -> np.ndarray:
    """Symmetrise and drop duplicate/self edges.

    Jiang et al. (2021) motivate this directly: their L-GAT "views all edges as
    bi-directional" precisely because splitting top-down from bottom-up (as
    Bi-GCN does) "restricts the interaction between nodes".
    """
    if edges.size == 0:
        return edges.reshape(2, 0)
    both = np.concatenate([edges, edges[::-1]], axis=1)
    both = both[:, both[0] != both[1]]
    return np.unique(both, axis=1)


# --------------------------------------------------------------------------
# similarity graphs (Sect. 5.3, first homogeneous approach)
# --------------------------------------------------------------------------

def knn_similarity_graph(X: np.ndarray, k: int = 4,
                         metric: str = "cosine",
                         undirected: bool = True) -> np.ndarray:
    """k-nearest-neighbour graph over item features.

    Benamira et al. (2019) -- the first row of Table 1 -- build exactly this:
    "the k-nearest-neighbor method (where k is set to 4) is employed to
    establish a graph among articles based on word embedding similarities".
    Sect. 5.3.1 calls kNN on content similarity "one of the prevailing
    techniques" for similarity graphs.

    The edge is directed *from* each neighbour *into* the query node, so that
    after symmetrisation every node has degree at least k.
    """
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return np.zeros((2, 0), dtype=np.int64)

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    _, idx = nn.kneighbors(X)

    # Column 0 is the point itself; drop it.
    dst = np.repeat(np.arange(n), k)
    src = idx[:, 1:].reshape(-1)
    edges = np.stack([src, dst]).astype(np.int64)
    return _to_undirected(edges) if undirected else edges


def threshold_similarity_graph(X: np.ndarray, threshold: float = 0.8,
                               max_degree: int = 64) -> np.ndarray:
    """Join items whose cosine similarity exceeds a threshold.

    Yuan et al. (2021), the DAGA-NN row of Table 1: "an edge is created between
    corresponding nodes if the cosine similarity between their feature
    representations exceeds a predefined threshold value".

    ``max_degree`` caps each node's neighbours at the most similar ones. Without
    it a threshold slightly too low yields a near-complete graph -- quadratic
    memory and a GNN that averages the whole dataset into every node.
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)

    keep = S >= threshold
    if max_degree is not None:
        order = np.argsort(-S, axis=1)[:, :max_degree]
        cap = np.zeros_like(keep)
        np.put_along_axis(cap, order, True, axis=1)
        keep &= cap

    src, dst = np.nonzero(keep)
    return _to_undirected(np.stack([src, dst]).astype(np.int64))


def attribute_graph(ds: DisinfoDataset,
                    attributes: tuple[str, ...] = ("party", "subject", "state"),
                    max_group: int = 200,
                    seed: int = 0) -> np.ndarray:
    """Join items that share a speaker-profile attribute.

    This is the M-GCN construction of Hu et al. (2019), the LIAR row of Table 1,
    described in Sect. 5.3.2: "a speaker's profile, including attributes such as
    their political party, the news topic, or their home state, is considered.
    If two nodes share the same value for a given profile attribute, they are
    connected via an edge."

    Taken literally this is a union of cliques, and LIAR's ``party`` field has
    two values covering most of the corpus -- a clique of ~5000 nodes, i.e. 12
    million edges that carry almost no information. ``max_group`` therefore
    subsamples a random ``max_group``-regular graph within any group larger than
    that, preserving "these items share an attribute" while keeping the degree
    finite. Groups of one contribute nothing.
    """
    rng = np.random.default_rng(seed)
    src: list[int] = []
    dst: list[int] = []

    for attr in attributes:
        groups: dict[str, list[int]] = {}
        for i, it in enumerate(ds.items):
            for value in str(it.meta.get(attr) or "").split(","):
                value = value.strip()
                if value:
                    groups.setdefault(f"{attr}={value}", []).append(i)

        for members in groups.values():
            m = len(members)
            if m < 2:
                continue
            arr = np.asarray(members)
            if m <= max_group:
                a, b = np.triu_indices(m, k=1)
                src.extend(arr[a])
                dst.extend(arr[b])
            else:
                # Random max_group-regular-ish subsample of the clique.
                for _ in range(max_group):
                    perm = rng.permutation(arr)
                    src.extend(perm)
                    dst.extend(np.roll(perm, 1))

    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return _to_undirected(np.stack([np.asarray(src), np.asarray(dst)]).astype(np.int64))


# --------------------------------------------------------------------------
# propagation graphs (Sect. 5.3, second homogeneous approach)
# --------------------------------------------------------------------------

def propagation_graph(item, item_feature: np.ndarray, cfg: Config) -> Graph:
    """One graph per news item, from its repost tree.

    "The second approach involves creating a propagation graph for each news
    event by considering the source post and its subsequent reposts."

    Node features concatenate three blocks, which is what the Table 1-2
    propagation methods do in one form or another:

    * the source item's own feature vector, broadcast to every node -- Bian et
      al.'s (2020) *root feature enhancement*, "the source post information is
      integrated into each layer of GCN to enhance the influence from the roots
      of rumors";
    * that node's user profile features, where the corpus provides them
      (Malhotra et al. 2020 build the tree at user level with 12 profile
      features);
    * the node's own position in the cascade: log delay and normalised depth.

    ``cfg.add_root_edges`` additionally connects every node to the root. Bian et
    al. motivate this; it also guarantees connectivity, since a 2-layer GNN
    otherwise cannot see past depth 2 of a chain-shaped cascade.
    """
    c = item.cascade
    n = 1 if c is None else max(c.size, 1)

    src_feat = np.repeat(item_feature[None, :], n, axis=0)

    if c is not None and c.user_features is not None and len(c.user_features) == n:
        prof = c.user_features
    else:
        prof = np.zeros((n, 0), dtype=np.float32)

    if c is not None and len(c.times) == n:
        t = np.asarray(c.times, dtype=np.float64)
        t = np.where(np.isfinite(t), t, 0.0)
        pos = np.stack([np.log1p(np.clip(t, 0, None)),
                        np.arange(n) / n], axis=1).astype(np.float32)
    else:
        pos = np.zeros((n, 2), dtype=np.float32)

    x = np.concatenate([src_feat, prof, pos], axis=1).astype(np.float32)

    edges = (np.asarray(c.edges, dtype=np.int64).T if (c and c.edges)
             else np.zeros((2, 0), dtype=np.int64))
    if cfg.add_root_edges and n > 1:
        others = np.arange(1, n)
        root = np.stack([np.zeros_like(others), others])
        edges = np.concatenate([edges, root], axis=1) if edges.size else root
    if cfg.undirected:
        edges = _to_undirected(edges)

    return Graph(torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(edges)),
                 meta={"id": item.id, "n_nodes": n})


# --------------------------------------------------------------------------
# heterogeneous graphs (Sect. 5.3)
# --------------------------------------------------------------------------

def heterogeneous_graph(item, item_feature: np.ndarray, cfg: Config) -> Graph:
    """A post-user bipartite graph plus the repost tree.

    "Social networks are characterized by heterogeneous graphs, encompassing
    various types of nodes (users, posts, comments, etc) and edges."

    Node type 0 is a post (one per cascade node), type 1 a distinct user. A user
    node is joined to every post it authored, which is the tweet-user subgraph
    of Huang et al. (2020) and the mechanism by which a user reposting twice
    creates a path between two otherwise distant branches. Type is returned in
    ``node_type`` so a type-aware readout can use it; the layers themselves are
    type-agnostic, which is the honest limit of this implementation -- AA-HGNN
    and MFAN use schema-level attention that Sect. 5.3.2 describes but no
    equation in the survey defines.
    """
    base = propagation_graph(item, item_feature, cfg)
    c = item.cascade
    n_post = base.num_nodes
    if c is None or c.size == 0:
        return Graph(base.x, base.edge_index,
                     node_type=torch.zeros(n_post, dtype=torch.long),
                     meta=base.meta)

    uids = list(c.node_uids)
    uniq = {u: i for i, u in enumerate(sorted(set(uids)))}
    n_user = len(uniq)

    d = base.x.size(1)
    user_x = torch.zeros(n_user, d)
    for p, u in enumerate(uids):
        user_x[uniq[u]] += base.x[p]
    counts = torch.zeros(n_user, 1)
    for u in uids:
        counts[uniq[u]] += 1
    user_x /= counts.clamp(min=1)          # a user node averages its posts

    x = torch.cat([base.x, user_x], dim=0)
    node_type = torch.cat([torch.zeros(n_post, dtype=torch.long),
                           torch.ones(n_user, dtype=torch.long)])

    authored = np.stack([
        np.asarray([n_post + uniq[u] for u in uids], dtype=np.int64),
        np.arange(n_post, dtype=np.int64),
    ])
    edges = np.concatenate([base.edge_index.numpy(), _to_undirected(authored)],
                           axis=1)
    return Graph(x, torch.from_numpy(np.ascontiguousarray(edges)),
                 node_type=node_type,
                 meta={**(base.meta or {}), "n_post": n_post, "n_user": n_user})


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

def build_graph(ds: DisinfoDataset, X: np.ndarray, cfg: Config) -> Graph:
    """Build the single item-level graph used for node classification."""
    if cfg.graph == "similarity":
        if cfg.sim_threshold is not None:
            e = threshold_similarity_graph(X, cfg.sim_threshold)
        else:
            e = knn_similarity_graph(X, cfg.knn_k, cfg.similarity, cfg.undirected)
    elif cfg.graph == "attribute":
        e = attribute_graph(ds, seed=cfg.seed)
    else:
        raise ValueError(f"{cfg.graph!r} is a per-item graph; use build_graphs")

    return Graph(torch.from_numpy(X), torch.from_numpy(np.ascontiguousarray(e)),
                 y=torch.from_numpy(ds.y))


def build_graphs(ds: DisinfoDataset, X: np.ndarray, cfg: Config) -> list[Graph]:
    """Build one graph per item, for graph classification."""
    if not ds.has_cascades:
        raise ValueError(f"{ds.name} has no cascades; "
                         f"only similarity/attribute graphs are possible")
    fn = heterogeneous_graph if cfg.graph == "heterogeneous" else propagation_graph
    y = ds.y
    out = []
    for i, item in enumerate(ds.items):
        g = fn(item, X[i], cfg)
        g.y = torch.tensor(int(y[i]))
        out.append(g)
    return out


def graph_stats(g: Graph | list[Graph]) -> dict:
    """Descriptive statistics, for the notebook's graph-construction section."""
    if isinstance(g, list):
        n = np.array([x.num_nodes for x in g])
        e = np.array([x.num_edges for x in g])
        return {"n_graphs": len(g), "nodes_mean": float(n.mean()),
                "nodes_median": float(np.median(n)), "nodes_max": int(n.max()),
                "edges_mean": float(e.mean()),
                "isolated_frac": float(np.mean(e == 0))}
    deg = np.bincount(g.edge_index[1].numpy(), minlength=g.num_nodes)
    return {"n_nodes": g.num_nodes, "n_edges": g.num_edges,
            "mean_degree": float(deg.mean()), "max_degree": int(deg.max()),
            "isolated_frac": float(np.mean(deg == 0))}
