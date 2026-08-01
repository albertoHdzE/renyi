"""Dynamic Time-Weighted Rényi Entropy (DTWRE).

Implements Equations 1-4 of Tong et al., *Entropy* 2025, 27, 516.

    Eq. 1  H_a(v,t)        = 1/(1-a) * log( sum_{u in N(v,t)} p_u^a(t) )
    Eq. 2  H_a^global(t)   = sum_{v in V_t} H_a(v,t)
    Eq. 3  H_a^time(G,t)   = sum_{t_k < t} w(t - t_k) * H_a^global(t_k)
    Eq. 4  w(t - t_k)      = exp(-lam * (t - t_k))

All logarithms are natural (nats); the paper does not state a base, and the
choice only rescales every entropy by a constant factor.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

__all__ = [
    "renyi_entropy",
    "node_probabilities",
    "local_node_entropy",
    "global_timestep_entropy",
    "time_weight",
    "dtwre_series",
    "static_renyi_entropy",
]

_ALPHA_TOL = 1e-8


def renyi_entropy(p: np.ndarray, alpha: float) -> float:
    """Rényi entropy of order ``alpha`` for the weights ``p`` (Eq. 1).

    Computed as ``1/(1-alpha) * logsumexp(alpha * log p)`` for numerical
    stability. At ``alpha == 1`` the Rényi entropy is defined by its limit,
    the Shannon entropy ``-sum(p_norm * log p_norm)``; the limit only exists
    when ``p`` sums to one, so ``p`` is renormalised first.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0

    if abs(alpha - 1.0) < _ALPHA_TOL:
        q = p / p.sum()
        return float(-np.sum(q * np.log(q)))

    logp = np.log(p)
    m = np.max(alpha * logp)
    log_sum = m + np.log(np.sum(np.exp(alpha * logp - m)))
    return float(log_sum / (1.0 - alpha))


def node_probabilities(
    graph: nx.Graph,
    metric: str = "degree",
    normalisation: str = "local",
) -> dict:
    """Per-node information weight ``p_u`` used inside Eq. 1.

    ``metric`` selects the information measure the paper compares in
    Section 2.2: ``"degree"`` (Algorithm 1) or ``"pagerank"``.

    ``normalisation`` controls how Eq. 1 is made a probability distribution:

    * ``"global"`` -- literal Algorithm 1: ``p(v) = deg(v) / sum_i deg(v_i)``,
      normalised once over the whole snapshot. The neighbourhood sum in Eq. 1
      is then below 1 and the entropy diverges as ``alpha -> 1``.
    * ``"local"`` -- renormalise within each neighbourhood ``N(v,t)`` so Eq. 1
      is a genuine Rényi entropy for every ``alpha``. This is the default.

    Returns the *unnormalised* score per node; local renormalisation happens
    per neighbourhood in :func:`local_node_entropy`.
    """
    if metric == "degree":
        raw = {n: float(d) for n, d in graph.degree()}
    elif metric == "pagerank":
        raw = nx.pagerank(graph)
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'degree' or 'pagerank'")

    if normalisation == "global":
        total = sum(raw.values())
        if total <= 0:
            return {n: 0.0 for n in raw}
        return {n: v / total for n, v in raw.items()}
    if normalisation == "local":
        return raw
    raise ValueError(f"unknown normalisation {normalisation!r}")


def local_node_entropy(
    graph: nx.Graph,
    alpha: float = 0.6,
    metric: str = "degree",
    normalisation: str = "local",
) -> dict:
    """Local Node Entropy for every node of one snapshot (Eq. 1).

    A node with no neighbours has entropy 0 -- the paper removes isolated
    nodes during preprocessing, so this is a guard rather than a policy.
    """
    p = node_probabilities(graph, metric=metric, normalisation=normalisation)

    lne = {}
    for v in graph.nodes():
        neigh = list(graph.neighbors(v))
        if not neigh:
            lne[v] = 0.0
            continue
        weights = np.array([p[u] for u in neigh], dtype=np.float64)
        if normalisation == "local":
            s = weights.sum()
            weights = weights / s if s > 0 else weights
        lne[v] = renyi_entropy(weights, alpha)
    return lne


def global_timestep_entropy(lne: dict) -> float:
    """Global entropy of one snapshot: the sum of its node entropies (Eq. 2)."""
    return float(sum(lne.values()))


def time_weight(delta: np.ndarray | float, lam: float) -> np.ndarray | float:
    """Exponential attenuation weight ``exp(-lam * delta)`` (Eq. 4).

    ``delta`` is measured in *time-step indices*, not seconds. With raw
    seconds (``delta`` ~ 6e5 for a 7-day window) ``exp(-1.2 * 6e5)``
    underflows to exactly 0 and DTWRE collapses to zero for every snapshot,
    so the reported range ``lam in [0.1, 2]`` is only meaningful per step.
    """
    return np.exp(-lam * np.asarray(delta, dtype=np.float64))


def dtwre_series(global_entropies, lam: float = 1.2) -> np.ndarray:
    """Dynamic Time-Weighted Rényi Entropy for each time step (Eq. 3).

    ``global_entropies[k]`` is ``H_a^global(t_k)``. Eq. 3 sums strictly over
    ``t_k < t``, so the first step has no history and DTWRE is 0 there.
    """
    h = np.asarray(global_entropies, dtype=np.float64)
    T = len(h)
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        if t == 0:
            continue
        k = np.arange(t)                       # strictly earlier steps
        out[t] = float(np.sum(time_weight(t - k, lam) * h[k]))
    return out


def static_renyi_entropy(
    graph: nx.Graph,
    alpha: float = 0.6,
    metric: str = "degree",
    normalisation: str = "local",
) -> dict:
    """Baseline (4): Rényi entropy on the aggregated graph, no time weighting.

    Identical to :func:`local_node_entropy` but intended to be called on the
    full static graph rather than a temporal snapshot.
    """
    return local_node_entropy(graph, alpha=alpha, metric=metric,
                              normalisation=normalisation)
