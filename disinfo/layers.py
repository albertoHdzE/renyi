"""The four GNN families defined in Section 3.1, implemented from the survey's
own equations.

Lakzaei, Haghir Chehreghani & Bagheri (2024), *Artificial Intelligence Review*
57:52, Section 3.1 prints five update rules. Each is implemented here in the
form the survey prints, with the literature-standard form available as an
option where the two differ. Every such difference is recorded in
``docs/DISCREPANCIES_SURVEY.md``; the short version is in each class docstring.

    Eq. 1   h_v^l = UPDATE(h_v^{l-1}, AGG({h_u^{l-1} : u in N(v)}))   -- the
            message-passing skeleton every layer below instantiates.
    Eq. 2   GCN
    Eq. 3-4 GAT
    Eq. 5   GATv2
    Eq. 6-9 GraphSAGE with mean / pool / LSTM aggregation
    Eq. 10  GIN

Graphs are passed as ``edge_index``: a ``(2, E)`` int64 tensor whose columns are
``(u, v)`` meaning "u is a neighbour of v", i.e. messages flow u -> v. This
convention keeps ``edge_index[1]`` the *receiver* index used by every scatter.

No PyTorch Geometric or DGL, per the repository's tooling policy: the scatter
primitives these layers need are ``index_add_`` and ``index_select``, both of
which are in core PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "add_self_loops", "degree", "scatter_sum", "scatter_mean", "scatter_max",
    "scatter_softmax", "GCNLayer", "GATLayer", "GATv2Layer", "SAGELayer",
    "GINLayer", "LAYERS",
]


# --------------------------------------------------------------------------
# scatter primitives
# --------------------------------------------------------------------------

def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Append (v, v) for every node. Used by GCN (Eq. 2 sums over N(v) u {v})."""
    loop = torch.arange(num_nodes, device=edge_index.device)
    return torch.cat([edge_index, torch.stack([loop, loop])], dim=1)


def degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """|N(v)| for every v, counting incoming edges."""
    out = torch.zeros(num_nodes, device=edge_index.device)
    out.index_add_(0, edge_index[1], torch.ones(edge_index.size(1),
                                                device=edge_index.device))
    return out


def scatter_sum(src: torch.Tensor, index: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
    """Sum rows of ``src`` into the row of ``index`` they are addressed to."""
    shape = (num_nodes,) + src.shape[1:]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor,
                 num_nodes: int) -> torch.Tensor:
    """Mean over the rows addressed to each node; empty neighbourhoods give 0."""
    total = scatter_sum(src, index, num_nodes)
    count = torch.zeros(num_nodes, dtype=src.dtype, device=src.device)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    count = count.clamp(min=1).view(-1, *([1] * (src.dim() - 1)))
    return total / count


def scatter_max(src: torch.Tensor, index: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
    """Element-wise max; empty neighbourhoods give 0 rather than -inf."""
    shape = (num_nodes,) + src.shape[1:]
    out = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    return out.masked_fill(torch.isinf(out), 0.0)


def scatter_softmax(src: torch.Tensor, index: torch.Tensor,
                    num_nodes: int) -> torch.Tensor:
    """Softmax over each receiving node's incoming edges.

    This is the neighbourhood normalisation of Velickovic et al. (2017). The
    survey's Eq. 4 omits it (see ``GATLayer``). Subtracting the per-node max
    before exponentiating keeps the attention logits numerically safe: raw
    LeakyReLU logits on high-degree nodes routinely reach magnitudes where
    ``exp`` overflows float32.
    """
    src_max = scatter_max(src.detach(), index, num_nodes).index_select(0, index)
    out = (src - src_max).exp()
    denom = scatter_sum(out, index, num_nodes).index_select(0, index)
    return out / denom.clamp(min=1e-16)


# --------------------------------------------------------------------------
# Eq. 2 -- graph convolutional network
# --------------------------------------------------------------------------

class GCNLayer(nn.Module):
    r"""Graph convolutional network, Eq. 2.

    The survey prints

    .. math::
        h_v^l = \sigma\Big( W^{l-1} \sum_{u \in N(v) \cup \{v\}}
                 \frac{h_u^{l-1}}{|N(v)|} \Big)

    and attributes it to Welling & Kipf (2016). It is not quite their rule:
    Kipf & Welling normalise symmetrically by
    :math:`\tilde D^{-1/2}\tilde A\tilde D^{-1/2}`, whereas Eq. 2 divides by the
    receiver's degree alone. Note also that Eq. 2 puts :math:`|N(v)|` in the
    denominator while summing over :math:`|N(v)| + 1` terms, so the self term is
    not averaged with the rest and an isolated node divides by zero.

    ``normalization`` selects between them:

    ``"paper"``      Eq. 2 exactly, with the degree clamped at 1 so that
                     isolated nodes pass through their own feature instead of
                     producing NaN.
    ``"symmetric"``  Kipf & Welling. This is the default, because it is what
                     every method in Tables 1-2 that says "GCN" actually ran.
    """

    def __init__(self, in_dim: int, out_dim: int, normalization: str = "symmetric",
                 bias: bool = True):
        super().__init__()
        if normalization not in ("paper", "symmetric"):
            raise ValueError(f"unknown normalization {normalization!r}")
        self.normalization = normalization
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        ei = add_self_loops(edge_index, n)          # N(v) u {v}
        u, v = ei[0], ei[1]

        if self.normalization == "paper":
            # Eq. 2 literally: one factor of 1/|N(v)|, self-loops excluded from
            # the count exactly as printed.
            deg = degree(edge_index, n).clamp(min=1)
            msg = x.index_select(0, u) / deg.index_select(0, v).unsqueeze(-1)
        else:
            deg = degree(ei, n).clamp(min=1)         # includes the self-loop
            dis = deg.pow(-0.5)
            msg = (x.index_select(0, u)
                   * (dis.index_select(0, u) * dis.index_select(0, v)).unsqueeze(-1))

        return self.lin(scatter_sum(msg, v, n))


# --------------------------------------------------------------------------
# Eqs. 3-4 -- graph attention network
# --------------------------------------------------------------------------

class GATLayer(nn.Module):
    r"""Graph attention network, Eqs. 3 and 4.

    .. math::
        h_v^l = \Big\|_{k=1}^{K} \sigma\Big(
                \sum_{u \in N(v)} \alpha_{vu}^k W^{l-1} h_u^{l-1} \Big)
        \qquad
        \alpha_{vu} = \sigma\big(a(W^{l-1}h_v^{l-1}, W^{l-1}h_u^{l-1})\big)

    Eq. 4 as printed has no softmax, so the "attention coefficients" would not
    sum to one over a neighbourhood and the aggregated message would grow with
    degree. Velickovic et al. (2017), which the survey cites, normalise with a
    softmax over :math:`N(v)` and use LeakyReLU(0.2) for :math:`\sigma`. Both
    are applied here; ``softmax=False`` reproduces the printed form and is kept
    only so the notebook can show what it does to training.

    ``concat`` implements the :math:`\|` of Eq. 3. The convention of the
    original paper -- concatenate on hidden layers, average on the output layer
    -- is followed by ``GNNEncoder``.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1,
                 concat: bool = True, dropout: float = 0.0,
                 negative_slope: float = 0.2, softmax: bool = True):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.negative_slope, self.softmax, self.dropout = negative_slope, softmax, dropout
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        # `a` of Eq. 4, split into the halves acting on the receiver and sender.
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_src)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        ei = add_self_loops(edge_index, n)
        u, v = ei[0], ei[1]

        h = self.lin(x).view(n, self.heads, self.out_dim)       # W h
        # a(Wh_v, Wh_u) decomposed as <a_dst, Wh_v> + <a_src, Wh_u>, the
        # standard rewriting of the concatenation-then-dot-product of Eq. 4.
        alpha = ((h * self.att_dst).sum(-1).index_select(0, v)
                 + (h * self.att_src).sum(-1).index_select(0, u))
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = scatter_softmax(alpha, v, n) if self.softmax else torch.sigmoid(alpha)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = h.index_select(0, u) * alpha.unsqueeze(-1)
        out = scatter_sum(msg, v, n)
        out = out.reshape(n, -1) if self.concat else out.mean(dim=1)
        return out + self.bias


class GATv2Layer(GATLayer):
    r"""GATv2, Eq. 5: :math:`\alpha_{vu} = a \cdot \sigma(W[h_v, h_u])`.

    The survey's own justification (Sect. 3.1) is the one that matters here:
    in Eq. 4 the two linear maps ``a`` and ``W`` compose into a single linear
    map, which makes the ranking of neighbours independent of the receiving
    node. GATv2 fixes this by applying the nonlinearity *before* ``a`` rather
    than after, so only the order of two operations changes -- but that order is
    the whole difference between static and dynamic attention (Brody et al.
    2021).
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1,
                 concat: bool = True, dropout: float = 0.0,
                 negative_slope: float = 0.2, softmax: bool = True):
        super().__init__(in_dim, out_dim, heads, concat, dropout,
                         negative_slope, softmax)
        # GATv2 keeps separate transforms for receiver and sender.
        self.lin_dst = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, out_dim))
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        ei = add_self_loops(edge_index, n)
        u, v = ei[0], ei[1]

        h_src = self.lin(x).view(n, self.heads, self.out_dim)
        h_dst = self.lin_dst(x).view(n, self.heads, self.out_dim)

        # sigma first (Eq. 5), then the linear form a -- the reverse of Eq. 4.
        e = F.leaky_relu(h_dst.index_select(0, v) + h_src.index_select(0, u),
                         self.negative_slope)
        alpha = (e * self.att).sum(-1)
        alpha = scatter_softmax(alpha, v, n) if self.softmax else torch.sigmoid(alpha)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = h_src.index_select(0, u) * alpha.unsqueeze(-1)
        out = scatter_sum(msg, v, n)
        out = out.reshape(n, -1) if self.concat else out.mean(dim=1)
        return out + self.bias


# --------------------------------------------------------------------------
# Eqs. 6-9 -- GraphSAGE
# --------------------------------------------------------------------------

class SAGELayer(nn.Module):
    r"""GraphSAGE, Eq. 6, with the three aggregators of Eqs. 7-9.

    .. math::
        h_v^l = \sigma\big(W^{l-1}\,[\,h_v^{l-1} \,\|\,
                AGG(h_u^{l-1}, \forall u \in N(v))\,]\big)

    ``aggregator``:

    ``"mean"``  Eq. 7, :math:`\sum_{u} h_u / |N(v)|`.
    ``"pool"``  Eq. 8, :math:`\mathrm{mean}(MLP(h_u))`. The survey's prose names
                the three aggregators "average pooling, maximum pooling and
                LSTM", but Eq. 8 prints ``mean``; Hamilton et al. (2017) use
                ``max``. ``pool_reduce`` selects, defaulting to ``"mean"`` to
                follow the printed equation.
    ``"lstm"``  Eq. 9. Order-dependent by construction, so GraphSAGE permutes
                neighbours at random; ``shuffle`` keeps that, without which the
                layer silently learns the node-id ordering.

    ``fanout`` is the neighbour sampling that distinguishes GraphSAGE from GCN
    in the survey's own description ("utilizes the sampled neighborhood").
    ``None`` uses the full neighbourhood, which is exact and affordable here.
    """

    def __init__(self, in_dim: int, out_dim: int, aggregator: str = "mean",
                 pool_reduce: str = "mean", fanout: int | None = None,
                 shuffle: bool = True, bias: bool = True):
        super().__init__()
        if aggregator not in ("mean", "pool", "lstm"):
            raise ValueError(f"unknown aggregator {aggregator!r}")
        if pool_reduce not in ("mean", "max"):
            raise ValueError(f"unknown pool_reduce {pool_reduce!r}")
        self.aggregator, self.pool_reduce = aggregator, pool_reduce
        self.fanout, self.shuffle = fanout, shuffle

        if aggregator == "pool":
            self.pool_mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        elif aggregator == "lstm":
            self.lstm = nn.LSTM(in_dim, in_dim, batch_first=True)

        self.lin = nn.Linear(in_dim * 2, out_dim, bias=bias)   # the [.||.] of Eq. 6

    def _sample(self, edge_index: torch.Tensor, n: int) -> torch.Tensor:
        if self.fanout is None or not self.training or edge_index.numel() == 0:
            return edge_index
        # Keep an edge with probability fanout/deg(receiver): an unbiased
        # stand-in for "at most `fanout` neighbours" that stays vectorised.
        deg = degree(edge_index, n).clamp(min=1)
        p = (self.fanout / deg).clamp(max=1.0).index_select(0, edge_index[1])
        keep = torch.rand(edge_index.size(1), device=edge_index.device)
        return edge_index[:, keep < p]

    def _aggregate(self, x: torch.Tensor, edge_index: torch.Tensor,
                   n: int) -> torch.Tensor:
        u, v = edge_index[0], edge_index[1]
        if self.aggregator == "mean":                                  # Eq. 7
            return scatter_mean(x.index_select(0, u), v, n)
        if self.aggregator == "pool":                                  # Eq. 8
            m = self.pool_mlp(x).index_select(0, u)
            return (scatter_mean(m, v, n) if self.pool_reduce == "mean"
                    else scatter_max(m, v, n))
        return self._aggregate_lstm(x, u, v, n)                        # Eq. 9

    def _aggregate_lstm(self, x, u, v, n):
        """Eq. 9. Neighbourhoods are packed into a padded (n, max_deg, d) batch."""
        out = torch.zeros(n, x.size(1), dtype=x.dtype, device=x.device)
        if u.numel() == 0:
            return out

        order = torch.argsort(v)
        u_s, v_s = u[order], v[order]
        if self.shuffle and self.training:
            # Random tie-break within each neighbourhood: sort a key whose
            # integer part is the receiver, so groups stay contiguous.
            key = v.to(torch.float64) + torch.rand(v.numel(), device=v.device) * 0.5
            order = torch.argsort(key)
            u_s, v_s = u[order], v[order]

        deg = degree(torch.stack([u_s, v_s]), n)
        max_deg = int(deg.max().item())
        starts = torch.cumsum(deg, 0) - deg
        slot = torch.arange(u_s.numel(), device=x.device) - starts.index_select(0, v_s).long()

        padded = torch.zeros(n, max_deg, x.size(1), dtype=x.dtype, device=x.device)
        padded[v_s, slot] = x.index_select(0, u_s)

        active = deg > 0
        packed = nn.utils.rnn.pack_padded_sequence(
            padded[active], deg[active].cpu().long(),
            batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        out[active] = h_n[-1]
        return out

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        agg = self._aggregate(x, self._sample(edge_index, n), n)
        return self.lin(torch.cat([x, agg], dim=1))


# --------------------------------------------------------------------------
# Eq. 10 -- graph isomorphism network
# --------------------------------------------------------------------------

class GINLayer(nn.Module):
    r"""Graph isomorphism network, Eq. 10.

    .. math::
        h_v^l = MLP^l\big((1 + \epsilon^l) h_v^{l-1}
                          + \sum_{u \in N(v)} h_u^{l-1}\big)

    The **sum** is the point: Xu et al. (2018) prove that mean and max
    aggregation cannot distinguish neighbourhood multisets that differ only in
    multiplicity, so sum is what makes the layer as discriminative as the
    Weisfeiler-Lehman test the survey invokes. ``train_eps`` covers the survey's
    "fixed scalar value, or it can be learned" (GIN-0 vs GIN-eps).
    """

    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0,
                 train_eps: bool = False, hidden: int | None = None):
        super().__init__()
        hidden = hidden or out_dim
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(float(eps)))
        else:
            self.register_buffer("eps", torch.tensor(float(eps)))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        agg = scatter_sum(x.index_select(0, edge_index[0]), edge_index[1], n)
        return self.mlp((1.0 + self.eps) * x + agg)


LAYERS = {
    "gcn": GCNLayer,
    "gat": GATLayer,
    "gatv2": GATv2Layer,
    "sage": SAGELayer,
    "gin": GINLayer,
}
