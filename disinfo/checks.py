"""Property checks for the layers of Section 3.1.

The repository has no test suite; correctness is asserted against properties
the mathematics must satisfy. These functions are called from the notebook so a
reader watches each equation earn its claim, and from ``scripts/run_disinfo.py``
so a run that violates one fails loudly rather than producing a plausible table.

Each function raises ``AssertionError`` on failure and returns a short string
describing what it confirmed.
"""

from __future__ import annotations

import torch

from .layers import (GCNLayer, GATLayer, GATv2Layer, SAGELayer, GINLayer,
                     degree, scatter_softmax, scatter_sum)

__all__ = ["check_permutation_equivariance", "check_gcn_normalization",
           "check_attention_is_a_distribution", "check_gin_injectivity",
           "check_isolated_nodes_are_finite", "check_gat_v2_is_dynamic",
           "run_all_checks"]


def _ring(n: int = 6) -> torch.Tensor:
    """Undirected ring on n nodes, as a (2, 2n) edge_index."""
    i = torch.arange(n)
    fwd = torch.stack([i, (i + 1) % n])
    return torch.cat([fwd, fwd.flip(0)], dim=1)


def check_permutation_equivariance(tol: float = 1e-5) -> str:
    """A GNN layer must commute with relabelling of the nodes.

    Permuting node ids must permute the output rows identically: the layer may
    only use the graph, never the numbering. This is the single property that
    separates a genuine GNN from an MLP that happens to read an adjacency
    matrix, so every non-LSTM aggregator is held to it. GraphSAGE-LSTM is
    excluded by construction -- Eq. 9 is order-dependent, which is why the
    original paper permutes neighbours at random.
    """
    torch.manual_seed(0)
    n, d = 6, 5
    x = torch.randn(n, d)
    ei = _ring(n)
    perm = torch.randperm(n)
    inv = torch.argsort(perm)
    # Relabel: new node i is old node perm[i], so old u becomes inv[u].
    ei_p = inv[ei]
    x_p = x[perm]

    layers = {
        "GCN (Eq. 2)": GCNLayer(d, 4),
        "GCN (Eq. 2, paper norm)": GCNLayer(d, 4, normalization="paper"),
        "GAT (Eqs. 3-4)": GATLayer(d, 4, heads=2),
        "GATv2 (Eq. 5)": GATv2Layer(d, 4, heads=2),
        "SAGE-mean (Eq. 7)": SAGELayer(d, 4, "mean"),
        "SAGE-pool (Eq. 8)": SAGELayer(d, 4, "pool"),
        "GIN (Eq. 10)": GINLayer(d, 4),
    }
    for name, layer in layers.items():
        layer.eval()
        out = layer(x, ei)
        out_p = layer(x_p, ei_p)
        err = (out[perm] - out_p).abs().max().item()
        assert err < tol, f"{name} is not permutation-equivariant (err={err:.2e})"
    return f"permutation equivariance holds for {len(layers)} layers (max err < {tol:g})"


def check_gcn_normalization(tol: float = 1e-5) -> str:
    """On a regular graph with constant features, GCN must return constant rows.

    A ring is 2-regular, so every node sees an identical neighbourhood. Both
    normalisations must therefore map a constant signal to a constant signal --
    a direct check that the degree normalisation of Eq. 2 is applied per
    receiving node and not, say, globally.
    """
    n, d = 6, 4
    x = torch.ones(n, d)
    ei = _ring(n)
    for norm in ("paper", "symmetric"):
        layer = GCNLayer(d, 3, normalization=norm).eval()
        out = layer(x, ei)
        spread = (out - out[0]).abs().max().item()
        assert spread < tol, f"GCN({norm}) broke regularity (spread={spread:.2e})"

    # The two normalisations must genuinely differ: Eq. 2 divides by |N(v)| = 2
    # and sums 3 terms, giving 3/2 of the input; Kipf & Welling give exactly 1.
    lin = torch.eye(4)[:3]
    paper = GCNLayer(4, 3, normalization="paper").eval()
    sym = GCNLayer(4, 3, normalization="symmetric").eval()
    with torch.no_grad():
        for m in (paper, sym):
            m.lin.weight.copy_(lin)
            m.lin.bias.zero_()
    p, s = paper(x, ei)[0, 0].item(), sym(x, ei)[0, 0].item()
    assert abs(p - 1.5) < tol, f"Eq. 2 literal form gave {p}, expected 3/2"
    assert abs(s - 1.0) < tol, f"symmetric form gave {s}, expected 1"
    return f"GCN preserves regularity; Eq.2 scales by {p:.2f} vs Kipf-Welling {s:.2f}"


def check_attention_is_a_distribution(tol: float = 1e-5) -> str:
    """GAT attention coefficients must sum to 1 over each neighbourhood.

    This is what Eq. 4 omits. Without it the aggregated message scales with
    degree, so hub nodes dominate purely by being hubs. The check is run on the
    scatter softmax directly, on a graph with deliberately unequal degrees.
    """
    n = 5
    # A star plus a chain: degrees 4, 1, 1, 1, 2 -- nothing regular.
    ei = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4, 4],
                       [1, 2, 3, 4, 0, 0, 0, 0, 3]])
    torch.manual_seed(0)
    logits = torch.randn(ei.size(1)) * 10.0     # large: also exercises overflow
    alpha = scatter_softmax(logits, ei[1], n)
    sums = scatter_sum(alpha, ei[1], n)
    deg = degree(ei, n)
    err = (sums[deg > 0] - 1.0).abs().max().item()
    assert err < tol, f"attention does not normalise (err={err:.2e})"
    assert torch.isfinite(alpha).all(), "attention overflowed to inf/nan"
    return f"attention sums to 1 on every non-isolated node (err < {tol:g})"


def check_gin_injectivity() -> str:
    """GIN's sum must separate multisets that mean and max cannot.

    Xu et al. (2018) motivate Eq. 10 by injectivity on neighbourhood multisets.
    Node A with neighbours {x, x} and node B with neighbour {x} have the same
    mean and the same max but different sums, so a sum aggregator must tell
    them apart and a mean aggregator must not. This is the concrete reason the
    survey groups GIN with the Weisfeiler-Lehman test.
    """
    d = 3
    x = torch.ones(4, d)
    # node 0 receives from 2 and 3; node 1 receives from 2 only.
    ei = torch.tensor([[2, 3, 2], [0, 0, 1]])

    gin = GINLayer(d, d).eval()
    with torch.no_grad():
        for layer in (gin.mlp[0], gin.mlp[2]):
            layer.weight.copy_(torch.eye(d))
            layer.bias.zero_()
    g = gin(x, ei)
    assert not torch.allclose(g[0], g[1]), "GIN sum failed to separate multisets"

    sage = SAGELayer(d, d, "mean").eval()
    agg = sage._aggregate(x, ei, 4)
    assert torch.allclose(agg[0], agg[1]), "mean aggregation unexpectedly separated"
    return "GIN sum separates {x,x} from {x}; mean aggregation does not"


def check_isolated_nodes_are_finite() -> str:
    """An isolated node must not produce NaN.

    Eq. 2 divides by |N(v)|, which is 0 for an isolated node. kNN similarity
    graphs built at a high cosine threshold, and PHEME threads with no replies,
    both produce isolated nodes routinely, so this is a live failure mode rather
    than a theoretical one.
    """
    d = 4
    x = torch.randn(5, d)
    ei = torch.tensor([[0, 1], [1, 0]])           # nodes 2, 3, 4 isolated
    for name, layer in {
        "GCN-paper": GCNLayer(d, 3, normalization="paper"),
        "GCN-sym": GCNLayer(d, 3),
        "GAT": GATLayer(d, 3, heads=2),
        "GATv2": GATv2Layer(d, 3, heads=2),
        "SAGE-mean": SAGELayer(d, 3, "mean"),
        "SAGE-lstm": SAGELayer(d, 3, "lstm"),
        "GIN": GINLayer(d, 3),
    }.items():
        layer.eval()
        out = layer(x, ei)
        assert torch.isfinite(out).all(), f"{name} produced non-finite output"
    return "isolated nodes give finite output in all 7 layer variants"


def check_gat_v2_is_dynamic() -> str:
    """GATv2 must rank neighbours differently for different receivers; GAT cannot.

    This is the survey's stated motivation for Eq. 5 (Sect. 3.1: the composed
    linear maps make GAT's attention "a monotonic function of the neighbors of
    a node rather than the node itself"). With a shared softmax removed, GAT's
    ranking of any two senders is identical for every receiver, because the
    receiver contributes an additive constant. GATv2's does not have to be.
    """
    torch.manual_seed(3)
    d, n = 6, 6
    x = torch.randn(n, d)
    # Brody et al.'s construction: a *shared* sender set, so the two receivers
    # rank the identical four neighbours and any difference is the model's.
    receivers, senders = [0, 1], [2, 3, 4, 5]
    src, dst = zip(*[(u, v) for u in senders for v in receivers])
    ei = torch.tensor([list(src), list(dst)])

    def rankings(layer):
        layer.eval()
        out = {}
        h_src = layer.lin(x).view(n, layer.heads, layer.out_dim)
        if isinstance(layer, GATv2Layer):
            h_dst = layer.lin_dst(x).view(n, layer.heads, layer.out_dim)
            e = torch.nn.functional.leaky_relu(
                h_dst[ei[1]] + h_src[ei[0]], layer.negative_slope)
            logits = (e * layer.att).sum(-1)[:, 0]
        else:
            logits = torch.nn.functional.leaky_relu(
                (h_src * layer.att_dst).sum(-1)[ei[1]]
                + (h_src * layer.att_src).sum(-1)[ei[0]], layer.negative_slope)[:, 0]
        for v in receivers:
            m = ei[1] == v
            out[v] = tuple(ei[0][m][torch.argsort(logits[m])].tolist())
        return out

    gat = rankings(GATLayer(d, 4, heads=1))
    assert len(set(gat.values())) == 1, "GAT ranking varied by receiver -- unexpected"

    v2 = rankings(GATv2Layer(d, 4, heads=1))
    assert len(set(v2.values())) > 1, "GATv2 ranking did not vary by receiver"
    return (f"GAT gives every receiver the same neighbour ranking "
            f"({len(set(gat.values()))} distinct); GATv2 gives {len(set(v2.values()))}")


def run_all_checks(verbose: bool = True) -> list[str]:
    """Run every property check; raise on the first violation."""
    results = []
    for fn in (check_permutation_equivariance, check_gcn_normalization,
               check_attention_is_a_distribution, check_gin_injectivity,
               check_isolated_nodes_are_finite, check_gat_v2_is_dynamic):
        msg = fn()
        results.append(f"{fn.__name__}: {msg}")
        if verbose:
            print(f"  OK  {msg}")
    return results
