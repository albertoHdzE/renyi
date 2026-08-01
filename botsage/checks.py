"""Property checks for the method of Deshmukh (2025).

The repository has no test suite by convention; correctness is asserted against
properties the mathematics must satisfy. These run from the notebook and from
``scripts/run_botsage.py``, and a violation fails the run rather than producing
a plausible table.

The first four checks establish, numerically, what Sect. 3.5's untrained
GraphSAGE layer can and cannot do. They are the analytical core of this
replication: everything the experiments later measure is a consequence of them.

Each function raises ``AssertionError`` on failure and returns a short string.
"""

from __future__ import annotations

import numpy as np
import torch

from .sage import SAGEConv, effective_input, mean_aggregate, sage_embeddings

__all__ = ["check_sage_matches_pyg_form", "check_untrained_sage_is_linear",
           "check_embedding_rank_is_bounded",
           "check_linear_svm_gains_nothing_from_projection",
           "check_permutation_equivariance", "check_isolated_nodes_are_zero",
           "check_untrained_output_depends_on_seed", "run_all_checks"]


def _toy_graph(n: int = 40, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n, d)), dtype=torch.float32)
    src = rng.integers(0, n, size=3 * n)
    dst = rng.integers(0, n, size=3 * n)
    keep = src != dst
    ei = torch.tensor(np.stack([src[keep], dst[keep]]), dtype=torch.long)
    return x, ei


def check_sage_matches_pyg_form(tol: float = 1e-5) -> str:
    """Our SAGEConv must equal ``W_l·mean(N(v)) + b + W_r·x_v`` exactly.

    This is PyTorch Geometric's ``SAGEConv`` with default arguments, which is
    what the paper instantiates. Checking the closed form rather than trusting
    the implementation means the rest of the analysis rests on the right object.
    """
    x, ei = _toy_graph()
    conv = SAGEConv(5, 12).eval()
    with torch.no_grad():
        got = conv(x, ei)
        want = (conv.lin_l(mean_aggregate(x, ei)) + conv.lin_r(x))
    err = (got - want).abs().max().item()
    assert err < tol, f"SAGEConv does not match the PyG form (err={err:.2e})"
    return f"SAGEConv == W_l·mean(N(v)) + b + W_r·x_v (err < {tol:g})"


def check_untrained_sage_is_linear(tol: float = 1e-4) -> str:
    r"""The layer is affine in ``[x_v || mean of N(v)]``.

    With no activation function -- and Sect. 3.5 describes none, since a single
    SAGEConv is used purely to "generate embeddings" -- the layer is
    :math:`h = W z + b` for :math:`z = [\,x_v \| \overline{x}_{N(v)}\,]`. We
    verify by reconstructing the output from ``weight_matrix()`` alone.
    """
    x, ei = _toy_graph()
    conv = SAGEConv(5, 128).eval()
    with torch.no_grad():
        got = conv(x, ei)
        z = effective_input(x, ei)
        want = z @ conv.weight_matrix().T + conv.lin_l.bias
    err = (got - want).abs().max().item()
    assert err < tol, f"layer is not affine in the 10-dim input (err={err:.2e})"
    return "the untrained layer is exactly affine in the 10 numbers [x_v || mean N(v)]"


def check_embedding_rank_is_bounded() -> str:
    """The 128-dim embedding has rank at most 2 x n_features = 10.

    This is the quantitative form of the paper's central weakness. Sect. 3.6
    describes the concatenation as producing "a rich blend of network and text
    features" of size 896. In fact the network half spans a subspace of
    dimension at most 10, so 118 of its 128 columns are linearly dependent on
    the others and carry no information a linear classifier can use.
    """
    x, ei = _toy_graph(n=200)
    emb = sage_embeddings(x, ei, out_channels=128, seed=0)
    centred = (emb - emb.mean(0, keepdim=True)).double()
    bound = 2 * x.size(1)

    # A *relative* tolerance is the right test: the embedding is computed in
    # float32, so the null-space singular values sit at the rounding floor
    # rather than at zero. The evidence is the gap, not the absolute value.
    rank = int(torch.linalg.matrix_rank(centred, rtol=1e-5))
    sv = torch.linalg.svdvals(centred)
    gap = float(sv[bound - 1] / sv[bound])

    assert rank <= bound, f"rank {rank} exceeds the 2*d bound {bound}"
    assert emb.shape[1] == 128, "embedding is not 128-dimensional"
    assert gap > 1e4, f"no clear spectral gap at index {bound} (ratio {gap:.1e})"
    return (f"128-dim embedding has rank {rank} (bound 2*{x.size(1)}={bound}); "
            f"sigma_{bound}/sigma_{bound + 1} = {gap:.1e}, so "
            f"{128 - rank} of 128 dimensions are redundant")


def check_linear_svm_gains_nothing_from_projection(tol: float = 0.02) -> str:
    """A linear model on the 128-dim embedding ~= one on the 10 raw dimensions.

    Because the embedding is an affine image of ``effective_input``, any linear
    decision boundary expressible on one is expressible on the other. The two
    are not *identical* in practice -- L2 regularisation is not invariant under
    a change of basis, so the random projection reshapes the penalty -- but the
    achievable accuracy should match closely.

    If this holds, the GraphSAGE stage of the pipeline is a reparameterisation,
    not a feature extractor.
    """
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(0)
    n = 600
    x, ei = _toy_graph(n=n, seed=1)
    z = effective_input(x, ei).numpy()
    # A label that genuinely depends on both self and neighbourhood features.
    w = rng.normal(size=z.shape[1])
    y = ((z @ w + rng.normal(scale=0.5, size=n)) > 0).astype(int)

    emb = sage_embeddings(x, ei, out_channels=128, seed=0).numpy()

    def score(features):
        clf = make_pipeline(StandardScaler(),
                            LinearSVC(C=1.0, max_iter=5000))
        return cross_val_score(clf, features, y, cv=5, scoring="accuracy").mean()

    a_emb, a_raw = score(emb), score(z)
    assert abs(a_emb - a_raw) < tol, (
        f"embedding {a_emb:.3f} vs raw 10-dim {a_raw:.3f} differ by more than {tol}")
    return (f"linear SVM: 128-dim embedding {a_emb:.3f} vs 10 raw dims "
            f"{a_raw:.3f} (|delta| < {tol})")


def check_permutation_equivariance(tol: float = 1e-4) -> str:
    """Relabelling nodes must permute the embedding rows identically."""
    x, ei = _toy_graph(n=30, seed=2)
    conv = SAGEConv(5, 16).eval()
    perm = torch.randperm(30, generator=torch.Generator().manual_seed(3))
    inv = torch.argsort(perm)
    with torch.no_grad():
        out = conv(x, ei)
        out_p = conv(x[perm], inv[ei])
    err = (out[perm] - out_p).abs().max().item()
    assert err < tol, f"SAGEConv is not permutation-equivariant (err={err:.2e})"
    return f"SAGEConv is permutation-equivariant (err < {tol:g})"


def check_isolated_nodes_are_zero() -> str:
    """An isolated node aggregates to zeros, so its embedding is ``W_r x_v + b``.

    Not a corner case here: TwiBot-20's released graph has 229,580 nodes and
    227,979 edges, so a large majority of nodes have no neighbours at all. For
    those users the "graph embedding" is a random linear map of their own five
    features -- no network information whatsoever.
    """
    x = torch.randn(5, 4)
    ei = torch.tensor([[0, 1], [1, 0]])          # nodes 2, 3, 4 isolated
    agg = mean_aggregate(x, ei)
    assert torch.allclose(agg[2:], torch.zeros(3, 4)), "isolated agg is not zero"

    conv = SAGEConv(4, 8).eval()
    with torch.no_grad():
        out = conv(x, ei)
        want = conv.lin_r(x[2:]) + conv.lin_l.bias
    assert torch.allclose(out[2:], want, atol=1e-5), "isolated node output mismatch"
    assert torch.isfinite(out).all(), "non-finite output"
    return "isolated nodes aggregate to zeros; their embedding is W_r·x_v + b only"


def check_untrained_output_depends_on_seed() -> str:
    """Two seeds give different embeddings -- so the result is seed-dependent.

    The paper fixes no seed and reports none. For a *trained* model the seed is
    a nuisance parameter that training largely washes out; for an untrained one
    it is the entire model. This check simply establishes that the dependence is
    real, so ``suite_seed_sensitivity`` has something to measure.
    """
    x, ei = _toy_graph(n=100, seed=4)
    a = sage_embeddings(x, ei, out_channels=128, seed=0)
    b = sage_embeddings(x, ei, out_channels=128, seed=1)
    diff = (a - b).abs().mean().item()
    assert diff > 1e-3, "two seeds produced the same embedding"

    # But the *column space* is the same 10-dim subspace either way, so the
    # information content does not change with the seed -- only its coordinates.
    ra = int(torch.linalg.matrix_rank((a - a.mean(0)).double(), rtol=1e-5))
    rb = int(torch.linalg.matrix_rank((b - b.mean(0)).double(), rtol=1e-5))
    assert ra == rb, f"rank changed with seed ({ra} vs {rb})"
    return (f"seed changes the embedding (mean |delta| {diff:.3f}) but not its "
            f"rank ({ra}): same information, different coordinates")


def run_all_checks(verbose: bool = True) -> list[str]:
    """Run every property check; raise on the first violation."""
    results = []
    for fn in (check_sage_matches_pyg_form,
               check_untrained_sage_is_linear,
               check_embedding_rank_is_bounded,
               check_linear_svm_gains_nothing_from_projection,
               check_permutation_equivariance,
               check_isolated_nodes_are_zero,
               check_untrained_output_depends_on_seed):
        msg = fn()
        results.append(f"{fn.__name__}: {msg}")
        if verbose:
            print(f"  OK  {msg}")
    return results
