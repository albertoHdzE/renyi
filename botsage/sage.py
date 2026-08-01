r"""GraphSAGE (Sect. 3.5), implemented to match PyTorch Geometric's ``SAGEConv``.

The paper uses PyG's ``SAGEConv(in_channels=5, out_channels=128)`` and, per
Sect. 3.5, **never trains it**:

    "GraphSage is only used with the intention of generating embeddings for
     nodes. Hence, training epochs and optimization tasks are not required due
     to the lack of a prediction head... With the model in evaluation mode,
     embeddings are generated in a single forward pass."

That sentence is the most consequential in the paper, so this module implements
the layer exactly and then makes its consequences measurable.

**What an untrained SAGEConv actually computes.** With PyG's defaults
(``aggr="mean"``, ``root_weight=True``, ``normalize=False``):

.. math::
    h_v = W_l \cdot \frac{1}{|N(v)|}\sum_{u \in N(v)} x_u + b + W_r \cdot x_v

which is an affine map of the concatenation :math:`[\,x_v \,\|\,
\overline{x}_{N(v)}\,] \in \mathbb{R}^{10}` into :math:`\mathbb{R}^{128}`. It is
**linear**, and its matrix has rank at most 10. So the 128-dimensional
"embedding" carries no more information than those ten numbers, and a *linear*
SVM on top of it can express nothing that a linear SVM on the ten numbers
cannot. ``checks.py`` verifies both claims numerically.

This is not a criticism of GraphSAGE, which is trained in Hamilton et al.
(2017). It is a property of using it untrained. ``TrainedSAGE`` is provided so
the notebook can quantify what training would have added.

PyTorch Geometric is deliberately not a dependency, per the repository's tooling
policy; the layer is 30 lines of core PyTorch and reproducing it here also lets
the initialisation be pinned exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["SAGEConv", "TrainedSAGE", "sage_embeddings", "mean_aggregate",
           "effective_input"]


def mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean of each node's neighbours' features; isolated nodes give zeros.

    ``edge_index`` is ``(2, E)`` with column ``(u, v)`` meaning "u is a
    neighbour of v", so ``edge_index[1]`` indexes the receiver. Isolated nodes
    returning **zeros** rather than their own features matches PyG, and matters
    here: TwiBot-20's graph leaves most of its 229,580 nodes isolated.
    """
    n = x.size(0)
    out = torch.zeros(n, x.size(1), dtype=x.dtype, device=x.device)
    if edge_index.numel() == 0:
        return out
    src, dst = edge_index[0], edge_index[1]
    out.index_add_(0, dst, x.index_select(0, src))
    count = torch.zeros(n, dtype=x.dtype, device=x.device)
    count.index_add_(0, dst, torch.ones(dst.numel(), dtype=x.dtype,
                                        device=x.device))
    return out / count.clamp(min=1).unsqueeze(-1)


def effective_input(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """``[x_v || mean of N(v)]`` -- the 10 numbers SAGEConv actually sees.

    Everything the untrained layer can express is an affine function of this.
    The notebook trains an SVM directly on it to show the 128-dim embedding adds
    nothing.
    """
    return torch.cat([x, mean_aggregate(x, edge_index)], dim=1)


class SAGEConv(nn.Module):
    r"""One GraphSAGE convolution, matching ``torch_geometric.nn.SAGEConv``.

    Defaults are PyG's, because the paper uses PyG's defaults: mean aggregation,
    a root weight, no L2 normalisation, bias on the neighbour branch only.

    Initialisation is PyG's too: ``nn.Linear.reset_parameters``, i.e. Kaiming
    uniform with ``a = sqrt(5)``, and a bias drawn from
    :math:`U(-1/\sqrt{fan\_in}, 1/\sqrt{fan\_in})`. Since the paper never trains
    the layer, this initialisation *is* the model, so it is reproduced exactly
    rather than approximated.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 aggr: str = "mean", root_weight: bool = True,
                 normalize: bool = False, bias: bool = True):
        super().__init__()
        if aggr not in ("mean", "sum", "max"):
            raise ValueError(f"unsupported aggr {aggr!r}")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.aggr, self.normalize, self.root_weight = aggr, normalize, root_weight

        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = (nn.Linear(in_channels, out_channels, bias=False)
                      if root_weight else None)
        self.reset_parameters()

    def reset_parameters(self):
        """PyG's reset: plain ``nn.Linear`` defaults on both branches."""
        for lin in (self.lin_l, self.lin_r):
            if lin is None:
                continue
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            if lin.bias is not None:
                fan_in = lin.weight.size(1)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(lin.bias, -bound, bound)

    def _aggregate(self, x, edge_index):
        if self.aggr == "mean":
            return mean_aggregate(x, edge_index)
        n = x.size(0)
        if self.aggr == "sum":
            out = torch.zeros(n, x.size(1), dtype=x.dtype, device=x.device)
            if edge_index.numel():
                out.index_add_(0, edge_index[1],
                               x.index_select(0, edge_index[0]))
            return out
        out = torch.full((n, x.size(1)), float("-inf"), dtype=x.dtype,
                         device=x.device)
        if edge_index.numel():
            idx = edge_index[1].unsqueeze(-1).expand(-1, x.size(1))
            out = out.scatter_reduce(0, idx, x.index_select(0, edge_index[0]),
                                     reduce="amax", include_self=True)
        return out.masked_fill(torch.isinf(out), 0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.lin_l(self._aggregate(x, edge_index))
        if self.lin_r is not None:
            out = out + self.lin_r(x)
        if self.normalize:
            out = torch.nn.functional.normalize(out, p=2.0, dim=-1)
        return out

    def weight_matrix(self) -> torch.Tensor:
        """The single ``(out, 2*in)`` matrix this layer applies to ``effective_input``.

        Making it explicit is the point: the layer *is* this matrix, and its rank
        bounds the information the embedding can carry.
        """
        if self.lin_r is None:
            return self.lin_l.weight.detach()
        return torch.cat([self.lin_r.weight.detach(),
                          self.lin_l.weight.detach()], dim=1)


class TrainedSAGE(nn.Module):
    """GraphSAGE *with* a prediction head -- what the paper chose not to build.

    Used only by ``experiments.suite_trained_vs_untrained`` to quantify what the
    untrained layer gives up. Two layers with ReLU between them, then a linear
    classifier, trained with cross-entropy on the training fold only.
    """

    def __init__(self, in_channels: int, hidden: int = 128, n_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        d = in_channels
        for _ in range(num_layers):
            self.convs.append(SAGEConv(d, hidden))
            d = hidden
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def embed(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = torch.relu(h)
                h = self.dropout(h)
        return h

    def forward(self, x, edge_index):
        return self.head(self.embed(x, edge_index))


@torch.no_grad()
def sage_embeddings(x: torch.Tensor, edge_index: torch.Tensor,
                    out_channels: int = 128, seed: int = 0,
                    **kwargs) -> torch.Tensor:
    """Sect. 3.5 verbatim: one untrained forward pass, eval mode, no gradients.

    ``seed`` fixes the initialisation. The paper does not report one, and
    because the layer is never trained the seed fully determines the output --
    see ``experiments.suite_seed_sensitivity``.
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
    conv = SAGEConv(x.size(1), out_channels, **kwargs)
    conv.eval()
    return conv(x, edge_index)
