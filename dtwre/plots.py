"""Figure reproduction for Tong et al. (2025).

Each function regenerates one published figure. Styling deliberately mirrors
the paper (default matplotlib colour cycle, same titles and axis labels) so a
reader can hold the notebook output next to the PDF and compare directly.

Figures 1-4 of the paper are hand-drawn schematics of the architecture and
workflow, not plots of data; they are described in the notebook but cannot be
"reproduced" from results. Figures 5-11 are data plots and are all covered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .config import METRICS, BASELINE_LABELS, FIGURES

__all__ = [
    "figure5_comparison_bars", "figure6_alpha", "figure7_lambda",
    "figure8_ratio", "figure9_temporal", "figure10_training_curves",
    "figure11_network", "entropy_diagnostics", "save",
]

METRIC_LABELS = {"auc": "AUC", "precision": "Precision", "recall": "Recall",
                 "f1": "F1-Score", "accuracy": "Accuracy"}


def save(fig, name: str, outdir: Path | None = None) -> Path:
    outdir = Path(outdir or FIGURES)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def figure5_comparison_bars(results: dict, order=None, ax=None):
    """Figure 5 -- grouped bars comparing each method across the five metrics.

    ``results`` maps method name -> metric dict. Dashed horizontal lines mark
    the DTWRE values, as in the published figure.
    """
    order = order or list(results)
    x = np.arange(len(order))
    width = 0.15

    ax = ax or plt.subplots(figsize=(9, 5))[1]
    for i, m in enumerate(METRICS):
        vals = [results[k][m] for k in order]
        ax.bar(x + (i - 2) * width, vals, width, label=METRIC_LABELS[m])

    if "dtwre" in results:
        for i, m in enumerate(METRICS):
            ax.axhline(results["dtwre"][m], ls=":", lw=0.8,
                       color=f"C{i}", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([BASELINE_LABELS.get(k, k) for k in order])
    ax.set_ylabel("Score")
    # The paper's fixed [0.75, 1.02] window suits its higher scores but would
    # clip bars here, hiding values entirely. Scale to the data instead.
    lo = min(results[k][m] for k in order for m in METRICS)
    hi = max(results[k][m] for k in order for m in METRICS)
    pad = max(0.02, 0.1 * (hi - lo))
    ax.set_ylim(max(0.0, lo - pad), min(1.02, hi + pad))
    ax.set_title("Performance Comparison Between Innovative Method "
                 "And Baseline Method")
    ax.legend(ncol=5, loc="upper center", fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.4)
    return ax.figure


def _sweep_plot(xs, results, xlabel, title, ax=None, logx=False):
    """Line plot per metric, with a +/-1 std band when the sweep was seeded."""
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    xs = np.asarray(xs, dtype=float)
    for i, m in enumerate(METRICS):
        ys = np.array([r[m] for r in results], dtype=float)
        ax.plot(xs, ys, marker=".", label=METRIC_LABELS[m], color=f"C{i}")
        if all(f"{m}_std" in r for r in results):
            sd = np.array([r[f"{m}_std"] for r in results], dtype=float)
            if np.any(sd > 0):
                ax.fill_between(xs, ys - sd, ys + sd, color=f"C{i}", alpha=0.15,
                                linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Performance Score")
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(ls=":", alpha=0.4)
    return ax.figure


def figure6_alpha(alphas, results, ax=None):
    """Figure 6 -- effect of the Rényi order alpha."""
    return _sweep_plot(alphas, results, "Renyi Entropy Order (α)",
                       "Impact of Renyi Entropy Order (α) on Model Performance",
                       ax)


def figure7_lambda(lams, results, ax=None):
    """Figure 7 -- effect of the temporal weighting parameter lambda."""
    return _sweep_plot(lams, results, "Time-Weighted Parameter (λ)",
                       "Impact of Time-Weighted Parameter λ on Model Performance",
                       ax)


def figure8_ratio(ratios, results, ax=None):
    """Figure 8 -- effect of the positive-to-negative sample ratio."""
    return _sweep_plot(ratios, results, "Positive-to-Negative Sample Ratio",
                       "Impact of Positive-to-Negative Sample Ratio on Model Performance",
                       ax)


def figure9_temporal(windows_seconds, results, ax=None):
    """Figure 9 -- effect of the temporal window length (plotted in days)."""
    days = [w / 86400.0 for w in windows_seconds]
    return _sweep_plot(days, results, "Time Window (Days)",
                       "Temporal Performance Evaluation", ax)


def figure10_training_curves(history, ax=None):
    """Figure 10 -- metric evolution across the 100 training epochs."""
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    epochs = [h["epoch"] for h in history]
    for m in METRICS:
        ax.plot(epochs, [h[m] for h in history], marker=".", ms=3,
                label=METRIC_LABELS[m])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.set_title("Training Metrics over Epochs")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(ls=":", alpha=0.4)
    return ax.figure


def figure11_network(graph, key_nodes, pred_pos_edges, neg_edges=None,
                     max_nodes=800, seed=42, ax=None):
    """Figure 11 -- centre-periphery view of the predicted propagation network.

    Red nodes are the high-entropy key nodes, red lines the edges predicted
    positive, grey lines the negatives -- matching the published caption.
    """
    ax = ax or plt.subplots(figsize=(9, 6))[1]

    nodes = list(graph.nodes())
    if len(nodes) > max_nodes:
        deg = dict(graph.degree())
        keep = set(sorted(nodes, key=lambda n: -deg[n])[:max_nodes])
        keep |= set(key_nodes)
        graph = graph.subgraph(keep).copy()

    pos = nx.spring_layout(graph, seed=seed, k=0.15, iterations=50)
    present = set(graph.nodes())

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=12,
                           node_color="tab:blue", linewidths=0)
    keys = [n for n in key_nodes if n in present]
    if keys:
        nx.draw_networkx_nodes(graph, pos, nodelist=keys, ax=ax,
                               node_size=28, node_color="darkred", linewidths=0)

    def _draw(edges, color, alpha, width):
        e = [(u, v) for u, v in edges if u in present and v in present]
        if e:
            nx.draw_networkx_edges(graph, pos, edgelist=e, ax=ax,
                                   edge_color=color, alpha=alpha, width=width)

    if neg_edges is not None:
        _draw(neg_edges, "grey", 0.25, 0.4)
    _draw(pred_pos_edges, "red", 0.55, 0.7)

    ax.set_title("Public Opinion Prediction Visualization\n"
                 "(Red Dot: Key Node, Blue Dot: Normal Node, "
                 "Red Line: Predicted Positive Edge, Gray Line: Negative Edge)",
                 fontsize=9)
    ax.axis("off")
    return ax.figure


def entropy_diagnostics(global_series, dtwre_global, lne_matrix=None, axes=None):
    """Supplementary view: Eq. 2 and Eq. 3 over time, plus the LNE spread.

    Not a published figure -- included so the notebook can show *what the
    entropy features actually look like* before they enter the model.
    """
    n = 3 if lne_matrix is not None else 2
    if axes is None:
        _, axes = plt.subplots(1, n, figsize=(5 * n, 3.6))
    axes = np.atleast_1d(axes)

    steps = np.arange(len(global_series))
    axes[0].plot(steps, global_series, marker="o", ms=3, color="tab:blue")
    axes[0].set_title("Eq. 2: global entropy $H_\\alpha^{global}(t)$")
    axes[0].set_xlabel("time step")

    axes[1].plot(steps, dtwre_global, marker="o", ms=3, color="tab:red")
    axes[1].set_title("Eq. 3: DTWRE $H_\\alpha^{time}(G,t)$")
    axes[1].set_xlabel("time step")

    if lne_matrix is not None:
        active = lne_matrix[:, lne_matrix.sum(axis=0) > 0]
        axes[2].imshow(active[:, :200], aspect="auto", cmap="magma")
        axes[2].set_title("Eq. 1: LNE per node and time step")
        axes[2].set_xlabel("node (first 200 active)")
        axes[2].set_ylabel("time step")

    for a in axes[:2]:
        a.grid(ls=":", alpha=0.4)
    return axes[0].figure
