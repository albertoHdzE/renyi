"""Every figure: the survey's own six, plus the ones its tables imply.

Three groups.

**Reproductions of Figs. 1-6.** All six originals are diagrams. They are
redrawn from the structures in ``taxonomy.py`` rather than traced, so the
picture is a rendering of the data and a transcription error shows up as a wrong
figure. Fig. 3 goes further: its computation tree is *unrolled from the
adjacency*, so it is generated rather than copied.

**Meta-analysis of Tables 1-2.** Sect. 5.3.2 draws four conclusions from those
tables in prose and plots none of them. These are those plots.

**Results of the live experiments**, including a comparison against the
Performance column that puts our numbers in the range the literature reports.

Colours come from the validated categorical palette in the dataviz reference
(slots 1-5, adjacent pairlist, light mode). Three of those slots fall below 3:1
against the surface, so every categorical mark here also carries a direct value
label -- the relief the validator requires, and the labelling a reader wants on
a comparison chart regardless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import FIGURES
from .taxonomy import FIG1_STAGES, FIG3_EXAMPLE, Node

__all__ = ["setup_style", "save", "SERIES", "fig1_framework", "fig2_false_information",
           "fig3_gnn_schematic", "fig4_features", "fig5_approaches",
           "fig6_algorithms", "draw_taxonomy", "plot_gnn_usage",
           "plot_graph_type_usage", "plot_feature_usage", "plot_year_trend",
           "plot_dataset_usage", "plot_reported_performance",
           "plot_performance_heatmap", "plot_claims", "plot_gnn_comparison",
           "plot_graph_comparison", "plot_vs_literature", "plot_confusion",
           "plot_ablation"]


# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e3e2dd"

# Categorical slots 1-5 of the validated reference palette (light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Single hue for magnitude-by-category bars, where colour carries no identity.
MAGNITUDE = "#2a78d6"
# Sequential ramp for heatmaps: one hue, light -> dark.
SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list(
    "disinfo_seq", ["#eef4fc", "#c5daf5", "#8fb8ea", "#5495dc", "#2a78d6",
                    "#1b5296"])


def setup_style() -> None:
    """House matplotlib style: recessive axes, no chartjunk, readable defaults."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
    })


def save(fig, name: str, outdir: Path | None = None) -> Path:
    outdir = Path(outdir or FIGURES)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path)
    return path


def _bar_labels(ax, bars, fmt="{:.0f}", pad=0.01, horizontal=False):
    """Direct value labels -- the relief the palette validator requires."""
    span = (ax.get_xlim()[1] if horizontal else ax.get_ylim()[1])
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v + span * pad, b.get_y() + b.get_height() / 2, fmt.format(v),
                    va="center", ha="left", fontsize=9, color=INK_2)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + span * pad, fmt.format(v),
                    ha="center", va="bottom", fontsize=9, color=INK_2)


def _source(ax, text):
    ax.annotate(text, xy=(0, -0.16), xycoords="axes fraction",
                fontsize=8, color=INK_MUTED, va="top")


# ==========================================================================
# Part 1 -- reproductions of the survey's own figures
# ==========================================================================

def fig1_framework(figsize=(12, 2.9)):
    """Fig. 1: the general framework of disinformation detection using GNNs."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)

    # The input stack of documents on the left of the original.
    for i, (dx, dy) in enumerate([(0, 0), (1.6, -1.6), (3.2, -3.2)]):
        ax.add_patch(FancyBboxPatch((3 + dx, 12 + dy), 7, 9,
                                    boxstyle="round,pad=0.3,rounding_size=0.4",
                                    fc="#ffffff", ec=INK_2, lw=1.2, zorder=3 - i))
    ax.text(7.5, 8.0, "information\nitems  $d_i$", ha="center", va="top",
            fontsize=9, color=INK_2)

    x0, w, gap = 18.0, 15.0, 3.5
    for i, stage in enumerate(FIG1_STAGES):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 11), w, 10,
                                    boxstyle="round,pad=0.3,rounding_size=0.6",
                                    fc="#ffffff", ec=INK, lw=1.6))
        ax.text(x + w / 2, 16, stage["name"], ha="center", va="center",
                fontsize=11, weight="semibold", color=INK)
        ax.text(x + w / 2, 8.6, f"Sect. {stage['section']}", ha="center",
                va="center", fontsize=8.5, color=INK_MUTED)
        ax.add_patch(FancyArrowPatch((x - gap + 0.4, 16), (x - 0.6, 16),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=INK_2, lw=1.4))

    x_end = x0 + 4 * (w + gap) - gap
    ax.add_patch(FancyArrowPatch((x_end + 0.4, 16), (x_end + 3.4, 16),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=INK_2, lw=1.4))
    ax.plot([x_end + 4.2, x_end + 5.4, x_end + 5.4, x_end + 4.2],
            [21, 20, 12, 11], color=INK_2, lw=1.4, solid_joinstyle="round")
    ax.text(x_end + 6.4, 20, "Fake", va="center", fontsize=10, color=INK)
    ax.text(x_end + 6.4, 12, "Real", va="center", fontsize=10, color=INK)

    ax.set_title("Fig. 1  The general framework of disinformation detection "
                 "using GNNs", pad=8)
    fig.tight_layout()
    return fig


# ---- generic taxonomy renderer, used for Figs. 2, 4, 5 and 6 ----

_CHAR_W = 0.62          # approximate width of one character, in box units
_BOX_H = 1.0
_V_GAP = 1.35
_H_PAD = 1.9
_H_GAP = 0.6            # clear space between sibling boxes
_REF_H = 0.78


def _node_width(node: Node) -> float:
    label_w = len(node.label) * _CHAR_W + _H_PAD
    if node.refs:
        ref_w = max(len(f"[{i}]-{y}") for i, y in node.refs) * _CHAR_W + _H_PAD
        return max(label_w, ref_w)
    return label_w


def _layout(node: Node, depth: int, cursor: float, pos: dict, fontsize: float):
    """Assign (x, depth) to every node; leaves are packed left to right."""
    if not node.children:
        w = _node_width(node)
        pos[id(node)] = (cursor + w / 2, depth, w)
        return cursor + w + _H_GAP
    start = cursor
    for c in node.children:
        cursor = _layout(c, depth + 1, cursor, pos, fontsize)
    xs = [pos[id(c)][0] for c in node.children]
    own = _node_width(node)
    centre = (min(xs) + max(xs)) / 2
    pos[id(node)] = (centre, depth, own)
    return max(cursor, start + own + _H_GAP)


def _draw_node(ax, node: Node, x, y, w, fontsize, fc="#ffffff"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - _BOX_H / 2), w, _BOX_H,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=INK_2, lw=1.0, zorder=3))
    ax.text(x, y, node.label, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=4)


def draw_taxonomy(root: Node, title: str, figsize=None, fontsize=8.5,
                  show_refs=True):
    """Render a taxonomy tree in the survey's boxed-and-elbowed style.

    Reference lists in Figs. 5 and 6 hang below their category as a vertical
    stack, matching the original layout. A trailing "..." box marks a list the
    authors truncated.
    """
    setup_style()
    pos: dict = {}
    total_w = _layout(root, 0, 0.0, pos, fontsize)
    max_depth = max(d for _, d, _ in pos.values())

    # Vertical room for the deepest reference stack.
    max_refs = max((len(n.refs) + (1 if n.truncated else 0)
                    for n in root if n.refs), default=0) if show_refs else 0
    total_h = (max_depth + 1) * _V_GAP + max_refs * _REF_H + 1.5

    if figsize is None:
        figsize = (min(max(total_w * 0.34, 7), 19), max(total_h * 0.46, 3))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(-1, total_w + 1)
    ax.set_ylim(-total_h + _V_GAP, _V_GAP)

    def y_of(depth):
        return -depth * _V_GAP

    # Elbow connectors: down from the parent, across, then down into the child.
    for node in root:
        if not node.children:
            continue
        px, pd, _ = pos[id(node)]
        py = y_of(pd)
        mid = py - _V_GAP / 2
        ax.plot([px, px], [py - _BOX_H / 2, mid], color=INK_2, lw=0.9, zorder=1)
        xs = [pos[id(c)][0] for c in node.children]
        ax.plot([min(xs), max(xs)], [mid, mid], color=INK_2, lw=0.9, zorder=1)
        for c in node.children:
            cx, cd, _ = pos[id(c)]
            ax.plot([cx, cx], [mid, y_of(cd) + _BOX_H / 2], color=INK_2,
                    lw=0.9, zorder=1)

    for node in root:
        x, d, w = pos[id(node)]
        _draw_node(ax, node, x, y_of(d), w, fontsize,
                   fc="#f2f5fa" if d == 0 else "#ffffff")

        if show_refs and node.refs:
            entries = [f"[{i}]-{y}" for i, y in node.refs]
            if node.truncated:
                entries.append("...")
            top = y_of(d) - _BOX_H / 2 - 0.45
            ax.plot([x, x], [y_of(d) - _BOX_H / 2, top], color=INK_2, lw=0.9,
                    zorder=1)
            for j, txt in enumerate(entries):
                yy = top - j * _REF_H - _REF_H / 2
                ax.add_patch(FancyBboxPatch((x - w / 2 + 0.25, yy - _REF_H / 2 + 0.06),
                                            w - 0.5, _REF_H - 0.12,
                                            boxstyle="round,pad=0.01,rounding_size=0.1",
                                            fc="#ffffff", ec=INK_MUTED, lw=0.7,
                                            zorder=3))
                ax.text(x, yy, txt, ha="center", va="center",
                        fontsize=fontsize - 1.2, color=INK_2, zorder=4)

    ax.set_title(title, pad=10)
    fig.tight_layout()
    return fig


def fig2_false_information():
    from .taxonomy import FIG2_FALSE_INFORMATION
    return draw_taxonomy(FIG2_FALSE_INFORMATION,
                         "Fig. 2  Categorization of various types of false "
                         "information", fontsize=9)


def fig4_features():
    from .taxonomy import FIG4_FEATURES
    return draw_taxonomy(FIG4_FEATURES,
                         "Fig. 4  Different types of features used in the "
                         "literature for disinformation detection", fontsize=9)


def fig5_approaches():
    from .taxonomy import FIG5_APPROACHES
    return draw_taxonomy(FIG5_APPROACHES,
                         "Fig. 5  A categorization of combating disinformation "
                         "approaches   ([i]-j: reference i, year j)")


def fig6_algorithms():
    from .taxonomy import FIG6_ALGORITHMS
    return draw_taxonomy(FIG6_ALGORITHMS,
                         "Fig. 6  Algorithm-based categorization of "
                         "disinformation detection approaches   "
                         "([i]-j: reference i, year j)", fontsize=8)


def fig3_gnn_schematic(figsize=(13, 5.2)):
    """Fig. 3: a graph, and the 3-layer computation tree unrolled from it.

    The right-hand tree is *computed* by expanding N(v) layer by layer, not
    transcribed. That makes the figure a check on the message-passing semantics
    of Eq. 1: if the unrolling were wrong, the picture would not match the
    paper's.
    """
    setup_style()
    spec = FIG3_EXAMPLE
    nodes, edges, root, L = (spec["nodes"], spec["edges"], spec["root"],
                             spec["layers"])
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    colours = dict(zip(nodes, ["#cfc9e8", "#d3e3c4", "#bfe0ec", "#f6d5b8",
                              "#eec4c0"]))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [1, 2.1]})
    for ax in (axl, axr):
        ax.set_axis_off()

    # ---- left: the graph, with N(a) circled ----
    layout = {"a": (0.5, 0.92), "b": (0.16, 0.58), "c": (0.84, 0.58),
              "d": (0.28, 0.16), "e": (0.72, 0.16)}
    for a, b in edges:
        axl.plot(*zip(layout[a], layout[b]), color=INK, lw=1.4, zorder=1)
    axl.add_patch(plt.Circle((0.5, 0.74), 0.36, fill=False, ec=INK_MUTED,
                             ls=(0, (4, 3)), lw=1.2, zorder=0))
    for n, (x, y) in layout.items():
        axl.add_patch(plt.Circle((x, y), 0.075, fc=colours[n], ec=INK_2,
                                 lw=1.1, zorder=2))
        axl.text(x, y, n, ha="center", va="center", fontsize=11, zorder=3)
    axl.set_xlim(-0.05, 1.05)
    axl.set_ylim(0.02, 1.12)
    axl.set_title("a graph, with $N(a)$ circled", fontsize=10)

    # ---- right: unroll the computation tree for `root` ----
    # Layer L holds the root; layer l-1 holds, for each node at layer l, that
    # node together with its neighbours -- exactly Eq. 1 read bottom-up.
    levels = [[root]]
    for _ in range(L - 1):
        prev = levels[-1]
        levels.append([m for v in prev for m in [v] + sorted(adj[v])])
    levels.reverse()                       # levels[0] is the input layer

    for depth, level in enumerate(levels):
        y = depth / max(L - 1, 1)
        axr.axhspan(y - 0.11, y + 0.11, color="#f1f0ec", zorder=0)
        axr.text(1.045, y, f"Layer {depth + 1}", va="center", fontsize=10,
                 color=INK_2)
        n = len(level)
        for i, name in enumerate(level):
            x = (i + 0.5) / n
            # scatter, not Circle: the axes are not equal-aspect, and a Circle
            # in data coordinates would render as an ellipse.
            axr.scatter([x], [y], s=430, c=colours[name], edgecolor=INK_2,
                        linewidth=1.0, zorder=3)
            axr.text(x, y, name, ha="center", va="center", fontsize=9, zorder=4)

    # Rhombi: one aggregation per receiving node, fed by its own layer's block.
    for depth in range(len(levels) - 1):
        lower, upper = levels[depth], levels[depth + 1]
        y_low, y_up = depth / max(L - 1, 1), (depth + 1) / max(L - 1, 1)
        block = len(lower) // len(upper)
        for j, _ in enumerate(upper):
            xj = (j + 0.5) / len(upper)
            ym = (y_low + y_up) / 2
            axr.plot([xj], [ym], marker="D", ms=15, mfc="#b9b8b2", mec=INK_2,
                     mew=1.0, zorder=3)
            axr.annotate("", xy=(xj, y_up - 0.045), xytext=(xj, ym + 0.03),
                         arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.1))
            # Plain elbows into the rhombus; the only arrowhead on this edge is
            # the one leaving the rhombus for the layer above.
            for i in range(j * block, (j + 1) * block):
                xi = (i + 0.5) / len(lower)
                axr.plot([xi, xi, xj], [y_low + 0.045, ym, ym],
                         color=INK_2, lw=0.9, zorder=2,
                         solid_joinstyle="round")

    axr.set_xlim(-0.03, 1.14)
    axr.set_ylim(-0.16, 1.16)
    axr.set_title(f"the embedding of node '{root}' from a {L}-layer GNN "
                  "(unrolled from the graph)", fontsize=10)

    fig.suptitle("Fig. 3  The general scheme of a graph neural network",
                 x=0.02, ha="left", fontsize=12, weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ==========================================================================
# Part 2 -- the meta-analysis Tables 1-2 imply but the survey never plots
# ==========================================================================

def _explode_counts(methods, field):
    vals = []
    for m in methods:
        vals.extend(m[field])
    return pd.Series(vals).value_counts()


def plot_gnn_usage(methods, figsize=(7, 3.6)):
    """Sect. 5.3.2: "GCN and GAT stand out as the most widely utilized"."""
    setup_style()
    counts = _explode_counts(methods, "gnn")
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index, counts.values, color=MAGNITUDE, width=0.62)
    ax.set_ylabel("methods citing this architecture")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="x", visible=False)
    _bar_labels(ax, bars)
    ax.set_title("Which GNN architectures the literature uses\n"
                 "Tables 1-2, counted per mention (a method may use several)")
    _source(ax, "Source: Lakzaei et al. (2024), Tables 1-2, n=34 methods. "
                "Survey claim: 'GCN and GAT stand out as the most widely utilized' (Sect. 5.3.2).")
    fig.tight_layout()
    return fig


def plot_graph_type_usage(methods, figsize=(7, 3.6)):
    """Sect. 5.3.2: "The majority of methods employ the propagation graph"."""
    setup_style()
    counts = pd.Series([m["graph"] for m in methods]).value_counts()
    total = len(methods)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=MAGNITUDE,
                   height=0.6)
    ax.set_xlabel("number of methods")
    ax.set_xlim(0, counts.max() * 1.22)
    ax.grid(axis="y", visible=False)
    for b, v in zip(bars, counts.values[::-1]):
        ax.text(v + total * 0.015, b.get_y() + b.get_height() / 2,
                f"{v}  ({v / total:.0%})", va="center", fontsize=9, color=INK_2)
    ax.set_title("How the graph is constructed\n"
                 "the survey's own novel axis of comparison (Sect. 5.3)")
    _source(ax, f"Source: Tables 1-2, n={total} methods. "
                "Propagation graphs are homogeneous and per-item; similarity "
                "graphs join all items into one graph.")
    fig.tight_layout()
    return fig


def plot_feature_usage(methods, figsize=(7.5, 3.6)):
    """Sect. 5.3.2: textual dominates; comments/semantic/temporal neglected."""
    setup_style()
    counts = _explode_counts(methods, "features")
    total = len(methods)
    neglected = {"Comments", "Semantic", "Temporal"}
    colors = ["#c9c8c2" if k in neglected else MAGNITUDE for k in counts.index]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.62)
    ax.set_ylabel("methods using this feature")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="x", visible=False)
    _bar_labels(ax, bars)
    ax.set_title("Which features the literature uses\n"
                 "grey: the three the survey calls under-used")
    _source(ax, f"Source: Tables 1-2, n={total} methods, counted per mention. "
                "Visual features appear in only 2 of 34 despite Sect. 7 "
                "listing them as an open problem.")
    fig.tight_layout()
    return fig


def plot_year_trend(methods, figsize=(7, 3.6)):
    """Sect. 5.3.2: first GNN work in 2019, "popularity... on the rise"."""
    setup_style()
    counts = pd.Series([m["year"] for m in methods]).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index.astype(str), counts.values, color=MAGNITUDE,
                  width=0.6)
    ax.set_ylabel("methods published")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.grid(axis="x", visible=False)
    _bar_labels(ax, bars)
    ax.set_title("GNN-based disinformation detection by year\n"
                 "the field is five years old in this survey")
    _source(ax, "Source: Tables 1-2 Year column, n=34. The 2023 count is a "
                "partial year: the survey was accepted 4 January 2024.")
    fig.tight_layout()
    return fig


def plot_dataset_usage(long_df, figsize=(7.5, 4.2)):
    """Which benchmarks the field actually converged on."""
    setup_style()
    counts = long_df["dataset"].value_counts()
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=MAGNITUDE,
                   height=0.68)
    ax.set_xlabel("number of reported results")
    ax.set_xlim(0, counts.max() * 1.15)
    ax.grid(axis="y", visible=False)
    _bar_labels(ax, bars, horizontal=True)
    ax.set_title("Which datasets the literature reports on\n"
                 "Twitter15/16 and Weibo carry most of the field's evidence")
    _source(ax, "Source: Tables 1-2, one row per (method, dataset) pair, "
                "n=69 reported results.")
    fig.tight_layout()
    return fig


def plot_reported_performance(long_df, min_n=3, figsize=(8.5, 4.4)):
    """Spread of reported accuracy per dataset -- the comparison the tables invite.

    Only ACC rows, only datasets with at least ``min_n`` of them. The spread
    within a dataset is the story: on PHEME the reported range is wider than the
    gap between most architectures, which is why the survey's own tables cannot
    rank methods.
    """
    setup_style()
    acc = long_df[long_df["metric"] == "ACC"]
    keep = acc["dataset"].value_counts()
    keep = keep[keep >= min_n].index
    acc = acc[acc["dataset"].isin(keep)]
    order = acc.groupby("dataset")["value"].median().sort_values().index

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    for i, dset in enumerate(order):
        vals = acc.loc[acc["dataset"] == dset, "value"].values
        ax.plot([vals.min(), vals.max()], [i, i], color=GRID, lw=6,
                solid_capstyle="round", zorder=1)
        ax.scatter(vals, i + rng.uniform(-0.13, 0.13, len(vals)),
                   s=58, color=MAGNITUDE, alpha=0.85, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)
        ax.text(vals.max() + 0.012, i, f"n={len(vals)}   "
                f"range {vals.max() - vals.min():.2f}",
                va="center", fontsize=8.5, color=INK_2)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("reported accuracy")
    ax.set_xlim(0.4, 1.14)
    ax.grid(axis="y", visible=False)
    ax.set_title("Accuracies reported per dataset\n"
                 "each dot is one published number, as printed in Tables 1-2")
    _source(ax, "Source: Tables 1-2, ACC rows only. No paper reports a "
                "standard deviation or states its split, so within-dataset "
                "spread cannot be attributed to architecture.")
    fig.tight_layout()
    return fig


def plot_performance_heatmap(long_df, figsize=(8.5, 4.4)):
    """Mean reported accuracy by dataset and GNN family.

    Sequential single hue: the value is a magnitude, not an identity. Cells with
    no evidence are left blank rather than imputed -- the emptiness is the
    finding, since it shows how little of the grid the field has covered.
    """
    setup_style()
    acc = long_df[long_df["metric"] == "ACC"].copy()
    acc["family"] = acc["gnn"].str.split(", ").str[0]
    piv = acc.pivot_table(index="dataset", columns="family", values="value",
                          aggfunc="mean")
    cnt = acc.pivot_table(index="dataset", columns="family", values="value",
                          aggfunc="size")
    piv = piv.loc[piv.notna().sum(axis=1).sort_values(ascending=False).index]
    cnt = cnt.reindex(index=piv.index, columns=piv.columns)   # keep them aligned

    fig, ax = plt.subplots(figsize=figsize)
    data = np.ma.masked_invalid(piv.values)
    im = ax.imshow(data, cmap=SEQUENTIAL, aspect="auto", vmin=0.6, vmax=1.0)
    im.cmap.set_bad("#f4f3ef")

    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            n = int(cnt.values[i, j])
            ax.text(j, i, f"{v:.3f}\nn={n}", ha="center", va="center",
                    fontsize=8, color="#ffffff" if v > 0.87 else INK)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("mean reported accuracy", fontsize=9, color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Reported accuracy by dataset and GNN family\n"
                 "blank = no method in Tables 1-2 reports that pairing")
    _source(ax, "Source: Tables 1-2, ACC rows. Family is the first "
                "architecture listed. Means over as few as one paper; cells "
                "are not comparable across datasets.")
    fig.tight_layout()
    return fig


def plot_claims(claims_df, figsize=(9.5, 3.2)):
    """Verdicts on the survey's own prose claims, checked against its tables."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    ok, bad = "#1baf7a", "#e34948"
    y = 1.0
    for _, r in claims_df.iterrows():
        colour = ok if r["supported"] else bad
        mark = "supported" if r["supported"] else "CONTRADICTED"
        ax.add_patch(plt.Rectangle((0, y - 0.055), 0.008, 0.11, color=colour,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(0.022, y + 0.026, f"{r['claim']}  (Sect. {r['section']})",
                fontsize=9.5, weight="semibold", color=INK, va="center",
                transform=ax.transAxes)
        ax.text(0.022, y - 0.024, r["evidence"], fontsize=8.3, color=INK_2,
                va="center", transform=ax.transAxes)
        ax.text(1.0, y + 0.026, mark, fontsize=8.5, weight="semibold",
                color=colour, ha="right", va="center", transform=ax.transAxes)
        y -= 0.17

    ax.set_title("The survey's prose claims, evaluated against its own Tables 1-2",
                 pad=6)
    fig.tight_layout()
    return fig


# ==========================================================================
# Part 3 -- results of the live experiments
# ==========================================================================

def _grouped_bars(ax, df, index, columns, value, err=None, palette=None):
    palette = palette or SERIES
    idx = list(dict.fromkeys(df[index]))
    cols = list(dict.fromkeys(df[columns]))
    width = 0.8 / len(cols)
    x = np.arange(len(idx))
    for j, c in enumerate(cols):
        sub = df[df[columns] == c].set_index(index)
        vals = [sub[value].get(i, np.nan) for i in idx]
        errs = ([sub[err].get(i, np.nan) for i in idx] if err else None)
        pos = x + (j - (len(cols) - 1) / 2) * width
        bars = ax.bar(pos, vals, width=width * 0.9,
                      color=palette[j % len(palette)], label=str(c),
                      yerr=errs, capsize=2.5,
                      error_kw=dict(elinewidth=1.1, ecolor=INK_2))
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, 0.012, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.4, rotation=90,
                        color="#ffffff")
    ax.set_xticks(x, idx)
    return cols


def plot_gnn_comparison(df, metric="accuracy", figsize=(9.5, 4.2)):
    """Our five architectures across datasets, mean +/- std over seeds."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    _grouped_bars(ax, df, "dataset", "gnn", metric, err=f"{metric}_std")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    ax.legend(title="architecture", ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Section 3.1's five layers, run on the same graphs\n"
                 f"{metric.replace('_', ' ')}, mean +/- s.d. over seeds")
    fig.tight_layout()
    return fig


def plot_graph_comparison(df, metric="accuracy", figsize=(8.5, 4.2)):
    """Similarity vs propagation vs heterogeneous -- the survey's own axis."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    _grouped_bars(ax, df, "dataset", "graph", metric, err=f"{metric}_std")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    ax.legend(title="graph construction", ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Graph construction dominates architecture\n"
                 "the survey's central claim, tested directly")
    fig.tight_layout()
    return fig


def plot_vs_literature(ours_df, long_df, figsize=(9, 4.6)):
    """Our accuracy against the range Tables 1-2 report for the same dataset.

    The literature is drawn as a range, not a point, because that is what it
    is: a set of numbers from different splits and protocols. Landing inside the
    range is the strongest claim this replication can honestly make -- matching
    any single row would be a coincidence, since no row states its split.
    """
    setup_style()
    acc = long_df[long_df["metric"] == "ACC"]
    datasets = [d for d in ours_df["dataset"].unique()
                if (acc["dataset"] == d).any()]

    fig, ax = plt.subplots(figsize=figsize)
    for i, d in enumerate(datasets):
        lit = acc.loc[acc["dataset"] == d, "value"].values
        ax.plot([lit.min(), lit.max()], [i, i], color=GRID, lw=9,
                solid_capstyle="round", zorder=1,
                label="reported in Tables 1-2" if i == 0 else None)
        ax.scatter(lit, np.full(len(lit), i), s=34, color=INK_MUTED, zorder=2,
                   alpha=0.7, edgecolor=SURFACE, linewidth=0.8,
                   label="individual published result" if i == 0 else None)

        sub = ours_df[ours_df["dataset"] == d]
        best = sub.loc[sub["accuracy"].idxmax()]
        ax.errorbar(best["accuracy"], i, xerr=best.get("accuracy_std", 0),
                    fmt="D", ms=10, color=SERIES[1], zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.4,
                    ecolor=SERIES[1], elinewidth=1.6, capsize=3,
                    label="this replication (best config)" if i == 0 else None)
        ax.text(best["accuracy"], i + 0.28,
                f"{best['accuracy']:.3f}  ({best['gnn']}, {best['graph']})",
                ha="center", fontsize=8, color=SERIES[1])

    ax.set_yticks(range(len(datasets)), datasets)
    ax.set_xlabel("accuracy")
    # Bound the axis by both series: our LIAR result sits below every published
    # number, and a fixed lower limit would silently clip the marker off-plot.
    lo = min(acc[acc["dataset"].isin(datasets)]["value"].min(),
             ours_df[ours_df["dataset"].isin(datasets)]["accuracy"].min())
    ax.set_xlim(max(0.0, lo - 0.08), 1.02)
    ax.set_ylim(-0.6, len(datasets) - 0.35)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.set_title("This replication against the published range\n"
                 "the literature is a range because no paper states its split")
    fig.tight_layout()
    return fig


def plot_confusion(cm, label_names, title="Confusion matrix", figsize=(4.8, 4.2)):
    """Row-normalised confusion, for reading which classes actually confuse."""
    setup_style()
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(label_names)), label_names, rotation=35, ha="right")
    ax.set_yticks(range(len(label_names)), label_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8.5,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_ablation(df, figsize=(8, 4.2)):
    """Accuracy as feature blocks of Fig. 4 are added or removed."""
    setup_style()
    df = df.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df["variant"], df["accuracy"], color=MAGNITUDE, height=0.62,
                   xerr=df.get("accuracy_std"), capsize=3,
                   error_kw=dict(elinewidth=1.1, ecolor=INK_2))
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, min(1.0, df["accuracy"].max() * 1.25))
    ax.grid(axis="y", visible=False)
    _bar_labels(ax, bars, fmt="{:.3f}", horizontal=True)
    ax.set_title("Which of Fig. 4's feature types carry the signal")
    fig.tight_layout()
    return fig
