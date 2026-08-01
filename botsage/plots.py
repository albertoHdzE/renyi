"""Figures: the paper's own three diagrams, and the ones its results require.

The paper has three figures, all schematic (a GraphSAGE cartoon, the methodology
workflow, a two-node graph-construction example) and no data plots at all. Its
quantitative content is Tables 1-5. So this module does two jobs: redraw the
three diagrams, and draw the comparisons that Tables 4-5 invite but omit --
above all, the majority-class baseline, which neither table states.

Colours follow the same validated categorical palette used by ``disinfo.plots``
(slots 1-5, adjacent pairlist, light mode), so the two replications in this
repository read as one system. Three of those slots fall below 3:1 contrast
against the surface, so every categorical mark also carries a direct value
label.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import FIGURES, PAPER_TABLE1, PAPER_TABLE2, PAPER_TABLE3

__all__ = ["setup_style", "save", "fig1_graphsage", "fig2_workflow",
           "fig3_graph_construction", "plot_table4", "plot_table5",
           "plot_singular_spectrum", "plot_ablation", "plot_seed_sensitivity",
           "plot_trained_vs_untrained", "plot_graph_scope", "plot_timings",
           "plot_twibot22_splits", "plot_neighbour_degeneracy"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e3e2dd"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MAGNITUDE = "#2a78d6"
ACCENT = "#eb6834"
GOOD, BAD = "#1baf7a", "#e34948"
SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list(
    "botsage_seq", ["#eef4fc", "#c5daf5", "#8fb8ea", "#5495dc", "#2a78d6",
                    "#1b5296"])


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "lines.linewidth": 2.0, "lines.markersize": 8,
    })


def save(fig, name: str, outdir: Path | None = None) -> Path:
    outdir = Path(outdir or FIGURES)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path)
    return path


def _source(ax, text):
    ax.annotate(text, xy=(0, -0.18), xycoords="axes fraction", fontsize=8,
                color=INK_MUTED, va="top")


# ==========================================================================
# the paper's own figures
# ==========================================================================

def fig1_graphsage(figsize=(11, 3.0)):
    """Fig. 1: sample neighbours -> aggregate -> predict.

    Sect. 2.1 notes the third stage is *not* used here: "In this project, we do
    not use the prediction head but rather use it to generate embeddings." The
    dropped stage is drawn greyed out, because its absence is the single most
    consequential design choice in the paper.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)

    stages = [
        ("1. Sample\nneighbourhood", INK, "#ffffff"),
        ("2. Aggregate\nfeature information", INK, "#ffffff"),
        ("3. Predict graph context\nand label", INK_MUTED, "#f2f1ed"),
    ]
    x0, w, gap = 6.0, 25.0, 8.0
    for i, (label, ec, fc) in enumerate(stages):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 9), w, 12,
                                    boxstyle="round,pad=0.4,rounding_size=0.7",
                                    fc=fc, ec=ec, lw=1.6,
                                    ls="--" if i == 2 else "-"))
        ax.text(x + w / 2, 15, label, ha="center", va="center", fontsize=10.5,
                color=ec, weight="semibold" if i < 2 else "normal")
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.6, 15), (x - 0.8, 15),
                                         arrowstyle="-|>", mutation_scale=15,
                                         color=INK_2 if i < 2 else INK_MUTED,
                                         lw=1.5))
    ax.text(x0 + 2 * (w + gap) + w / 2, 5.4,
            "not used in this project (Sect. 2.1)\n"
            "-> the layer is never trained",
            ha="center", va="top", fontsize=9, color=ACCENT)
    ax.set_title("Fig. 1  GraphSage aggregation approach", pad=6)
    fig.tight_layout()
    return fig


def fig2_workflow(figsize=(12.5, 4.4)):
    """Fig. 2: the methodology workflow of Chapter 3."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)

    def box(x, y, w, h, text, sub=None, fc="#ffffff", ec=INK, lw=1.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.35,rounding_size=0.6",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2 + (1.4 if sub else 0), text,
                ha="center", va="center", fontsize=10, weight="semibold",
                color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center",
                    fontsize=8, color=INK_MUTED)

    def arrow(x1, y1, x2, y2, color=INK_2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, color=color, lw=1.4,
                                     connectionstyle="arc3,rad=0"))

    box(2, 18, 15, 10, "Dataset\npreprocessing", "Sect. 3.1")
    box(24, 30, 20, 10, "DistilBERT / BERT", "Sect. 3.2  ->  768", fc="#eef4fc")
    box(24, 6, 20, 10, "GraphSage", "Sect. 3.5  ->  128", fc="#eef4fc")
    box(51, 18, 16, 10, "Concatenate", "Sect. 3.6  ->  896")
    box(72, 18, 12, 10, "SVM", "Sect. 3.7")
    box(88, 18, 10, 10, "Evaluate", "Sect. 3.8")

    arrow(17.5, 25, 23.5, 33)
    arrow(17.5, 21, 23.5, 13)
    arrow(44.5, 33, 50.5, 25)
    arrow(44.5, 13, 50.5, 21)
    arrow(67.5, 23, 71.5, 23)
    arrow(84.5, 23, 87.5, 23)

    ax.text(34, 42.5, "tweet text", ha="center", fontsize=9, color=INK_MUTED)
    ax.text(34, 2.5, "user features + edge index", ha="center", fontsize=9,
            color=INK_MUTED)
    ax.set_title("Fig. 2  Methodology workflow", pad=6)
    fig.tight_layout()
    return fig


def fig3_graph_construction(figsize=(6.4, 3.4)):
    """Fig. 3: the paper's worked example, edges [1,0] and [0,2]."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    pos = {0: (0.5, 0.28), 1: (0.16, 0.78), 2: (0.84, 0.78)}
    for a, b in [(1, 0), (0, 2)]:
        ax.plot(*zip(pos[a], pos[b]), color=INK, lw=1.8, zorder=1)
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=1500, c="#eef4fc", edgecolor=INK, linewidth=1.6,
                   zorder=2)
        ax.text(x, y, str(n), ha="center", va="center", fontsize=13, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.98)
    ax.set_title("Fig. 3  Graph construction example\n"
                 "edge list [1,0] and [0,2]", pad=6)
    fig.tight_layout()
    return fig


# ==========================================================================
# results
# ==========================================================================

def _comparison(table, ours, baseline, title, source, figsize, baseline_label):
    """Shared renderer for Tables 4 and 5, with the majority baseline drawn."""
    setup_style()
    rows = [(m, t, a, f, mine) for m, t, a, f, mine in table]
    if ours is not None:
        rows = rows + [(f"{r['variant']} (ours)", "G", r["accuracy"] * 100,
                        r["f1"] * 100, "ours") for _, r in ours.iterrows()]
    rows.sort(key=lambda r: r[2])

    labels = [r[0] for r in rows]
    acc = np.array([r[2] for r in rows])
    colours = [ACCENT if r[4] == "ours" else
               (SERIES[0] if r[4] else "#c9c8c2") for r in rows]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(rows))
    bars = ax.barh(y, acc, color=colours, height=0.66)
    ax.set_yticks(y, labels)
    ax.set_xlabel("accuracy (%)")
    # Room for a value column to the right of every bar, so the labels never
    # collide with the baseline rule.
    right = max(105, acc.max() * 1.06)
    ax.set_xlim(0, right + 26)
    ax.set_xticks([t for t in (0, 25, 50, 75, 100) if t <= right])
    ax.grid(axis="y", visible=False)

    for b, r in zip(bars, rows):
        f1 = f"F1 {r[3]:.2f}" if r[3] is not None else "F1 n/r"
        ax.text(right + 1.5, b.get_y() + b.get_height() / 2,
                f"{r[2]:6.2f}   {f1}", va="center", fontsize=8.6,
                color=INK_2, family="monospace")

    ax.axvline(baseline * 100, color=BAD, lw=2.0, ls="--", zorder=5)
    ax.text(baseline * 100, len(rows) - 0.25, f"  {baseline_label}",
            color=BAD, fontsize=9, va="top", weight="semibold")

    ax.set_title(title)
    _source(ax, source)
    fig.tight_layout()
    return fig


def plot_table4(ours=None, baseline=0.6321, figsize=(9, 4.6)):
    """Cresci-15. Blue = the paper's own rows, grey = baselines it cites."""
    from .config import PAPER_TABLE4
    return _comparison(
        PAPER_TABLE4, ours, baseline,
        "Cresci-15: reported accuracy against the majority baseline",
        "Source: Deshmukh (2025) Table 4. Baseline = always predict 'bot' "
        "(3,351 of 5,301 labelled users), computed from the released labels.",
        figsize, "majority baseline 63.21%")


def plot_table5(ours=None, baseline=0.860057, figsize=(9, 4.8)):
    """TwiBot-22 -- the headline finding: every row is below the baseline."""
    from .config import PAPER_TABLE5
    return _comparison(
        PAPER_TABLE5, ours, baseline,
        "TwiBot-22: every reported accuracy is below the majority baseline",
        "Source: Deshmukh (2025) Table 5. Baseline = always predict 'human' "
        "(860,057 of 1,000,000), from the released label.csv. The paper "
        "evaluates by 5-fold CV over the corpus, so this is the figure its "
        "accuracies must clear.",
        figsize, "majority baseline 86.01%")


def plot_twibot22_splits(report: dict, figsize=(8, 3.8)):
    """The official split is not stratified -- so 'the' baseline is ambiguous."""
    setup_style()
    keys = [k for k in ("corpus", "train", "val", "test") if k in report]
    bot = np.array([report[k]["bot_frac"] for k in keys]) * 100
    base = np.array([report[k]["majority_baseline"] for k in keys]) * 100

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(keys))
    b1 = ax.bar(x - 0.2, bot, width=0.38, color=SERIES[0], label="% bot")
    b2 = ax.bar(x + 0.2, base, width=0.38, color=SERIES[3],
                label="majority-class accuracy")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{b.get_height():.1f}", ha="center", fontsize=8.5,
                    color=INK_2)
    ax.set_xticks(x, keys)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("TwiBot-22's official split is not stratified\n"
                 "the baseline a result must beat depends on the protocol")
    _source(ax, "Computed from the released label.csv and split.csv "
                "(Zenodo 7012904). The paper uses 5-fold CV over the corpus "
                "(86.0%); the leaderboard it cites uses the test split (70.6%).")
    fig.tight_layout()
    return fig


def plot_singular_spectrum(emb: np.ndarray, n_features: int = 5,
                           figsize=(7.5, 4.0)):
    """The rank of the 128-dimensional GraphSAGE embedding.

    Log scale, because the story is a cliff: the first ``2*n_features``
    singular values are O(1) and the rest sit at the float32 rounding floor.
    """
    setup_style()
    centred = np.asarray(emb, dtype=np.float64)
    centred = centred - centred.mean(0, keepdims=True)
    s = np.linalg.svd(centred, compute_uv=False)
    s = np.maximum(s, 1e-20) / s[0]
    k = 2 * n_features

    fig, ax = plt.subplots(figsize=figsize)
    idx = np.arange(1, len(s) + 1)
    ax.scatter(idx[:k], s[:k], s=46, color=SERIES[0], zorder=3,
               label=f"first {k} (= 2 x {n_features} features)")
    ax.scatter(idx[k:], s[k:], s=22, color="#c9c8c2", zorder=2,
               label="the other 118 -- numerical noise")
    ax.axvline(k + 0.5, color=BAD, ls="--", lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("singular value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.legend(loc="upper right")
    gap = s[k - 1] / max(s[k], 1e-20)
    ax.set_title(f"The 128-dim 'embedding' has rank {k}\n"
                 f"$\\sigma_{{{k}}}/\\sigma_{{{k + 1}}}$ = {gap:.1e}")
    _source(ax, "An untrained SAGEConv is an affine map of "
                "[x_v || mean of N(v)], so its image is a "
                f"{k}-dimensional subspace regardless of out_channels.")
    fig.tight_layout()
    return fig


def plot_neighbour_degeneracy(stats: dict, figsize=(7.5, 3.6)):
    """Why the graph branch contributes nothing on Cresci-15."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(stats)
    vals = [stats[k] for k in labels]
    bars = ax.barh(labels[::-1], vals[::-1], color=MAGNITUDE, height=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() * 1.02, b.get_y() + b.get_height() / 2,
                f"{v:,.4g}" if v < 1 else f"{v:,.0f}",
                va="center", fontsize=9, color=INK_2)
    ax.set_xscale("symlog")
    ax.grid(axis="y", visible=False)
    ax.set_title("The graph branch on Cresci-15")
    fig.tight_layout()
    return fig


def _labelled_barh(df, value, err, title, source, figsize, highlight=None,
                   baseline=None, xlabel="accuracy"):
    setup_style()
    df = df.sort_values(value)
    colours = [ACCENT if (highlight and h in str(v)) else MAGNITUDE
               for v in df["variant"] for h in ([highlight] if highlight else [""])]
    if not highlight:
        colours = MAGNITUDE

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df["variant"], df[value], color=colours, height=0.62,
                   xerr=df[err] if err in df else None, capsize=3,
                   error_kw=dict(elinewidth=1.1, ecolor=INK_2))
    for b, v in zip(bars, df[value]):
        ax.text(b.get_width() + 0.008, b.get_y() + b.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8.8, color=INK_2)
    if baseline is not None:
        ax.axvline(baseline, color=BAD, lw=1.8, ls="--")
        ax.text(baseline, -0.75, f" baseline {baseline:.3f}", color=BAD,
                fontsize=8.5, va="bottom")
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, min(1.06, max(df[value].max() * 1.15, 0.1)))
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    _source(ax, source)
    fig.tight_layout()
    return fig


def plot_ablation(df, baseline=None, figsize=(9, 4.4)):
    return _labelled_barh(
        df, "accuracy", "accuracy_std",
        "What the 896 dimensions actually contribute\n"
        "if GraphSage[128] does not beat 'effective 10 dims', it is a "
        "reparameterisation",
        "5-fold CV, linear SVM, identical folds. Error bars are s.d. across "
        "folds.", figsize, highlight="GraphSage[128]", baseline=baseline)


def plot_seed_sensitivity(df, paper_value=None, figsize=(8, 4.0)):
    """Accuracy across random initialisations of the *untrained* layer.

    The result is the opposite of what one might fear, and it is informative:
    the spread is tiny. A random projection changes the *coordinates* of the
    10-dimensional subspace but not the subspace itself, and a linear SVM is
    free to undo any invertible change of basis. So the seed cannot move the
    achievable accuracy much -- which is one more way of showing the layer adds
    nothing. The fold-to-fold standard deviation (error bars) dwarfs it.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    seeds = df["sage_seed"].to_numpy()
    acc = df["accuracy"].to_numpy()
    sd = df["accuracy_std"].to_numpy() if "accuracy_std" in df else None
    ax.errorbar(seeds, acc, yerr=sd, fmt="o-", color=SERIES[0],
                ecolor=INK_MUTED, elinewidth=1.1, capsize=3,
                label="accuracy (bars: s.d. across CV folds)")
    ax.axhline(acc.mean(), color=INK_MUTED, ls=":", lw=1.4)
    if paper_value is not None:
        ax.axhline(paper_value, color=ACCENT, ls="--", lw=1.8)
        ax.text(seeds.max(), paper_value, f" paper {paper_value:.4f}",
                color=ACCENT, fontsize=9, va="bottom", ha="right")
    ax.set_xlabel("random seed of the untrained SAGEConv")
    ax.set_ylabel("accuracy")
    ax.legend(loc="lower right")
    spread = acc.max() - acc.min()
    fold_sd = float(np.mean(sd)) if sd is not None else float("nan")
    ax.set_title(f"The initialisation barely matters\n"
                 f"seed-to-seed spread {spread:.4f} vs fold-to-fold s.d. "
                 f"{fold_sd:.4f}")
    _source(ax, "A random projection changes the coordinates of the "
                "10-dimensional subspace, not the subspace -- and a linear SVM "
                "can undo any invertible change of basis. Stability here is "
                "further evidence the layer adds no information.")
    fig.tight_layout()
    return fig


def plot_trained_vs_untrained(df, baseline=None, figsize=(8, 3.2)):
    return _labelled_barh(
        df, "accuracy", "accuracy_std",
        "What the missing prediction head costs",
        "Same graph, same features, same 5-fold protocol.", figsize,
        highlight="trained GraphSAGE", baseline=baseline)


def plot_graph_scope(df, baseline=None, figsize=(8, 3.2)):
    return _labelled_barh(
        df, "accuracy", "accuracy_std",
        "Both readings of Sect. 3.1.1's edge definition",
        "'all' keeps edges to users with no metadata; 'labelled' keeps only "
        "edges between users that survived cleaning.", figsize,
        baseline=baseline)


def plot_timings(figsize=(9, 4.0)):
    """Tables 1-3: the compute budget that shaped the paper's design."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    models = list(PAPER_TABLE1)
    x = np.arange(len(models))
    b1 = ax1.bar(x - 0.2, [PAPER_TABLE1[m] for m in models], width=0.38,
                 color=SERIES[0], label="all tweets")
    b2 = ax1.bar(x + 0.2, [PAPER_TABLE2[m] for m in models], width=0.38,
                 color=SERIES[1], label="15 tweets/user")
    for bars in (b1, b2):
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 6,
                     f"{b.get_height():.0f}h", ha="center", fontsize=8.5,
                     color=INK_2)
    ax1.set_xticks(x, models)
    ax1.set_ylabel("hours")
    ax1.set_ylim(0, 380)
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper right")
    ax1.set_title("Tables 1-2  preprocessing TwiBot-22")

    labels = [f"{m}\n{d}" for m, d, _ in PAPER_TABLE3]
    vals = [h for _, _, h in PAPER_TABLE3]
    bars = ax2.barh(labels[::-1], vals[::-1], color=MAGNITUDE, height=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax2.text(b.get_width() + 4, b.get_y() + b.get_height() / 2,
                 f"{v:.1f}h" + (" (>300)" if v >= 300 else ""),
                 va="center", fontsize=8.5, color=INK_2)
    ax2.set_xlabel("hours")
    ax2.set_xlim(0, 380)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Table 3  training time")

    fig.suptitle("The compute budget that shaped the method",
                 x=0.02, ha="left", fontsize=12, weight="semibold")
    _source(ax2, "Reported on a 2019 MacBook Pro, 2.4 GHz quad-core i5, 8 GB. "
                 "These numbers are why only 15 tweets per user are used.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
