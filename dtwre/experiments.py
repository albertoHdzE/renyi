"""Experiment drivers reproducing every table and data figure of the paper.

Node2Vec embeddings depend only on the message-passing graph, which is fixed
by ``(dataset, train_frac, mp_frac)``. The alpha, lambda, negative-ratio and
window sweeps all leave that graph untouched, so the embedding is computed
once and reused -- a large speedup that changes no result.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (Config, METRICS, BASELINES, BASELINE_LABELS, ALPHA_GRID,
                     LAMBDA_GRID, NEG_RATIO_GRID, WINDOW_GRID, PAPER_TABLE1,
                     PAPER_TABLE2, RESULTS)
from .embeddings import node2vec
from .pipeline import prepare_split, run_experiment

__all__ = [
    "EmbeddingCache", "baseline_comparison", "alpha_sweep", "lambda_sweep",
    "ratio_sweep", "window_sweep", "training_curves", "figure11_inputs",
    "fidelity_table", "results_frame", "save_results", "load_results",
]


class EmbeddingCache:
    """Memoises Node2Vec per message-passing graph."""

    def __init__(self):
        self._store = {}

    def get(self, tg, cfg: Config, split):
        key = (tg.name, cfg.train_frac, cfg.mp_frac, cfg.embedding_dim,
               cfg.seed, cfg.walk_length, cfg.num_walks, cfg.p, cfg.q)
        if key not in self._store:
            t0 = time.time()
            self._store[key] = node2vec(
                split["train_graph"], tg.num_nodes, dim=cfg.embedding_dim,
                num_walks=cfg.num_walks, walk_length=cfg.walk_length,
                window=cfg.context_window, p=cfg.p, q=cfg.q,
                epochs=cfg.n2v_epochs, lr=cfg.n2v_lr,
                negative=cfg.n2v_negative, seed=cfg.seed)
            self._store[key].flags.writeable = False
            self._elapsed = time.time() - t0
        return self._store[key]


def _metrics_only(res: dict, on="test") -> dict:
    return {m: res[on][m] for m in METRICS}


def baseline_comparison(tg, cfg: Config, cache: EmbeddingCache | None = None,
                        methods=None, verbose=True) -> dict:
    """Table 1 / Figure 5: every method on identical splits and features."""
    cache = cache or EmbeddingCache()
    methods = methods or BASELINES
    split = prepare_split(tg, cfg)
    emb = cache.get(tg, cfg, split)

    out = {}
    for m in methods:
        use_emb = emb if m in {"node2vec", "dtwre"} else None
        res = run_experiment(tg, cfg, method=m, split=split,
                             cached_embedding=use_emb)
        out[m] = _metrics_only(res)
        if verbose:
            print(f"  {BASELINE_LABELS.get(m, m):16s} "
                  + "  ".join(f"{k}={out[m][k]:.4f}" for k in METRICS))
    return out


def _sweep(tg, cfg, field, values, cache, rebuild_split, verbose=True,
           seeds=(42,)):
    """Sweep ``field`` over ``values``, averaging each point across ``seeds``.

    Single-seed sweeps on these splits are noise-dominated: the test set holds
    only a few hundred pairs, so threshold metrics can swing by tenths between
    seeds (recall occasionally collapses to 1.0 as the decoder degenerates to
    all-positive). Each returned row carries both the mean and a ``*_std`` so
    the figures can show error bands -- something the paper does not report.
    """
    cache = cache or EmbeddingCache()
    rows = []
    for v in values:
        c0 = replace(cfg, **{field: v})
        per_seed = []
        for s in seeds:
            c = replace(c0, seed=s)
            split = prepare_split(tg, c) if rebuild_split else \
                prepare_split(tg, replace(cfg, seed=s))
            emb = cache.get(tg, c, split)
            per_seed.append(_metrics_only(
                run_experiment(tg, c, method="dtwre", split=split,
                               cached_embedding=emb)))
        row = {m: float(np.mean([p[m] for p in per_seed])) for m in METRICS}
        row.update({f"{m}_std": float(np.std([p[m] for p in per_seed]))
                    for m in METRICS})
        rows.append(row)
        if verbose:
            print(f"  {field}={v:<10} " +
                  "  ".join(f"{k}={row[k]:.4f}" for k in METRICS))
    return rows


def alpha_sweep(tg, cfg, values=None, cache=None, verbose=True, seeds=(42,)):
    """Figure 6: alpha changes only the entropy features, not the split."""
    return _sweep(tg, cfg, "alpha", values or ALPHA_GRID, cache,
                  rebuild_split=False, verbose=verbose, seeds=seeds)


def lambda_sweep(tg, cfg, values=None, cache=None, verbose=True, seeds=(42,)):
    """Figure 7: lambda changes only the time weighting, not the split."""
    return _sweep(tg, cfg, "lam", values or LAMBDA_GRID, cache,
                  rebuild_split=False, verbose=verbose, seeds=seeds)


def ratio_sweep(tg, cfg, values=None, cache=None, verbose=True, seeds=(42,)):
    """Figure 8: the negative ratio changes the sampled pairs, so re-split."""
    return _sweep(tg, cfg, "neg_ratio", values or NEG_RATIO_GRID, cache,
                  rebuild_split=True, verbose=verbose, seeds=seeds)


def window_sweep(tg, cfg, values=None, cache=None, verbose=True, seeds=(42,)):
    """Table 2 / Figure 9: the window changes snapshot segmentation."""
    return _sweep(tg, cfg, "window_seconds", values or WINDOW_GRID, cache,
                  rebuild_split=True, verbose=verbose, seeds=seeds)


def training_curves(tg, cfg, cache=None, method="dtwre", split=None):
    """Figure 10: per-epoch metrics across the 100 training epochs."""
    cache = cache or EmbeddingCache()
    split = split or prepare_split(tg, cfg)
    emb = cache.get(tg, cfg, split)
    return run_experiment(tg, cfg, method=method, split=split,
                          cached_embedding=emb, track_history=True)


def figure11_inputs(res: dict, split: dict, top_k: int = 12,
                    threshold: float = 0.5):
    """Key nodes and predicted edges for Figure 11.

    Key nodes are the highest time-weighted Local Node Entropy, matching the
    caption: "red nodes ... exhibit high entropy values and are identified as
    key nodes".
    """
    extras = res.get("extras", {})
    score = extras.get("lne_tw")
    if score is None:
        score = extras.get("lne_last")
    if score is None:
        raise ValueError("run figure11_inputs on a 'dtwre' result")

    key_nodes = np.argsort(-np.asarray(score))[:top_k].tolist()
    pairs, scores = res["test_pairs"], res["test_scores"]
    pred_pos = pairs[scores >= threshold].tolist()
    pred_neg = pairs[scores < threshold].tolist()
    return key_nodes, pred_pos, pred_neg


def results_frame(results: dict, index_name="method") -> pd.DataFrame:
    df = pd.DataFrame(results).T[METRICS]
    df.index.name = index_name
    return df.round(4)


def fidelity_table(obtained: dict, published: dict,
                   label_map=None) -> pd.DataFrame:
    """Side-by-side obtained vs published values with signed differences."""
    label_map = label_map or BASELINE_LABELS
    rows = []
    for key, pub in published.items():
        obs = obtained.get(key)
        if obs is None:
            continue
        row = {"key": label_map.get(key, key)}
        for m in METRICS:
            row[f"{m}_paper"] = round(pub[m], 4)
            row[f"{m}_ours"] = round(obs[m], 4)
            row[f"{m}_diff"] = round(obs[m] - pub[m], 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("key")


def save_results(obj, name: str, outdir: Path | None = None) -> Path:
    outdir = Path(outdir or RESULTS)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default))
    return path


def load_results(name: str, outdir: Path | None = None):
    path = Path(outdir or RESULTS) / name
    return json.loads(path.read_text()) if path.exists() else None
