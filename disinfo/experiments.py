"""Experiment suites and the caches that make them affordable.

Five suites, each aimed at a specific claim of the survey:

``suite_gnn_comparison``      Sect. 3.1's five architectures on one graph. The
                             survey says GCN and GAT dominate the literature; it
                             never says they are better. This checks.
``suite_graph_comparison``    Similarity vs propagation vs heterogeneous. This
                             is the survey's own novel axis (Sect. 5.3) and its
                             claim that graph construction is what matters.
``suite_feature_ablation``    Which of Fig. 4's feature blocks carry signal, and
                             whether Sect. 5.3.2's "majority rely on textual
                             features" is a good idea or just a habit.
``suite_liar_granularity``    6-class vs binary LIAR, to test the Table 2 vs
                             Sect. 7 contradiction (ACC 0.868 vs "below 50%").
``suite_pheme_split``         Random vs leave-events-out on PHEME, to test how
                             much of a reported accuracy is event memorisation.

``FeatureCache`` keys stage 1 on the things that actually change it -- the
dataset, the feature-related config fields and the split seed. Graph type and
architecture do not change stage 1, so a five-architecture sweep extracts
features once instead of five times. On PHEME that is the difference between
minutes and tens of minutes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, RESULTS
from .data import DisinfoDataset, load_dataset
from .features import build_features
from .pipeline import make_splits

__all__ = ["FeatureCache", "suite_gnn_comparison", "suite_graph_comparison",
           "suite_feature_ablation", "suite_liar_granularity",
           "suite_pheme_split", "results_to_frame", "save_results",
           "load_results"]


# Fields of Config that stage 1 depends on. Anything not listed here can change
# without invalidating a cached FeatureBundle.
_FEATURE_KEYS = ("text_features", "max_features", "ngram_max", "min_df",
                 "svd_dim", "use_profile", "use_credit_history", "seed",
                 "test_frac", "val_frac", "split")


class FeatureCache:
    """Memoises ``build_features`` across configurations that do not change it."""

    def __init__(self):
        self._store: dict = {}
        self.hits = 0
        self.misses = 0

    def get(self, ds: DisinfoDataset, cfg: Config):
        key = (ds.name, len(ds)) + tuple(getattr(cfg, k) for k in _FEATURE_KEYS)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        train_idx = make_splits(ds, cfg)[0]
        bundle = build_features(ds, cfg, train_idx)
        self._store[key] = bundle
        return bundle

    def stats(self) -> str:
        return f"FeatureCache: {self.hits} hits, {self.misses} misses"


def _run(ds, cfg, cache, n_seeds, quiet):
    """One ``run_seeds`` with the cache supplying stage 1 per seed."""
    runs = []
    for s in range(n_seeds):
        c = cfg.with_(seed=s)
        bundle = cache.get(ds, c) if cache else None
        from .pipeline import run_experiment
        runs.append(run_experiment(ds, c, quiet=quiet, features=bundle))

    keys = [k for k, v in runs[0]["metrics"].items() if isinstance(v, float)]
    summary = {}
    for k in keys:
        vals = np.array([r["metrics"][k] for r in runs], dtype=float)
        summary[k] = {"mean": float(np.nanmean(vals)),
                      "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                      "values": vals.tolist()}
    return {"dataset": ds.name, "config": asdict(cfg), "n_seeds": n_seeds,
            "summary": summary,
            "confusion": runs[0]["metrics"]["confusion"],
            "label_names": ds.label_names,
            "graph": runs[0]["graph"],
            "feature_blocks": runs[0]["feature_blocks"]}


def _log(quiet, ds, cfg, res, t0, extra=""):
    if quiet:
        return
    a = res["summary"]["accuracy"]
    f = res["summary"]["macro_f1"]
    print(f"  {ds.name:10s} {cfg.graph:14s} {cfg.gnn:6s} "
          f"acc {a['mean']:.3f}+/-{a['std']:.3f}  "
          f"f1 {f['mean']:.3f}  [{time.time() - t0:.0f}s] {extra}")


# --------------------------------------------------------------------------
# suites
# --------------------------------------------------------------------------

def suite_gnn_comparison(datasets, base: Config, gnns=None, n_seeds=3,
                         quiet=True, cache=None) -> list[dict]:
    """Every architecture of Sect. 3.1 on each dataset's natural graph.

    "Natural" means the propagation graph where cascades exist and the
    similarity graph where they do not -- LIAR has no propagation structure, so
    the survey's own taxonomy forces a similarity graph there.
    """
    cache = cache if cache is not None else FeatureCache()
    gnns = gnns or ["gcn", "gat", "gatv2", "sage", "gin"]
    out = []
    for name in datasets:
        ds = load_dataset(name)
        graph = "propagation" if ds.has_cascades else "similarity"
        if not quiet:
            print(f"[gnn] {ds.name} ({graph} graph, {len(ds)} items)")
        for gnn in gnns:
            cfg = base.with_(graph=graph, gnn=gnn)
            t0 = time.time()
            res = _run(ds, cfg, cache, n_seeds, quiet)
            _log(quiet, ds, cfg, res, t0)
            out.append(res)
    return out


def suite_graph_comparison(datasets, base: Config, n_seeds=3, quiet=True,
                           cache=None) -> list[dict]:
    """The survey's central axis: how the graph is built, holding the GNN fixed."""
    cache = cache if cache is not None else FeatureCache()
    out = []
    for name in datasets:
        ds = load_dataset(name)
        kinds = ["similarity"]
        if ds.has_cascades:
            kinds += ["propagation", "heterogeneous"]
        if name == "liar":
            kinds += ["attribute"]          # Hu et al. (2019)'s M-GCN graph
        if not quiet:
            print(f"[graph] {ds.name} ({len(ds)} items)")
        for graph in kinds:
            cfg = base.with_(graph=graph)
            t0 = time.time()
            res = _run(ds, cfg, cache, n_seeds, quiet)
            _log(quiet, ds, cfg, res, t0)
            out.append(res)
    return out


def suite_feature_ablation(name: str, base: Config, n_seeds=3, quiet=True,
                           cache=None) -> list[dict]:
    """Remove one block of Fig. 4's feature tree at a time.

    Implemented by zeroing the block rather than dropping the columns, so input
    dimension and therefore parameter count stay fixed: a drop in accuracy is
    then attributable to the information removed, not to a smaller model.
    """
    cache = cache if cache is not None else FeatureCache()
    ds = load_dataset(name)
    graph = "propagation" if ds.has_cascades else "similarity"
    cfg = base.with_(graph=graph)

    from .pipeline import run_experiment
    probe = cache.get(ds, cfg.with_(seed=0))
    blocks = list(probe.blocks)

    variants = [("all features", None)] + [(f"without {b}", b) for b in blocks]
    if len(blocks) > 1:
        variants.append(("lexical only", "__only_lexical__"))

    out = []
    for label, drop in variants:
        accs, f1s = [], []
        for s in range(n_seeds):
            c = cfg.with_(seed=s)
            bundle = cache.get(ds, c)
            X = bundle.X.copy()
            if drop == "__only_lexical__":
                for b, sl in bundle.blocks.items():
                    if b != "lexical":
                        X[:, sl] = 0.0
            elif drop is not None:
                X[:, bundle.blocks[drop]] = 0.0

            from .features import FeatureBundle
            masked = FeatureBundle(X, bundle.names, bundle.blocks)
            r = run_experiment(ds, c, quiet=True, features=masked)
            accs.append(r["metrics"]["accuracy"])
            f1s.append(r["metrics"]["macro_f1"])

        res = {"dataset": ds.name, "config": asdict(cfg), "variant": label,
               "n_seeds": n_seeds,
               "summary": {
                   "accuracy": {"mean": float(np.mean(accs)),
                                "std": float(np.std(accs, ddof=1)) if n_seeds > 1 else 0.0,
                                "values": accs},
                   "macro_f1": {"mean": float(np.mean(f1s)),
                                "std": float(np.std(f1s, ddof=1)) if n_seeds > 1 else 0.0,
                                "values": f1s}}}
        if not quiet:
            print(f"  {label:24s} acc {np.mean(accs):.3f}")
        out.append(res)
    return out


def suite_liar_granularity(base: Config, n_seeds=3, quiet=True) -> list[dict]:
    """6-class vs binary LIAR, with and without the leaky credit-history block.

    Table 1 reports LIAR ACC 0.492 (Hu et al., 6-class); Table 2 reports 0.868
    (Cui et al.); Sect. 7 says multiclass accuracy is "typically below 50%".
    The three cannot all describe the same task. This measures what each task
    actually yields.
    """
    out = []
    liar = load_dataset("liar")
    settings = [
        ("6-class", liar, False),
        ("6-class + credit history", liar, True),
        ("binary", liar.binarised(), False),
        ("binary + credit history", liar.binarised(), True),
    ]
    for label, ds, credit in settings:
        cfg = base.with_(graph="similarity", use_credit_history=credit)
        cache = FeatureCache()
        t0 = time.time()
        res = _run(ds, cfg, cache, n_seeds, quiet)
        res["variant"] = label
        if not quiet:
            a = res["summary"]["accuracy"]
            print(f"  LIAR {label:26s} acc {a['mean']:.3f}+/-{a['std']:.3f} "
                  f"[{time.time() - t0:.0f}s]")
        out.append(res)
    return out


def suite_pheme_split(base: Config, n_seeds=3, quiet=True) -> list[dict]:
    """Random split vs leave-events-out on PHEME.

    Table 1-2 report PHEME accuracies from 0.694 to 0.887 without stating a
    protocol. If the gap between the two protocols here is of that order, the
    published spread is at least as much about evaluation as about method.
    """
    ds = load_dataset("pheme")
    out = []
    for split in ("stratified", "event"):
        cfg = base.with_(graph="propagation", split=split)
        cache = FeatureCache()
        t0 = time.time()
        res = _run(ds, cfg, cache, n_seeds, quiet)
        res["variant"] = f"{split} split"
        if not quiet:
            a = res["summary"]["accuracy"]
            print(f"  PHEME {split:12s} acc {a['mean']:.3f}+/-{a['std']:.3f} "
                  f"[{time.time() - t0:.0f}s]")
        out.append(res)
    return out


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------

def results_to_frame(results: list[dict]) -> pd.DataFrame:
    """Flatten result dicts into the tidy frame the plots consume."""
    rows = []
    for r in results:
        cfg = r["config"]
        row = {"dataset": r["dataset"], "gnn": cfg["gnn"],
               "graph": cfg["graph"], "split": cfg["split"],
               "n_seeds": r["n_seeds"], "variant": r.get("variant", "")}
        for metric, stats in r["summary"].items():
            row[metric] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(results: list[dict], name: str, outdir: Path | None = None) -> Path:
    outdir = Path(outdir or RESULTS)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.json"
    path.write_text(json.dumps(results, indent=1))
    return path


def load_results(name: str, outdir: Path | None = None) -> list[dict]:
    path = Path(outdir or RESULTS) / f"{name}.json"
    return json.loads(path.read_text())
