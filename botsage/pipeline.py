"""The Figure 2 workflow, end to end.

    preprocess -> [ DistilBERT on tweets ]  ->  concatenate -> SVM -> evaluate
                  [ GraphSAGE on the graph ]

``build_embeddings`` produces the 896-dimensional vector of Sect. 3.6;
``evaluate`` runs the linear SVM under 5-fold cross-validation (Sects. 3.7-3.8);
``run_experiment`` does both and reports accuracy and F1 next to the majority
baseline the paper never states.

Two protocol facts that shape everything below:

* The paper evaluates by **5-fold cross-validation over the whole dataset**, not
  on a held-out split. ``protocol="cv"`` reproduces that. ``protocol=
  "official_split"`` uses the corpus's own train/test split instead, which is
  what the baselines in Tables 4-5 were computed on.
* Because the GraphSAGE layer is untrained and the text encoder is frozen,
  **nothing is fitted before the SVM**, so building embeddings over all users at
  once leaks nothing. The one thing that *would* leak is feature scaling, so the
  scaler is fitted inside each CV fold.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import Config
from .data import BotDataset
from .sage import sage_embeddings

__all__ = ["restrict_graph", "build_embeddings", "evaluate", "run_experiment",
           "make_classifier", "expand_text"]


def expand_text(ds: BotDataset, te: np.ndarray) -> np.ndarray:
    """Accept text embeddings for all users *or* for the labelled ones only.

    Encoding only labelled users is the sensible default: Cresci-15 has 5,301
    labelled users among 1.29M edge-referenced nodes, so a full-length matrix
    would be 3.7 GB of mostly zeros. This scatters the compact form back to full
    length so the graph and text branches stay index-aligned.
    """
    te = np.asarray(te, dtype=np.float32)
    if te.shape[0] == len(ds):
        return te
    if te.shape[0] != len(ds.labelled):
        raise ValueError(f"text embeddings have {te.shape[0]} rows; "
                         f"expected {len(ds)} or {len(ds.labelled)}")
    full = np.zeros((len(ds), te.shape[1]), dtype=np.float32)
    full[ds.labelled] = te
    return full


def restrict_graph(ds: BotDataset, scope: str) -> np.ndarray:
    """Choose which edges the GraphSAGE layer sees.

    Sect. 3.1.1 is ambiguous, and the two readings differ enormously on
    Cresci-15:

    ``"all"``       Every relation in the corpus is an edge, including those to
                    users who have no metadata. This is the literal reading of
                    "If a relation exists, it is considered to be an edge".
                    On Cresci-15 that means 6,994,858 edges, of which only
                    8,550 (0.12%) join two users who actually have features --
                    so the neighbour mean is dominated by zero vectors.
    ``"labelled"``  Only edges between users that survived the cleaning of
                    Sect. 3.1.1 ("removing rows where values in critical fields
                    ... are missing"). Defensible, and arguably the intended
                    reading, but it leaves 3,381 of 5,301 Cresci-15 users
                    isolated.

    Neither reading gives the graph branch much to work with, which is the
    finding rather than a bug. See ``docs/DISCREPANCIES_BOTSAGE.md`` §3.
    """
    if scope == "all":
        return ds.edge_index
    if scope != "labelled":
        raise ValueError(f"unknown graph scope {scope!r}")
    has_feat = np.zeros(len(ds), dtype=bool)
    has_feat[np.flatnonzero(np.abs(ds.features).sum(1) > 0)] = True
    ei = ds.edge_index
    if ei.size == 0:
        return ei
    keep = has_feat[ei[0]] & has_feat[ei[1]]
    return ei[:, keep]


def build_embeddings(ds: BotDataset, cfg: Config,
                     text_embeddings: np.ndarray | None = None,
                     graph_scope: str = "all",
                     return_parts: bool = False):
    """Sect. 3.6: concatenate the GraphSAGE and BERT vectors into 896 dims.

    ``use_graph``/``use_text`` switch the branches off for the ablations; with
    both on and the paper's sizes this returns ``(n_users, 896)``.
    """
    parts, names = [], []

    if cfg.use_graph:
        x = torch.from_numpy(np.ascontiguousarray(ds.features)).float()
        if cfg.standardize_features:
            # Not stated by the paper. Without it the raw counts (up to 10^7)
            # dominate the concatenation and the BERT half is invisible to a
            # linear SVM. See DISCREPANCIES §7.
            mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
            x = (x - mu) / sd
        ei = torch.from_numpy(np.ascontiguousarray(restrict_graph(ds, graph_scope)))
        emb = sage_embeddings(x, ei, out_channels=cfg.sage_out,
                              seed=cfg.sage_seed, aggr=cfg.sage_aggr,
                              root_weight=cfg.sage_root_weight,
                              normalize=cfg.sage_normalize).numpy()
        parts.append(emb.astype(np.float32))
        names.append(f"graphsage[{emb.shape[1]}]")

    if cfg.use_text:
        te = text_embeddings if text_embeddings is not None else ds.text_embeddings
        if te is None:
            raise ValueError(f"{ds.name} has no text embeddings; "
                             f"run text.user_text_embeddings first "
                             f"or set use_text=False")
        te = expand_text(ds, te)
        parts.append(te)
        names.append(f"bert[{te.shape[1]}]")

    if not parts:
        raise ValueError("both branches disabled")

    X = np.concatenate(parts, axis=1)
    if return_parts:
        return X, names, parts
    return X, names


def make_classifier(cfg: Config):
    """Sect. 3.7: a linear-kernel SVM.

    ``LinearSVC`` rather than ``SVC(kernel="linear")``: identical model, but it
    solves the primal and scales to the sample sizes here, where ``SVC`` is
    quadratic in the number of samples. The scaler is part of the pipeline so
    that cross-validation refits it per fold rather than once over everything.
    """
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=cfg.svm_C, max_iter=cfg.svm_max_iter,
                  class_weight=cfg.class_weight, dual="auto",
                  random_state=cfg.seed),
    )


def _scores(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def evaluate(X: np.ndarray, y: np.ndarray, cfg: Config,
             split: dict | None = None, quiet: bool = True) -> dict:
    """5-fold cross-validation (the paper) or the corpus's official split.

    F1 is for the **bot** class, matching how Tables 4-5 report it: on Cresci-15
    bots are the majority (63.2%), which is why the paper's F1 sits above its
    accuracy there.
    """
    if cfg.protocol == "official_split" and split:
        tr = np.concatenate([split["train"], split.get("val", np.array([], int))])
        te = split["test"]
        clf = make_classifier(cfg).fit(X[tr], y[tr])
        m = _scores(y[te], clf.predict(X[te]))
        m["majority_baseline"] = float(np.bincount(y[te], minlength=2).max() / len(te))
        # A single fit, so there is no spread to report. Every value stays a
        # {mean, std, folds} dict so downstream code needs no special case.
        return {k: {"mean": v, "std": 0.0, "folds": [v]} for k, v in m.items()}

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                          random_state=cfg.seed)
    folds: list[dict] = []
    for k, (tr, te) in enumerate(skf.split(X, y)):
        clf = make_classifier(cfg).fit(X[tr], y[tr])
        folds.append(_scores(y[te], clf.predict(X[te])))
        if not quiet:
            print(f"    fold {k + 1}: acc {folds[-1]['accuracy']:.4f} "
                  f"f1 {folds[-1]['f1']:.4f}")

    out = {}
    for key in folds[0]:
        vals = np.array([f[key] for f in folds])
        out[key] = {"mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "folds": vals.tolist()}
    out["majority_baseline"] = {
        "mean": float(np.bincount(y, minlength=2).max() / len(y)),
        "std": 0.0, "folds": []}
    return out


def run_experiment(ds: BotDataset, cfg: Config,
                   text_embeddings: np.ndarray | None = None,
                   graph_scope: str = "all", quiet: bool = True) -> dict:
    """One full pass of Figure 2, restricted to the labelled users."""
    X, names = build_embeddings(ds, cfg, text_embeddings, graph_scope)
    lab = ds.labelled
    X, y = X[lab], ds.labels[lab]

    metrics = evaluate(X, y, cfg, split=_relabel_split(ds, lab), quiet=quiet)
    if not quiet:
        a, f = metrics["accuracy"], metrics["f1"]
        print(f"  {ds.name:10s} {'+'.join(names):24s} "
              f"acc {a['mean']:.4f}+/-{a['std']:.4f}  f1 {f['mean']:.4f}  "
              f"(baseline {metrics['majority_baseline']['mean']:.4f})")

    return {
        "dataset": ds.name,
        "branches": names,
        "embedding_dim": int(X.shape[1]),
        "n_labelled": int(len(lab)),
        "graph_scope": graph_scope,
        "config": {k: v for k, v in vars(cfg).items()},
        "metrics": metrics,
    }


def _relabel_split(ds: BotDataset, lab: np.ndarray) -> dict | None:
    """Map the dataset's official split indices into labelled-subset positions."""
    if not ds.split:
        return None
    pos = {int(g): i for i, g in enumerate(lab)}
    out = {}
    for k, v in ds.split.items():
        out[k] = np.array([pos[int(i)] for i in v if int(i) in pos], dtype=np.int64)
    return out if all(len(v) for v in out.values()) else None
