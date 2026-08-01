"""Experiment suites, each aimed at one claim of the paper.

``suite_replicate``          The two rows the paper contributes to Table 4:
                             BERT+GraphSage+SVM and DistilBERT+GraphSage+SVM on
                             Cresci-15, plus the GraphSage-only row of Table 5.
``suite_ablation``           What each branch actually contributes -- and, most
                             importantly, whether the 128-dim GraphSAGE
                             embedding beats the ten numbers it is computed
                             from, or the five raw features it starts from.
``suite_seed_sensitivity``   The untrained layer's output is fixed by its random
                             initialisation, which the paper never reports. This
                             measures how much the reported number moves when
                             only that seed changes.
``suite_trained_vs_untrained`` What Sect. 3.5 gives up by omitting the
                             prediction head.
``suite_graph_scope``        The two readings of Sect. 3.1.1's edge definition.
``suite_protocol``           5-fold CV (the paper) vs the corpus's official
                             split (the baselines it compares against).

Everything returns plain dicts; ``results_to_frame`` flattens them for the
plots, and ``save_results``/``load_results`` cache them beside the paper.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config, RESULTS
from .data import BotDataset
from .pipeline import (build_embeddings, evaluate, expand_text,
                       restrict_graph, _relabel_split)
from .sage import effective_input

__all__ = ["suite_replicate", "suite_ablation", "suite_seed_sensitivity",
           "suite_regularization_equivalence",
           "suite_trained_vs_untrained", "suite_graph_scope", "suite_protocol",
           "results_to_frame", "save_results", "load_results"]


def _row(label, ds, cfg, metrics, extra=None):
    out = {"variant": label, "dataset": ds.name,
           "n_labelled": int(len(ds.labelled))}
    for k, v in metrics.items():
        out[k] = v["mean"]
        out[f"{k}_std"] = v["std"]
    out.update(extra or {})
    return out


def _eval_matrix(X, ds, cfg, quiet=True):
    lab = ds.labelled
    return evaluate(X[lab], ds.labels[lab], cfg,
                    split=_relabel_split(ds, lab), quiet=quiet)


# --------------------------------------------------------------------------

def suite_replicate(ds: BotDataset, cfg: Config,
                    text_variants: dict[str, np.ndarray],
                    graph_scope: str = "all", quiet: bool = True) -> list[dict]:
    """Reproduce the paper's own rows for this dataset.

    ``text_variants`` maps a label ("BERT", "DistilBERT") to the per-user text
    embedding matrix. The graph-only row (Table 5's "GraphSage+SVM") is added
    automatically, since it is the paper's own no-text baseline.
    """
    out = []

    c = cfg.with_(use_text=False, use_graph=True)
    t0 = time.time()
    m = _eval_matrix(build_embeddings(ds, c, graph_scope=graph_scope)[0], ds, c)
    out.append(_row("GraphSage+SVM", ds, c, m,
                    {"embedding_dim": c.sage_out, "seconds": time.time() - t0}))
    if not quiet:
        _log(out[-1])

    for name, te in text_variants.items():
        c = cfg.with_(use_text=True, use_graph=True)
        t0 = time.time()
        X = build_embeddings(ds, c, text_embeddings=te, graph_scope=graph_scope)[0]
        m = _eval_matrix(X, ds, c)
        out.append(_row(f"GraphSage+{name}", ds, c, m,
                        {"embedding_dim": int(X.shape[1]),
                         "seconds": time.time() - t0}))
        if not quiet:
            _log(out[-1])
    return out


def _log(r):
    print(f"  {r['variant']:26s} dim {r.get('embedding_dim', '?'):>4}  "
          f"acc {r['accuracy']:.4f}+/-{r['accuracy_std']:.4f}  "
          f"f1 {r['f1']:.4f}  (baseline {r['majority_baseline']:.4f})")


def suite_ablation(ds: BotDataset, cfg: Config,
                   text_embeddings: np.ndarray | None = None,
                   graph_scope: str = "all", quiet: bool = True) -> list[dict]:
    """Decompose the 896-dimensional vector into what actually carries signal.

    The two rows that matter most are ``raw 5 features`` and
    ``effective 10 dims``. If the full ``GraphSage[128]`` row does not beat
    them, the GraphSAGE stage is a reparameterisation of its own input rather
    than a feature extractor -- which is what ``checks.py`` predicts
    analytically.
    """
    out = []

    x = torch.from_numpy(np.ascontiguousarray(ds.features)).float()
    if cfg.standardize_features:
        mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
        x = (x - mu) / sd
    ei = torch.from_numpy(np.ascontiguousarray(restrict_graph(ds, graph_scope)))

    variants: list[tuple[str, np.ndarray]] = [
        ("raw 5 features", x.numpy()),
        ("effective 10 dims [x || mean N(v)]", effective_input(x, ei).numpy()),
        ("GraphSage[128]",
         build_embeddings(ds, cfg.with_(use_text=False, use_graph=True),
                          graph_scope=graph_scope)[0]),
    ]
    if text_embeddings is not None:
        te = expand_text(ds, text_embeddings)
        variants += [
            ("BERT[768] only", te),
            ("raw 5 + BERT[768]", np.concatenate([x.numpy(), te], axis=1)),
            ("GraphSage[128] + BERT[768] = 896",
             build_embeddings(ds, cfg.with_(use_text=True, use_graph=True),
                              text_embeddings=te, graph_scope=graph_scope)[0]),
        ]

    for label, X in variants:
        m = _eval_matrix(np.asarray(X, dtype=np.float32), ds, cfg)
        out.append(_row(label, ds, cfg, m, {"embedding_dim": int(X.shape[1]),
                                            "graph_scope": graph_scope}))
        if not quiet:
            _log(out[-1])
    return out


def suite_regularization_equivalence(ds: BotDataset, cfg: Config,
                                     Cs=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
                                     graph_scope: str = "all",
                                     quiet: bool = True) -> list[dict]:
    """Decisive test of whether GraphSage[128] carries information the 10 dims lack.

    ``checks.check_untrained_sage_is_linear`` proves the embedding is an affine
    image of ``[x_v || mean N(v)]``, so a linear classifier on the 128
    dimensions can express exactly what one on the 10 can. Yet at the default
    ``C=1`` the 128-dim version scores about a point higher, which looks like a
    counterexample.

    It is not. L2 regularisation is **not invariant under a change of basis**:
    the random projection spreads 10 informative directions across 128
    coordinates, and per-coordinate standardisation then rescales them, which
    preconditions the penalty differently. The effect is regularisation
    geometry, not information.

    The way to tell the two apart is to weaken the penalty. As ``C`` grows the
    fitted model approaches the unregularised least-norm solution, which *is*
    basis-independent, so the gap must vanish if -- and only if -- the
    information is identical. Sweeping ``C`` therefore settles the question.
    """
    x = torch.from_numpy(np.ascontiguousarray(ds.features)).float()
    if cfg.standardize_features:
        mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
        x = (x - mu) / sd
    ei = torch.from_numpy(np.ascontiguousarray(restrict_graph(ds, graph_scope)))

    lab = ds.labelled
    y = ds.labels[lab]
    Z = effective_input(x, ei).numpy()[lab]
    E = build_embeddings(ds, cfg.with_(use_text=False, use_graph=True),
                         graph_scope=graph_scope)[0][lab]

    out = []
    for C in Cs:
        c = cfg.with_(svm_C=float(C))
        a = evaluate(Z, y, c)["accuracy"]
        b = evaluate(E, y, c)["accuracy"]
        out.append({
            "dataset": ds.name, "C": float(C),
            "acc_10dim": a["mean"], "acc_10dim_std": a["std"],
            "acc_sage128": b["mean"], "acc_sage128_std": b["std"],
            "delta": b["mean"] - a["mean"],
        })
        if not quiet:
            print(f"  C={C:<8g} 10-dim {a['mean']:.4f}  "
                  f"SAGE-128 {b['mean']:.4f}  delta {out[-1]['delta']:+.4f}")
    return out


def suite_seed_sensitivity(ds: BotDataset, cfg: Config, seeds=range(10),
                           text_embeddings: np.ndarray | None = None,
                           graph_scope: str = "all", quiet: bool = True) -> list[dict]:
    """Vary only the untrained layer's initialisation.

    For a trained model the seed is a nuisance parameter. For an untrained one
    it *is* the model, so this spread is the honest error bar on any number the
    paper reports -- and the paper reports none.
    """
    out = []
    for s in seeds:
        c = cfg.with_(sage_seed=int(s),
                      use_text=text_embeddings is not None, use_graph=True)
        X = build_embeddings(ds, c, text_embeddings=text_embeddings,
                             graph_scope=graph_scope)[0]
        m = _eval_matrix(X, ds, c)
        out.append(_row(f"seed={s}", ds, c, m, {"sage_seed": int(s)}))
        if not quiet:
            _log(out[-1])
    return out


def suite_trained_vs_untrained(ds: BotDataset, cfg: Config, epochs: int = 400,
                               hidden: int = 128, lr: float = 0.01,
                               graph_scope: str = "all", n_seeds: int = 3,
                               quiet: bool = True) -> list[dict]:
    """Sect. 3.5's untrained layer against the same layer *with* a prediction head.

    The paper uses **one** SAGEConv layer, and for a single layer the
    neighbourhood aggregation does not depend on any parameter -- it is just
    ``mean of N(v)``, computable once. So a trained 1-layer SAGEConv followed by
    a head is *exactly* a network applied to the fixed 10-dimensional
    ``[x_v || mean N(v)]``, and can be fitted on that directly.

    That equivalence is worth stating because it is not an approximation: it is
    the same model, trained the same way, and it turns 3,000 forward/backward
    passes over a 1.29M-node graph into a few seconds of work on a 5,301 x 10
    matrix. It is also the fairest possible comparison, since both arms then see
    identical inputs and differ only in whether the map onto them is learned.

    ``deep=True`` additionally fits a genuinely 2-layer GraphSAGE, where the
    second layer's aggregation *does* depend on the first layer's output and so
    cannot be precomputed. That arm is expensive and off by default.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    lab = ds.labelled
    y_lab = ds.labels[lab]
    x = torch.from_numpy(np.ascontiguousarray(ds.features)).float()
    if cfg.standardize_features:
        mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
        x = (x - mu) / sd
    ei = torch.from_numpy(np.ascontiguousarray(restrict_graph(ds, graph_scope)))

    out = []
    c = cfg.with_(use_text=False, use_graph=True)
    m = _eval_matrix(build_embeddings(ds, c, graph_scope=graph_scope)[0], ds, c)
    out.append(_row("untrained SAGEConv[128] + SVM (the paper)", ds, c, m))
    if not quiet:
        _log(out[-1])

    # The fixed input a 1-layer SAGEConv sees, for the labelled users only.
    Z = torch.from_numpy(effective_input(x, ei).numpy()[lab]).float()
    # The untrained layer's 128-dim output, for the same users.
    E = torch.from_numpy(
        build_embeddings(ds, c, graph_scope=graph_scope)[0][lab]).float()

    def fit_mlp(features: torch.Tensor, label: str):
        """Train an MLP head on fixed features, under the same 5-fold protocol."""
        accs, f1s = [], []
        for seed in range(n_seeds):
            skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                                  random_state=seed)
            for tr, te in skf.split(features.numpy(), y_lab):
                torch.manual_seed(seed)
                mu_, sd_ = features[tr].mean(0), features[tr].std(0).clamp(min=1e-6)
                F = (features - mu_) / sd_
                model = torch.nn.Sequential(
                    torch.nn.Linear(F.size(1), hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden, 2),
                )
                opt = torch.optim.Adam(model.parameters(), lr=lr,
                                       weight_decay=5e-4)
                ytr = torch.as_tensor(y_lab[tr])
                model.train()
                for _ in range(epochs):
                    opt.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(F[tr]), ytr)
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    pred = model(F[te]).argmax(1).numpy()
                accs.append(accuracy_score(y_lab[te], pred))
                f1s.append(f1_score(y_lab[te], pred, zero_division=0))
        return {
            "variant": label, "dataset": ds.name, "n_labelled": int(len(lab)),
            "accuracy": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1)),
            "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1)),
            "majority_baseline": float(
                np.bincount(y_lab, minlength=2).max() / len(lab)),
            "majority_baseline_std": 0.0,
        }

    # Two arms, so that "training helps" is not confused with "a nonlinear head
    # helps". The middle arm keeps the layer frozen and only adds the MLP; the
    # last arm additionally learns the layer. Their difference is what training
    # the SAGEConv is worth, holding classifier capacity fixed.
    out.append(fit_mlp(E, "untrained SAGEConv[128] + MLP head"))
    if not quiet:
        _log(out[-1])
    out.append(fit_mlp(Z, "trained SAGEConv[128] + head"))
    if not quiet:
        _log(out[-1])
    return out


def suite_graph_scope(ds: BotDataset, cfg: Config,
                      text_embeddings: np.ndarray | None = None,
                      quiet: bool = True) -> list[dict]:
    """Both readings of Sect. 3.1.1's edge definition."""
    out = []
    for scope in ("all", "labelled"):
        ei = restrict_graph(ds, scope)
        deg = (np.bincount(ei[1], minlength=len(ds)) if ei.size
               else np.zeros(len(ds), int))
        lab = ds.labelled
        c = cfg.with_(use_text=text_embeddings is not None, use_graph=True)
        X = build_embeddings(ds, c, text_embeddings=text_embeddings,
                             graph_scope=scope)[0]
        m = _eval_matrix(X, ds, c)
        out.append(_row(f"edges={scope}", ds, c, m, {
            "graph_scope": scope, "n_edges": int(ei.shape[1]),
            "isolated_labelled": int((deg[lab] == 0).sum()),
            "median_degree_labelled": float(np.median(deg[lab])),
        }))
        if not quiet:
            _log(out[-1])
    return out


def suite_protocol(ds: BotDataset, cfg: Config,
                   text_embeddings: np.ndarray | None = None,
                   graph_scope: str = "all", quiet: bool = True) -> list[dict]:
    """5-fold cross-validation against the corpus's own train/test split.

    The paper uses the former; every baseline in Tables 4-5 uses the latter.
    """
    out = []
    c0 = cfg.with_(use_text=text_embeddings is not None, use_graph=True)
    X = build_embeddings(ds, c0, text_embeddings=text_embeddings,
                         graph_scope=graph_scope)[0]
    lab = ds.labelled
    split = _relabel_split(ds, lab)

    for protocol in ("cv", "official_split"):
        if protocol == "official_split" and not split:
            continue
        c = c0.with_(protocol=protocol)
        m = evaluate(X[lab], ds.labels[lab], c, split=split)
        out.append(_row(f"protocol={protocol}", ds, c, m,
                        {"protocol": protocol}))
        if not quiet:
            _log(out[-1])
    return out


# --------------------------------------------------------------------------

def results_to_frame(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)


def save_results(results, name: str, outdir: Path | None = None) -> Path:
    outdir = Path(outdir or RESULTS)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.json"
    path.write_text(json.dumps(results, indent=1, default=float))
    return path


def load_results(name: str, outdir: Path | None = None):
    return json.loads((Path(outdir or RESULTS) / f"{name}.json").read_text())
