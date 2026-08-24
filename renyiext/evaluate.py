"""Evaluation layer: one shared, correctly-calibrated evaluation module.

Generalised from ``scripts/run_p2_temporal.py`` (plan WP-E task 1), which
until now owned its own copies of every metric; ``run_p2b_decomposition.py`
carried a second verbatim copy. Both scripts import from here.

Layering (``renyiext/__init__.py``): features -> models -> **evaluate**.
This module imports only numpy/scipy/sklearn -- nothing from the package --
so any producer (temporal, behavioural, text, AIT fronts) can sit on top of
it without cycles.

Protocol fixed by ``docs/02-PROTOCOL.md`` sect. 5-6 and plan §8:

    seeds       range(42, 42+n), default n=10; they drive fold assignment
                and classifier initialisation
    CV          StratifiedKFold(5, shuffle=True, random_state=seed)
    classifier  HistGradientBoostingClassifier(max_iter=200,
                early_stopping=False, random_state=seed);
                LR secondary = StandardScaler + LogisticRegression(max_iter=5000),
                scaler fitted on the training fold only
    metrics     AUC; TPR@1%FPR pooled AND per-fold mean (tpr01_foldmean --
                review C3: the pooled statistic mixes calibrations across
                folds); macro-F1; accuracy. Majority baseline is the
                producer's job (it needs only y).
    comparison  paired Wilcoxon over seeds, two-sided p < 0.05, effect-size
                floor 0.02 AUC (untouchable except by numbered amendment)

Dimensionality control (review C2), plan §8 D2: wherever
``dim(family) > dim(floor)``, evaluate the floor padded with ``k``
standard-normal columns, ``k = dim(family) - dim(floor)``. Noise draws come
from ``default_rng(seed*1000 + arm_index)`` -- distinct integer salts per arm
so no two arms ever share a draw. Failure semantics are PRE-REGISTERED
(plan WP-E task 3, [rev1]); :func:`interpret_dim_matched` implements them
and its verdict is binding on HANDOFF/FINDINGS, not advisory.

Uncertainty reporting (review D4), plan §8 D3: :func:`sigma_config` is the
population SD (ddof=0) of a comparison's delta across the full published
config sweep, reported beside every floor verdict. It measures how much the
comparison depends on the invented grid parameters -- the uncertainty the
seed SD does not capture. The 0.02 floor itself is untouched.

Determinism: every function here is a pure function of its inputs given the
seeds; two runs of a producer must agree to the digit (S2.6).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["tpr_at_fpr", "eval_arm", "run_arms", "paired", "partial_corr",
           "noise_padding", "dim_matched_arm", "interpret_dim_matched",
           "sigma_config"]


def tpr_at_fpr(y, s, target: float = 0.01) -> float:
    """TPR at FPR = ``target`` (default 1%) -- the deployment regime.

    Linear interpolation on the pooled ROC curve, exactly as
    ``scripts/run_p2_temporal.py`` computed it before WP-E (docs/02-PROTOCOL.md
    sect. 6: suspending a real user is the cost of a false positive).
    """
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target, fpr, tpr))


def _make_model(model: str, seed: int):
    if model == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, max_iter=200,
                                              early_stopping=False)
    if model == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000))
    raise ValueError(f"unknown model {model!r}; expected 'hgb' or 'lr'")


def eval_arm(X, y, seed: int, model: str = "hgb", n_folds: int = 5) -> dict:
    """One arm, one seed: stratified CV, out-of-fold scores, all metrics.

    Returns ``auc``, ``tpr_at_1pct_fpr`` (pooled OOF -- kept for continuity
    with every number already published), ``tpr01_foldmean`` (mean of the
    per-fold TPR@1%FPR -- review C3's calibration-clean statistic),
    ``macro_f1``, ``accuracy``. The split, classifier seeding and threshold
    are byte-compatible with the pre-WP-E producer: same calls, same order.
    """
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    fold_tpr = []
    for tr, te in skf.split(X, y):
        clf = _make_model(model, seed)
        clf.fit(X[tr], y[tr])          # scaler fitted on the training fold only
        oof[te] = clf.predict_proba(X[te])[:, 1]
        fold_tpr.append(tpr_at_fpr(y[te], oof[te]))
    return {
        "auc": float(roc_auc_score(y, oof)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, oof),
        "tpr01_foldmean": float(np.mean(fold_tpr)),
        "macro_f1": float(f1_score(y, (oof > 0.5).astype(int), average="macro")),
        "accuracy": float(accuracy_score(y, (oof > 0.5).astype(int))),
    }


def run_arms(arms: dict, y, seeds, model: str = "hgb", quiet: bool = False) -> dict:
    """Evaluate every arm over ``seeds``.

    ``arms`` maps name -> design matrix, or name -> callable(seed) -> matrix
    for arms whose features depend on the seed (the D2 noise-padded floors).
    Output per arm: ``n_features`` plus per-seed lists for ``auc``, ``tpr01``,
    ``tpr01_foldmean``, ``macro_f1``, ``accuracy``.
    """
    out = {}
    for name, spec in arms.items():
        per_seed = []
        n_features = None
        for s in seeds:
            X = spec(s) if callable(spec) else spec
            n_features = int(np.asarray(X).shape[1])
            per_seed.append(eval_arm(X, y, s, model))
        out[name] = {"n_features": n_features,
                     "auc": [r["auc"] for r in per_seed],
                     "tpr01": [r["tpr_at_1pct_fpr"] for r in per_seed],
                     "tpr01_foldmean": [r["tpr01_foldmean"] for r in per_seed],
                     "macro_f1": [r["macro_f1"] for r in per_seed],
                     "accuracy": [r["accuracy"] for r in per_seed]}
        if not quiet:
            a = np.array(out[name]["auc"])
            print(f"    {name:<22} AUC {a.mean():.4f} +- {a.std():.4f}", flush=True)
    return out


def paired(a, b) -> dict:
    """Paired Wilcoxon over seeds, in the repository's standard format.

    Keys: ``mean_diff``, ``std_diff``, ``wins``, ``p`` (two-sided),
    ``significant`` (p < 0.05), ``clears_floor`` (mean_diff > 0.02 AND
    significant). The 0.02 effect-size floor is protocol sect. 5 and moves
    only via a numbered amendment.
    """
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return {"mean_diff": float(d.mean()), "std_diff": float(d.std()),
            "wins": f"{int((d > 0).sum())}/{len(d)}", "p": p,
            "significant": bool(p < 0.05),
            "clears_floor": bool(d.mean() > 0.02 and p < 0.05)}


def partial_corr(x, yv, z) -> float:
    """Correlation of x and y, each linearly detrended on z.

    First-order partial correlation via linear residuals -- the G4 tool:
    e.g. correlation of a feature with the label GIVEN log event count.
    """
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = yv - np.polyval(np.polyfit(z, yv, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])


def noise_padding(X_floor, k: int, seed: int, arm_index: int = 0) -> np.ndarray:
    """Plan §8 D2: ``X_noisy = [X_floor || N]`` with pure-noise dimensions.

    ``N ~ default_rng(seed*1000 + arm_index).standard_normal((n, k))``.
    ``arm_index`` is the distinct integer salt that keeps different arms off
    shared draws (§8 preamble: derived streams never share noise). ``k`` is
    NOT tuned: it is always ``dim(family) - dim(floor)`` so the padded floor
    has exactly the family's dimensionality (G3: the knob is printed in the
    producer's config echo, formula-determined, nothing free).
    """
    X_floor = np.asarray(X_floor, dtype=float)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    rng = np.random.default_rng(seed * 1000 + arm_index)
    N = rng.standard_normal((X_floor.shape[0], k))
    return np.hstack([X_floor, N])


def dim_matched_arm(X_family, X_floor, seed: int, arm_index: int = 0) -> np.ndarray:
    """The D2 control: the floor padded to the family's dimensionality."""
    X_family = np.asarray(X_family)
    return noise_padding(X_floor, X_family.shape[1] - np.asarray(X_floor).shape[1],
                         seed, arm_index)


def interpret_dim_matched(mean_diff: float, significant: bool) -> str:
    """Pre-registered failure semantics for a dim-matched comparison
    (plan WP-E task 3, [rev1] -- binding, not advisory):

    * ``confounded_dimensionality``  (Delta <= 0): the family measured
      dimensionality, not information; clauses resting on it are recorded as
      confounded and their support is DOWNGRADED in HANDOFF and FINDINGS.
    * ``real_but_subfloor_not_claimable``  (0 < Delta < 0.02): the edge is
      real-but-subfloor under matched dimensions; the clause is recorded as
      not claimable at the registered floor.
    * ``supports_clause``  (Delta >= 0.02 AND significant): the matched
      control does not explain the edge away.
    * ``not_supported_not_significant``  (Delta >= 0.02, p >= 0.05): fails
      the significance half of rule 3; treated as not supporting (the
      conservative reading of "Only Delta >= 0.02 (+ significance)").

    The registered (unmatched) gate verdict stands as gated either way;
    this attaches the matched-dimension reading to it.
    """
    if mean_diff <= 0:
        return "confounded_dimensionality"
    if mean_diff < 0.02:
        return "real_but_subfloor_not_claimable"
    if significant:
        return "supports_clause"
    return "not_supported_not_significant"


def sigma_config(deltas) -> float:
    """Plan §8 D3: population SD (ddof=0) of a comparison's delta across the
    full published config sweep. Reported beside every floor verdict next to
    the seed SD: the seed sigma warrants the 0.02 floor, but the config
    sigma is often the larger honest uncertainty (review D4).
    """
    return float(np.std(np.asarray(deltas, dtype=float), ddof=0))
