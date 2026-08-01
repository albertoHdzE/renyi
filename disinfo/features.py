"""Stage 1 of Fig. 1: feature extraction (Sect. 4).

The survey splits features into content-based (Sect. 4.1) and context-based
(Sect. 4.2), and Fig. 4 refines that into the tree ``taxonomy.FIG4_FEATURES``.
This module implements the leaves that the available data can actually support:

    Content / Linguistic / Lexical   ``text_features``  -- unigrams and bigrams,
                                     the survey's own example of a lexical
                                     feature.
    Content / Linguistic / Syntactic ``syntactic_features`` -- the punctuation
                                     and pronoun counts Sect. 4.1 lists.
    Context / User / Profile         ``profile_features``
    Context / Network / Propagation  ``propagation_features`` -- root degree,
                                     node count, average degree and tree depth,
                                     the four Sect. 4.2 names explicitly.
    Context / Network / Temporal     ``temporal_features``

Not implemented, for want of data rather than want of interest: visual features
(no images are redistributed with any of the four corpora), semantic features
(needs an external knowledge graph), and stance/comment features (reply text is
withheld by Twitter15/16 and CED). Their absence is the honest reason this
replication cannot reach the hybrid methods of Tables 1-2.

Every extractor returns a dense ``float32`` array with one row per item and
exposes ``feature_names`` so a fitted model can be inspected rather than
trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .config import Config
from .data import DisinfoDataset

__all__ = ["FeatureBundle", "text_features", "syntactic_features",
           "profile_features", "propagation_features", "temporal_features",
           "build_features"]


@dataclass
class FeatureBundle:
    """Node feature matrix plus the provenance of each block of columns."""
    X: np.ndarray
    names: list[str]
    blocks: dict[str, slice]

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    def block(self, name: str) -> np.ndarray:
        return self.X[:, self.blocks[name]]

    def summary(self) -> str:
        parts = [f"{k}={s.stop - s.start}" for k, s in self.blocks.items()]
        return f"X{self.X.shape}  " + "  ".join(parts)


# --------------------------------------------------------------------------
# content-based (Sect. 4.1)
# --------------------------------------------------------------------------

def _is_cjk(texts: list[str], sample: int = 200) -> bool:
    """Detect Chinese text, which needs character n-grams (CED has no spaces)."""
    joined = " ".join(texts[:sample])
    if not joined:
        return False
    cjk = sum(1 for ch in joined if "一" <= ch <= "鿿")
    return cjk / max(len(joined), 1) > 0.1


def text_features(texts: list[str], cfg: Config,
                  fit_on: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Lexical features: TF-IDF over unigrams and bigrams, then truncated SVD.

    Sect. 4.1 gives "unigrams, bigrams and the surface forms of words" as the
    definition of lexical features, so ``ngram_range=(1, cfg.ngram_max)``
    follows the text directly.

    Two details that matter numerically:

    * ``fit_on`` restricts vocabulary fitting to the training rows. Fitting
      TF-IDF on all rows leaks test-set term statistics; it inflates accuracy by
      a point or two and is a common silent error in this literature.
    * SVD reduces the sparse matrix to ``cfg.svd_dim`` dense dimensions. A GNN
      layer over a 5000-column sparse input is both slow and badly conditioned,
      and every method in Tables 1-2 feeds its GNN a dense embedding.
    """
    analyzer = "char_wb" if _is_cjk(texts) else "word"
    ngram = (1, cfg.ngram_max) if analyzer == "word" else (2, 3)

    vec = TfidfVectorizer(max_features=cfg.max_features, ngram_range=ngram,
                          min_df=cfg.min_df, analyzer=analyzer,
                          sublinear_tf=True, strip_accents="unicode")
    idx = np.arange(len(texts)) if fit_on is None else fit_on
    vec.fit([texts[i] for i in idx])
    S = vec.transform(texts)

    dim = min(cfg.svd_dim, S.shape[1] - 1)
    if dim < 2:
        return S.toarray().astype(np.float32), list(vec.get_feature_names_out())

    svd = TruncatedSVD(n_components=dim, random_state=cfg.seed)
    svd.fit(S[idx])
    X = svd.transform(S).astype(np.float32)
    names = [f"svd{i}" for i in range(dim)]
    return X, names


_PRONOUN_1 = re.compile(r"\b(i|me|my|mine|we|us|our|ours)\b", re.I)
_PRONOUN_3 = re.compile(r"\b(he|him|his|she|her|hers|they|them|their)\b", re.I)


def syntactic_features(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Syntactic features named in Sect. 4.1.

    "syntactic markers such as dots, question marks and exclamation marks...
    the usage of first-person or third-person pronouns". Counts are normalised
    by length so they measure style rather than verbosity, which is the point
    of a style-based feature.
    """
    rows = []
    for t in texts:
        n = max(len(t), 1)
        words = t.split()
        rows.append([
            len(words),
            np.mean([len(w) for w in words]) if words else 0.0,
            t.count(".") / n, t.count("?") / n, t.count("!") / n,
            t.count(",") / n,
            sum(ch.isupper() for ch in t) / n,
            sum(ch.isdigit() for ch in t) / n,
            len(_PRONOUN_1.findall(t)) / max(len(words), 1),
            len(_PRONOUN_3.findall(t)) / max(len(words), 1),
        ])
    names = ["n_words", "mean_word_len", "frac_period", "frac_question",
             "frac_exclam", "frac_comma", "frac_upper", "frac_digit",
             "frac_pron1", "frac_pron3"]
    return np.asarray(rows, dtype=np.float32), names


# --------------------------------------------------------------------------
# context-based (Sect. 4.2)
# --------------------------------------------------------------------------

_PROFILE_NAMES = ["log_followers", "log_friends", "follower_friend_ratio",
                  "log_statuses", "log_listed", "verified", "has_description"]


def profile_features(ds: DisinfoDataset) -> tuple[np.ndarray, list[str]]:
    """User-based profile features (Sect. 4.2), taken from the source author.

    "the number of posts, the age of the user account, the number of
    friends/followers and the verification status". LIAR has no user accounts,
    so its speaker metadata is one-hot-encoded party affiliation instead -- the
    nearest available analogue, and the attribute Hu et al. (2019) join on.
    """
    if ds.name.startswith("LIAR"):
        parties = sorted({(it.meta.get("party") or "none") for it in ds.items})
        idx = {p: i for i, p in enumerate(parties)}
        X = np.zeros((len(ds), len(parties)), dtype=np.float32)
        for r, it in enumerate(ds.items):
            X[r, idx[it.meta.get("party") or "none"]] = 1.0
        return X, [f"party={p}" for p in parties]

    rows = []
    for it in ds.items:
        c = it.cascade
        if c is not None and c.user_features is not None and len(c.user_features):
            rows.append(c.user_features[0])       # the source post's author
        else:
            rows.append(np.zeros(len(_PROFILE_NAMES), dtype=np.float32))
    return np.asarray(rows, dtype=np.float32), list(_PROFILE_NAMES)


def propagation_features(ds: DisinfoDataset) -> tuple[np.ndarray, list[str]]:
    """The four propagation features Sect. 4.2 names, plus two cheap relatives.

    "constructing a propagation graph/tree and analyzing its properties, such as
    root degree, number of nodes, average degree of nodes and tree depth".
    """
    rows = []
    for it in ds.items:
        c = it.cascade
        if c is None or c.size == 0:
            rows.append([0.0] * 6)
            continue
        n = c.size
        root_degree = sum(1 for p, _ in c.edges if p == 0)
        avg_degree = (2.0 * len(c.edges) / n) if n else 0.0
        unique_users = len(set(c.node_uids))
        rows.append([
            np.log1p(n), float(root_degree) / n, avg_degree, float(c.depth),
            float(len(c.edges)) / n,
            unique_users / n,          # 1.0 means nobody reposted twice
        ])
    names = ["log_n_nodes", "root_degree_frac", "avg_degree", "depth",
             "edges_per_node", "unique_user_frac"]
    return np.asarray(rows, dtype=np.float32), names


def temporal_features(ds: DisinfoDataset) -> tuple[np.ndarray, list[str]]:
    """Temporal features (Sect. 4.2): how fast the cascade ran.

    Times are minutes since the source post. Durations are log1p-compressed
    because cascade lifetimes span minutes to weeks.
    """
    rows = []
    for it in ds.items:
        c = it.cascade
        if c is None or c.size < 2:
            rows.append([0.0] * 5)
            continue
        t = np.asarray(c.times, dtype=np.float64)
        t = t[np.isfinite(t)]
        if t.size < 2:
            rows.append([0.0] * 5)
            continue
        span = float(t.max() - t.min())
        rows.append([
            np.log1p(max(span, 0.0)),
            np.log1p(max(float(np.median(t)), 0.0)),
            float(c.size) / (span / 60.0 + 1.0),       # reposts per hour
            np.log1p(max(float(np.percentile(t, 90)), 0.0)),
            float((t <= 60).mean()),                   # share within first hour
        ])
    names = ["log_span", "log_median_delay", "reposts_per_hour",
             "log_p90_delay", "frac_first_hour"]
    return np.asarray(rows, dtype=np.float32), names


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build_features(ds: DisinfoDataset, cfg: Config,
                   train_idx: np.ndarray | None = None) -> FeatureBundle:
    """Assemble the item-level feature matrix for stage 1 of Fig. 1.

    ``train_idx`` is passed to every fitted transformer so that vocabulary and
    scaling come from training rows only.
    """
    blocks: dict[str, slice] = {}
    parts: list[np.ndarray] = []
    names: list[str] = []

    def add(block: str, X: np.ndarray, ns: list[str]):
        start = sum(p.shape[1] for p in parts)
        parts.append(X)
        names.extend(ns)
        blocks[block] = slice(start, start + X.shape[1])

    Xt, nt = text_features(ds.texts, cfg, fit_on=train_idx)
    add("lexical", Xt, nt)
    add("syntactic", *syntactic_features(ds.texts))

    if cfg.use_profile:
        add("profile", *profile_features(ds))

    if ds.has_cascades:
        add("propagation", *propagation_features(ds))
        add("temporal", *temporal_features(ds))

    if cfg.use_credit_history and ds.name.startswith("LIAR"):
        cr = np.asarray([it.meta.get("credit", [0] * 5) for it in ds.items],
                        dtype=np.float32)
        add("credit", np.log1p(cr), [f"credit{i}" for i in range(cr.shape[1])])

    X = np.concatenate(parts, axis=1).astype(np.float32)

    # Scale on training rows only. Unscaled, the propagation block's raw depth
    # counts sit orders of magnitude above the SVD components and dominate both
    # the cosine similarities of the kNN graph and the first layer's gradients.
    idx = np.arange(len(ds)) if train_idx is None else train_idx
    scaler = StandardScaler().fit(X[idx])
    X = scaler.transform(X).astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return FeatureBundle(X, names, blocks)
