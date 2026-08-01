"""Dataset loaders for the bot-detection replication.

Fetch everything with ``bash scripts/get_bot_data.sh`` first.

The paper evaluates on **Cresci-15** and **TwiBot-22**. Neither is downloadable
without an application to its authors, so this replication uses what is openly
available:

``cresci-2015``   Complete, in the TwiBot-22 four-file schema published by the
                  TwiBot-22 authors: ``node.json`` (5,301 users **and** their
                  tweets), ``edge.csv`` (follow / friend / post), ``label.csv``,
                  ``split.csv``. This is the paper's headline dataset, and every
                  ingredient of the method exists for it, so the 98.68% result
                  is directly testable.

``twibot-22``     Partial: ``user.json``, ``label.csv`` and ``split.csv`` are
                  open on Zenodo (record 7012904), but ``edge.csv`` and the
                  tweet files are not. So the five user features and the labels
                  are exact, and the graph and text branches cannot run. What it
                  *does* support is the comparison that matters most for
                  Table 5 -- see ``twibot22_baseline_report``.

``twibot-20``     A preprocessed mirror in BotRGCN format: a graph, exactly five
                  standardised numeric user properties, 768-dimensional BERT
                  tweet embeddings, and labels. It is the only corpus here with
                  *all* ingredients already in the form the method needs, so it
                  carries the end-to-end pipeline. It is a **substitute** for
                  TwiBot-22, not the same dataset -- 11,826 labelled users
                  against 1,000,000 -- and results are labelled accordingly.

All three land in one container, ``BotDataset``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DATA_RAW

__all__ = ["BotDataset", "load_cresci15", "load_twibot20", "load_twibot22",
           "load_dataset", "twibot22_baseline_report", "FEATURE_NAMES"]

BOT_RAW = DATA_RAW / "bot"

# The five node features of Sect. 3.3, in the order used throughout.
FEATURE_NAMES = ["tweet_count", "followers_count", "following_count",
                 "favourites_count", "account_age_days"]


@dataclass
class BotDataset:
    """Users, their five features, the graph, their tweets, and labels."""

    name: str
    user_ids: list[str]
    features: np.ndarray                  # (n_users, 5) float32, raw
    labels: np.ndarray                    # (n_users,) int64; -1 = unlabelled
    edge_index: np.ndarray                # (2, E) int64, user-user
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    tweets: list[list[str]] | None = None   # per user, in file order
    text_embeddings: np.ndarray | None = None   # (n_users, 768) if precomputed
    split: dict[str, np.ndarray] | None = None  # official train/val/test indices
    label_names: tuple[str, str] = ("human", "bot")
    notes: str = ""

    def __len__(self) -> int:
        return len(self.user_ids)

    @property
    def labelled(self) -> np.ndarray:
        return np.flatnonzero(self.labels >= 0)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    def majority_baseline(self, idx: np.ndarray | None = None) -> float:
        """Accuracy of always predicting the more common class.

        The number every reported accuracy must be compared against, and the
        one the paper never states for either dataset.
        """
        y = self.labels[self.labelled if idx is None else idx]
        if y.size == 0:
            return float("nan")
        return float(np.bincount(y, minlength=2).max() / y.size)

    def isolated_fraction(self) -> float:
        """Share of users with no user-user edge -- they get no graph signal."""
        if self.n_edges == 0:
            return 1.0
        deg = np.bincount(self.edge_index[1], minlength=len(self))
        return float(np.mean(deg == 0))

    def summary(self) -> str:
        lab = self.labelled
        counts = np.bincount(self.labels[lab], minlength=2)
        out = [
            f"{self.name}: {len(self)} users "
            f"({len(lab)} labelled: {counts[0]} human, {counts[1]} bot)",
            f"  majority baseline : {self.majority_baseline():.4f}",
            f"  user-user edges   : {self.n_edges:,} "
            f"(isolated {self.isolated_fraction():.1%})",
            f"  features          : {self.features.shape}",
        ]
        if self.tweets is not None:
            n = sum(len(t) for t in self.tweets)
            out.append(f"  tweets            : {n:,} "
                       f"(median {int(np.median([len(t) for t in self.tweets]))}/user)")
        if self.text_embeddings is not None:
            out.append(f"  text embeddings   : {self.text_embeddings.shape}")
        if self.split:
            out.append("  official split    : " +
                       ", ".join(f"{k}={len(v)}" for k, v in self.split.items()))
        if self.notes:
            out.append(f"  note              : {self.notes}")
        return "\n".join(out)


# --------------------------------------------------------------------------
# Cresci-2015
# --------------------------------------------------------------------------

_CREATED_FMT = "%a %b %d %H:%M:%S %z %Y"
# Sect. 3.3 derives account age against "the current date". Because that is not
# a fixed quantity, we pin it to the corpus's collection year so the feature is
# reproducible. A constant shift is absorbed by a linear model's bias anyway.
CRESCI_REFERENCE_DATE = "2015-01-01"


def _account_age_days(created_at: str | None, reference: str) -> float:
    from datetime import datetime, timezone
    if not created_at:
        return 0.0
    try:
        t = datetime.strptime(created_at, _CREATED_FMT)
    except (ValueError, TypeError):
        return 0.0
    ref = datetime.fromisoformat(reference).replace(tzinfo=timezone.utc)
    return max((ref - t).days, 0)


def load_cresci15(root: Path | None = None, with_tweets: bool = True,
                  cache: bool = True) -> BotDataset:
    """Cresci-2015 in TwiBot-22 schema: 5,301 users, 3,351 bot / 1,950 human.

    Parsing is two-pass to keep memory bounded on a 367 MB ``node.json`` and a
    219 MB ``edge.csv``:

    1. ``edge.csv`` gives the user-user edges (``follow``, ``friend``) and the
       tweet-to-author map (``post``). Sect. 3.1.1 collapses every relation into
       one undirected edge type -- "the type of relation does not matter in this
       case" -- so follow and friend are merged.
    2. ``node.json`` is streamed; user nodes yield the five features, tweet
       nodes are routed to their author.

    **Feature substitution.** The TwiBot-22 conversion of Cresci-2015 exposes
    ``public_metrics`` = {followers_count, following_count, tweet_count,
    listed_count}. There is no ``favourites_count``, which is the paper's fourth
    feature, so ``listed_count`` stands in for it. Documented in
    ``docs/DISCREPANCIES_BOTSAGE.md``; the count of features (5) and everything
    downstream is unaffected.
    """
    base = Path(root or BOT_RAW) / "cresci-2015"
    cache_path = base / "_cached.npz"
    tweets_path = base / "_cached_tweets.json"

    if cache and cache_path.exists() and (tweets_path.exists() or not with_tweets):
        return _load_cresci_cache(base, with_tweets)

    # ---- labels and split ----
    labels_by_id: dict[str, int] = {}
    with open(base / "label.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            labels_by_id[r["id"]] = 1 if r["label"] == "bot" else 0

    split_by_id: dict[str, str] = {}
    with open(base / "split.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            split_by_id[r["id"]] = r["split"]

    # ---- pass 1: edges ----
    user_ids: list[str] = []
    index: dict[str, int] = {}

    def uidx(u: str) -> int:
        if u not in index:
            index[u] = len(user_ids)
            user_ids.append(u)
        return index[u]

    for u in labels_by_id:            # labelled users take the first indices
        uidx(u)

    src_list: list[int] = []
    dst_list: list[int] = []
    tweet_author: dict[str, int] = {}

    with open(base / "edge.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rel, s, t = r["relation"], r["source_id"], r["target_id"]
            if rel == "post":
                if s in index:
                    tweet_author[t] = index[s]
            elif s.startswith("u") and t.startswith("u"):
                # follow / friend -> one undirected relation (Sect. 3.1.1)
                a, b = uidx(s), uidx(t)
                if a != b:
                    src_list.append(a)
                    dst_list.append(b)

    n = len(user_ids)
    edges = np.stack([np.asarray(src_list, dtype=np.int64),
                      np.asarray(dst_list, dtype=np.int64)])
    edges = _to_undirected(edges)

    # ---- pass 2: nodes ----
    feats = np.zeros((n, 5), dtype=np.float32)
    tweets: list[list[str]] = [[] for _ in range(n)] if with_tweets else []

    for obj in _stream_json_array(base / "node.json"):
        oid = obj.get("id", "")
        if oid.startswith("u"):
            i = index.get(oid)
            if i is None:
                continue
            pm = obj.get("public_metrics") or {}
            feats[i] = [
                float(pm.get("tweet_count") or 0),
                float(pm.get("followers_count") or 0),
                float(pm.get("following_count") or 0),
                float(pm.get("listed_count") or 0),      # stands in for favourites
                _account_age_days(obj.get("created_at"), CRESCI_REFERENCE_DATE),
            ]
        elif with_tweets and oid.startswith("t"):
            a = tweet_author.get(oid)
            if a is not None:
                txt = obj.get("text")
                if txt:
                    tweets[a].append(txt)

    labels = np.full(n, -1, dtype=np.int64)
    for u, lab in labels_by_id.items():
        labels[index[u]] = lab

    split = {k: np.array(sorted(index[u] for u, s in split_by_id.items()
                                if s == k and u in index), dtype=np.int64)
             for k in ("train", "val", "test")}

    ds = BotDataset(
        name="Cresci-15", user_ids=user_ids, features=feats, labels=labels,
        edge_index=edges, tweets=tweets if with_tweets else None, split=split,
        feature_names=[*FEATURE_NAMES[:3], "listed_count", FEATURE_NAMES[4]],
        notes="favourites_count unavailable in the TwiBot-22 conversion; "
              "listed_count substituted (see DISCREPANCIES §13)",
    )
    if cache:
        _save_cresci_cache(base, ds)
    return ds


def _stream_json_array(path: Path):
    """Yield objects from a big top-level JSON array without loading it whole.

    ``node.json`` is 367 MB; ``json.load`` on it peaks around 3 GB of Python
    objects. Decoding incrementally with ``raw_decode`` keeps the footprint to
    one object plus the read buffer.
    """
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as fh:
        buf = fh.read(1 << 22)
        # skip whitespace and the opening bracket
        i = 0
        while i < len(buf) and buf[i] in " \n\r\t":
            i += 1
        if i < len(buf) and buf[i] == "[":
            i += 1
        buf = buf[i:]

        while True:
            buf = buf.lstrip(" \n\r\t,")
            if buf.startswith("]") or (not buf and not (chunk := fh.read(1 << 22))):
                return
            try:
                obj, end = dec.raw_decode(buf)
            except ValueError:
                chunk = fh.read(1 << 22)
                if not chunk:
                    return
                buf += chunk
                continue
            yield obj
            buf = buf[end:]
            if len(buf) < (1 << 20):
                buf += fh.read(1 << 22)


def _save_cresci_cache(base: Path, ds: BotDataset) -> None:
    np.savez_compressed(
        base / "_cached.npz", features=ds.features, labels=ds.labels,
        edge_index=ds.edge_index, user_ids=np.asarray(ds.user_ids),
        feature_names=np.asarray(ds.feature_names),
        **{f"split_{k}": v for k, v in (ds.split or {}).items()})
    if ds.tweets is not None:
        (base / "_cached_tweets.json").write_text(
            json.dumps(ds.tweets, ensure_ascii=False))


def _load_cresci_cache(base: Path, with_tweets: bool) -> BotDataset:
    z = np.load(base / "_cached.npz", allow_pickle=False)
    tweets = None
    if with_tweets:
        tweets = json.loads((base / "_cached_tweets.json").read_text())
    return BotDataset(
        name="Cresci-15", user_ids=list(z["user_ids"]), features=z["features"],
        labels=z["labels"], edge_index=z["edge_index"], tweets=tweets,
        feature_names=list(z["feature_names"]),
        split={k: z[f"split_{k}"] for k in ("train", "val", "test")
               if f"split_{k}" in z},
        notes="favourites_count unavailable in the TwiBot-22 conversion; "
              "listed_count substituted (see DISCREPANCIES §13)",
    )


def _to_undirected(edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return edges.reshape(2, 0)
    both = np.concatenate([edges, edges[::-1]], axis=1)
    both = both[:, both[0] != both[1]]
    return np.unique(both, axis=1)


# --------------------------------------------------------------------------
# TwiBot-20 (preprocessed, BotRGCN format)
# --------------------------------------------------------------------------

def load_twibot20(root: Path | None = None, with_text: bool = True) -> BotDataset:
    """TwiBot-20 in BotRGCN preprocessed form.

    Supplies all four ingredients the method needs, already in the right shape:
    ``num_properties_tensor.pt`` is (229580, **5**) -- the same feature count the
    paper uses -- ``edge_index.pt`` is the graph, ``tweets_tensor.pt`` is a
    768-dimensional per-user BERT embedding, and ``label.pt`` covers the 11,826
    annotated users.

    Two differences from the paper's own preprocessing, both documented:

    * The five properties are BotRGCN's (followers, active days, screen-name
      length, friends, statuses) and arrive **already z-scored**, so
      ``Config.standardize_features`` is a no-op here.
    * The tweet embeddings are BotRGCN's RoBERTa/BERT pooling over all tweets,
      not the paper's mean-of-token-means over the 15 most recent. Using them
      keeps the graph and text branches consistent with each other, at the cost
      of not being the paper's exact text pipeline. ``text.py`` implements the
      paper's pipeline and is used on Cresci-15, where raw tweets exist.
    """
    import torch

    base = Path(root or BOT_RAW) / "twibot-20"
    feats = torch.load(base / "num_properties_tensor.pt",
                       weights_only=False).numpy().astype(np.float32)
    ei = torch.load(base / "edge_index.pt", weights_only=False).numpy().astype(np.int64)
    lab = torch.load(base / "label.pt", weights_only=False).numpy().astype(np.int64)

    n = feats.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    labels[:lab.shape[0]] = lab           # labelled users occupy the first rows

    text = None
    tpath = base / "tweets_tensor.pt"
    if with_text and tpath.exists():
        text = torch.load(tpath, weights_only=False).numpy().astype(np.float32)

    split = None
    spath = base / "split_new.json"
    if spath.exists():
        raw = json.loads(spath.read_text())
        split = {("val" if k == "dev" else k): np.asarray(v, dtype=np.int64)
                 for k, v in raw.items()}

    return BotDataset(
        name="TwiBot-20", user_ids=[str(i) for i in range(n)], features=feats,
        labels=labels, edge_index=_to_undirected(ei), text_embeddings=text,
        split=split,
        feature_names=["followers", "active_days", "screen_name_length",
                       "friends", "statuses"],
        notes="preprocessed BotRGCN format; features already standardised. "
              "Substitute for TwiBot-22, which is not openly downloadable.",
    )


# --------------------------------------------------------------------------
# TwiBot-22 (partial: users + labels + split only)
# --------------------------------------------------------------------------

def load_twibot22(root: Path | None = None, max_users: int | None = None) -> BotDataset:
    """TwiBot-22 users, labels and official split, from the open Zenodo record.

    ``edge.csv`` and the tweet files are **not** in that record, so this loader
    returns an empty ``edge_index`` and no tweets. The graph and text branches
    of the method therefore cannot run on TwiBot-22 here; what can run is the
    five-feature classifier, and the baseline arithmetic of Table 5.
    """
    base = Path(root or BOT_RAW) / "twibot-22"

    labels_by_id: dict[str, int] = {}
    with open(base / "label.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            labels_by_id[r["id"]] = 1 if r["label"] == "bot" else 0

    split_by_id: dict[str, str] = {}
    with open(base / "split.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            split_by_id[r["id"]] = r["split"]

    user_ids: list[str] = []
    rows: list[list[float]] = []
    upath = base / "user.json"
    if upath.exists():
        for obj in _stream_json_array(upath):
            oid = obj.get("id", "")
            if oid not in labels_by_id:
                continue
            pm = obj.get("public_metrics") or {}
            user_ids.append(oid)
            rows.append([
                float(pm.get("tweet_count") or 0),
                float(pm.get("followers_count") or 0),
                float(pm.get("following_count") or 0),
                float(pm.get("listed_count") or 0),
                _account_age_days(obj.get("created_at"), "2022-01-01"),
            ])
            if max_users and len(user_ids) >= max_users:
                break
        feats = np.asarray(rows, dtype=np.float32)
    else:
        # Labels alone still support the baseline analysis.
        user_ids = list(labels_by_id)
        feats = np.zeros((len(user_ids), 5), dtype=np.float32)

    index = {u: i for i, u in enumerate(user_ids)}
    labels = np.asarray([labels_by_id[u] for u in user_ids], dtype=np.int64)
    split = {k: np.array(sorted(index[u] for u, s in split_by_id.items()
                                if s == k and u in index), dtype=np.int64)
             for k in ("train", "val", "test")}

    return BotDataset(
        name="TwiBot-22", user_ids=user_ids, features=feats, labels=labels,
        edge_index=np.zeros((2, 0), dtype=np.int64), split=split,
        notes="edge.csv and tweet files are not in the open Zenodo record; "
              "graph and text branches unavailable",
    )


def twibot22_baseline_report() -> dict:
    """The arithmetic behind DISCREPANCIES §2, from the released labels.

    Returns the class balance of the whole corpus and of each official split,
    with the corresponding majority-class accuracy. The paper evaluates by
    5-fold cross-validation over the corpus, so the figure its accuracies must
    clear is the **corpus-wide** baseline; the leaderboard numbers it compares
    itself against were computed on the official **test** split, whose balance
    is different. Both are reported here.
    """
    base = BOT_RAW / "twibot-22"
    labels: dict[str, int] = {}
    with open(base / "label.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            labels[r["id"]] = 1 if r["label"] == "bot" else 0

    out = {}
    y = np.asarray(list(labels.values()))
    out["corpus"] = _balance(y)

    if (base / "split.csv").exists():
        by_split: dict[str, list[int]] = {}
        with open(base / "split.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["id"] in labels:
                    by_split.setdefault(r["split"], []).append(labels[r["id"]])
        for k, v in by_split.items():
            out[k] = _balance(np.asarray(v))
    return out


def _balance(y: np.ndarray) -> dict:
    c = np.bincount(y, minlength=2)
    return {"n": int(y.size), "human": int(c[0]), "bot": int(c[1]),
            "bot_frac": float(c[1] / y.size),
            "majority_baseline": float(c.max() / y.size)}


# --------------------------------------------------------------------------

_LOADERS = {
    "cresci-15": load_cresci15, "cresci15": load_cresci15,
    "cresci-2015": load_cresci15,
    "twibot-20": load_twibot20, "twibot20": load_twibot20,
    "twibot-22": load_twibot22, "twibot22": load_twibot22,
}


def load_dataset(name: str, **kwargs) -> BotDataset:
    key = name.lower().replace(" ", "")
    if key not in _LOADERS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(set(_LOADERS))}")
    return _LOADERS[key](**kwargs)
