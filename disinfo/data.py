"""Loaders for the four datasets of Table 3 that are still publicly obtainable.

Fetch them with ``bash scripts/get_disinfo_data.sh`` first.

Everything is loaded into one structure, ``DisinfoDataset``, holding a list of
``Item``. An item always has text and a label; it *may* also carry a
``Cascade`` -- the propagation tree of Sect. 4.2. That optional field is exactly
the survey's own dividing line:

* items without a cascade support only **similarity graphs** (Sect. 5.3, first
  homogeneous approach), giving one graph over all items and a node
  classification task;
* items with a cascade also support **propagation graphs** (second homogeneous
  approach) and **heterogeneous graphs**, giving one graph per item and a graph
  classification task.

So LIAR can only exercise the content-based branch of the survey, while
Twitter15/16, PHEME and CED exercise both. That asymmetry is a fact about the
data, and it is why Table 1's LIAR row is a similarity-graph method.

Label spaces are kept in the source vocabulary and mapped to a canonical
4-class scheme (``true / false / unverified / non-rumor``) where the source
supports it, so results are comparable across Twitter15/16 and PHEME the way
Tables 1-2 compare them.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DATA_RAW

__all__ = ["Cascade", "Item", "DisinfoDataset", "load_liar", "load_twitter",
           "load_pheme", "load_ced", "load_dataset", "dataset_sizes",
           "CANONICAL_LABELS"]


CANONICAL_LABELS = ["non-rumor", "true", "false", "unverified"]


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------

@dataclass
class Cascade:
    """A propagation tree: who reposted what, when.

    ``node_uids[i]`` is the author of cascade node ``i``; node 0 is always the
    source post. ``times[i]`` is the delay from the source in **minutes**
    (Twitter15/16 record minutes natively; PHEME and CED timestamps are
    converted). ``edges`` are ``(parent, child)`` index pairs.
    """
    node_uids: list[str]
    times: np.ndarray
    edges: list[tuple[int, int]]
    user_features: np.ndarray | None = None   # (n_nodes, d) profile features

    @property
    def size(self) -> int:
        return len(self.node_uids)

    @property
    def depth(self) -> int:
        """Longest root-to-leaf path, a propagation feature named in Sect. 4.2."""
        if not self.edges:
            return 0
        children: dict[int, list[int]] = {}
        for p, c in self.edges:
            children.setdefault(p, []).append(c)
        depth, frontier = 0, [0]
        seen = {0}
        while frontier:
            nxt = [c for p in frontier for c in children.get(p, []) if c not in seen]
            seen.update(nxt)
            if nxt:
                depth += 1
            frontier = nxt
        return depth


@dataclass
class Item:
    """One information item d_i of the problem definition in Sect. 5.1."""
    id: str
    text: str
    label: str
    meta: dict = field(default_factory=dict)
    cascade: Cascade | None = None


@dataclass
class DisinfoDataset:
    name: str
    items: list[Item]
    label_names: list[str]
    canonical: bool = False        # labels drawn from CANONICAL_LABELS

    def __len__(self) -> int:
        return len(self.items)

    @property
    def texts(self) -> list[str]:
        return [it.text for it in self.items]

    @property
    def y(self) -> np.ndarray:
        idx = {lab: i for i, lab in enumerate(self.label_names)}
        return np.array([idx[it.label] for it in self.items], dtype=np.int64)

    @property
    def has_cascades(self) -> bool:
        return any(it.cascade is not None for it in self.items)

    def label_counts(self) -> dict[str, int]:
        out = {lab: 0 for lab in self.label_names}
        for it in self.items:
            out[it.label] += 1
        return out

    def binarised(self, fake_labels=("false", "pants-fire", "barely-true")) -> "DisinfoDataset":
        """Collapse to {fake, real}.

        Needed to test the Cui et al. LIAR anomaly (Table 2 reports ACC 0.868 on
        a 6-class dataset where Table 1 reports 0.492). See
        ``docs/DISCREPANCIES_SURVEY.md``.
        """
        items = [Item(it.id, it.text,
                      "fake" if it.label in fake_labels else "real",
                      it.meta, it.cascade) for it in self.items]
        return DisinfoDataset(f"{self.name}-binary", items, ["real", "fake"])

    def summary(self) -> str:
        c = self.label_counts()
        casc = [it.cascade for it in self.items if it.cascade is not None]
        line = (f"{self.name}: {len(self)} items, {len(self.label_names)} labels "
                f"({', '.join(f'{k}={v}' for k, v in c.items())})")
        if casc:
            sizes = np.array([x.size for x in casc])
            line += (f"\n  cascades: {len(casc)}, "
                     f"nodes median={int(np.median(sizes))} max={sizes.max()}, "
                     f"mean depth={np.mean([x.depth for x in casc]):.1f}")
        return line


# --------------------------------------------------------------------------
# LIAR
# --------------------------------------------------------------------------

LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true",
               "mostly-true", "true"]

# Column names of the released TSV, which ships without a header.
_LIAR_COLS = ["id", "label", "statement", "subject", "speaker", "job", "state",
              "party", "barely_true_counts", "false_counts",
              "half_true_counts", "mostly_true_counts", "pants_fire_counts",
              "context"]


def load_liar(root: Path | None = None, split: str = "all") -> DisinfoDataset:
    """LIAR (Wang 2017). Table 3: politics, text, 6 labels, 12,836 items.

    The speaker-profile columns are kept in ``meta`` because Hu et al. (2019),
    the LIAR row of Table 1, build their graph by joining items that share a
    profile attribute -- see ``graphs.attribute_graph``.

    The five ``*_counts`` columns are the speaker's credit history. They are
    loaded but **excluded from features by default**: each count is computed
    over the speaker's whole record including the statement being classified,
    so using them leaks the label. Reported LIAR numbers vary by roughly ten
    accuracy points depending on this choice alone.
    """
    root = Path(root or DATA_RAW) / "liar"
    splits = ["train", "valid", "test"] if split == "all" else [split]

    items = []
    for sp in splits:
        with open(root / f"{sp}.tsv", encoding="utf-8", newline="") as fh:
            # QUOTE_NONE is required: statements are full of unbalanced double
            # quotes, and the default dialect swallows 47 of the 12,836 rows by
            # treating them as field quoting.
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) < len(_LIAR_COLS):
                    continue
                rec = dict(zip(_LIAR_COLS, row))
                if rec["label"] not in LIAR_LABELS:
                    continue
                items.append(Item(
                    id=rec["id"], text=rec["statement"], label=rec["label"],
                    meta={"split": sp, "subject": rec["subject"],
                          "speaker": rec["speaker"], "job": rec["job"],
                          "state": rec["state"], "party": rec["party"],
                          "context": rec["context"],
                          "credit": [int(rec[c] or 0) for c in _LIAR_COLS[8:13]]},
                ))
    return DisinfoDataset("LIAR", items, LIAR_LABELS)


# --------------------------------------------------------------------------
# Twitter15 / Twitter16
# --------------------------------------------------------------------------

_TREE_LINE = re.compile(r"^(\[.*?\])->(\[.*?\])$")


def load_twitter(which: str = "twitter15", root: Path | None = None) -> DisinfoDataset:
    """Twitter15 / Twitter16 (Ma et al. 2017). Table 3: 1490 / 818 trees.

    Each tree file is a list of ``parent -> child`` edges, both given as
    ``['uid', 'tweet id', 'delay in minutes']``. The ``ROOT`` pseudo-node is
    dropped and its child becomes cascade node 0.

    Only the source tweet's text is distributed; the corpus README states the
    Twitter terms of service prevent redistributing the rest. So the reply
    nodes carry structure and timing but no text, which is why every Table 1-2
    method on these datasets is either source-text-only or structure-only.
    """
    base = Path(root or DATA_RAW) / "rumor_detection_acl2017" / which

    labels: dict[str, str] = {}
    with open(base / "label.txt", encoding="utf-8") as fh:
        for line in fh:
            if ":" in line:
                lab, tid = line.strip().split(":", 1)
                labels[tid] = lab

    texts: dict[str, str] = {}
    with open(base / "source_tweets.txt", encoding="utf-8") as fh:
        for line in fh:
            if "\t" in line:
                tid, txt = line.rstrip("\n").split("\t", 1)
                texts[tid] = txt

    items = []
    for tid, label in labels.items():
        tree = base / "tree" / f"{tid}.txt"
        if not tree.exists():
            continue
        cascade = _parse_twitter_tree(tree)
        items.append(Item(id=tid, text=texts.get(tid, ""),
                          label=label if label != "non-rumor" else "non-rumor",
                          meta={"source": which}, cascade=cascade))

    # label.txt uses 'true'/'false'/'unverified'/'non-rumor' already.
    return DisinfoDataset(which, items, CANONICAL_LABELS, canonical=True)


def _parse_twitter_tree(path: Path) -> Cascade:
    index: dict[str, int] = {}
    uids: list[str] = []
    times: list[float] = []
    edges: list[tuple[int, int]] = []

    def node(uid: str, tid: str, delay: str) -> int:
        if tid not in index:
            index[tid] = len(uids)
            uids.append(uid)
            times.append(float(delay))
        return index[tid]

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = _TREE_LINE.match(line.strip())
            if not m:
                continue
            parent = ast.literal_eval(m.group(1))
            child = ast.literal_eval(m.group(2))
            if parent[0] == "ROOT":
                node(child[0], child[1], child[2])       # becomes index 0
                continue
            p = node(parent[0], parent[1], parent[2])
            c = node(child[0], child[1], child[2])
            if p != c:
                edges.append((p, c))

    return Cascade(uids, np.asarray(times, dtype=np.float64), edges)


# --------------------------------------------------------------------------
# PHEME
# --------------------------------------------------------------------------

def _pheme_veracity(ann: dict) -> str | None:
    """The official ``convert_veracity_annotations.py`` from figshare, inlined.

    Reproduced rather than imported so the loader has no hidden dependency on a
    second download. Behaviour is identical, including returning None when
    ``misinformation`` and ``true`` are both 1.
    """
    has_mis, has_true = "misinformation" in ann, "true" in ann
    if has_mis and has_true:
        mis, tru = int(ann["misinformation"]), int(ann["true"])
        if mis == 0 and tru == 0:
            return "unverified"
        if mis == 0 and tru == 1:
            return "true"
        if mis == 1 and tru == 0:
            return "false"
        return None                      # both 1: contradictory annotation
    if has_mis and not has_true:
        return "unverified" if int(ann["misinformation"]) == 0 else "false"
    return None


def load_pheme(root: Path | None = None, rumours_only: bool = False) -> DisinfoDataset:
    """PHEME (Zubiaga et al. 2016). Table 3: 6425 threads, 4 labels.

    Threads live under ``{event}/rumours`` and ``{event}/non-rumours``.
    Non-rumours take the label ``non-rumor``; rumours are resolved by the
    official veracity converter above.

    ``meta['event']`` is retained because PHEME's nine breaking-news events are
    what make leave-one-event-out evaluation possible -- the setting that
    exposes how much of a reported accuracy is event memorisation. Table 1's
    PHEME numbers range from 0.694 to 0.887, a spread that split protocol
    explains more readily than architecture.
    """
    base = Path(root or DATA_RAW) / "pheme" / "all-rnr-annotated-threads"
    items = []

    for event_dir in sorted(base.iterdir()):
        if not event_dir.is_dir() or event_dir.name.startswith("."):
            continue
        event = event_dir.name.replace("-all-rnr-threads", "")
        for kind in ("rumours", "non-rumours"):
            kdir = event_dir / kind
            if not kdir.is_dir():
                continue
            for thread in sorted(kdir.iterdir()):
                if not thread.is_dir() or thread.name.startswith("."):
                    continue
                parsed = _parse_pheme_thread(thread)
                if parsed is None:
                    continue
                text, cascade = parsed

                if kind == "non-rumours":
                    label = "non-rumor"
                else:
                    ann_path = thread / "annotation.json"
                    if not ann_path.exists():
                        continue
                    label = _pheme_veracity(json.loads(ann_path.read_text()))
                    if label is None:
                        continue
                if rumours_only and label == "non-rumor":
                    continue
                items.append(Item(id=thread.name, text=text, label=label,
                                  meta={"event": event, "kind": kind},
                                  cascade=cascade))

    return DisinfoDataset("PHEME", items, CANONICAL_LABELS, canonical=True)


def _parse_pheme_thread(thread: Path):
    src_dir = thread / "source-tweets"
    if not src_dir.is_dir():
        return None
    src_files = [p for p in src_dir.glob("*.json")]
    if not src_files:
        return None
    src = json.loads(src_files[0].read_text())
    src_id = str(src["id"])

    tweets = {src_id: src}
    for p in (thread / "reactions").glob("*.json"):
        try:
            t = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        tweets[str(t["id"])] = t

    # structure.json gives the reply tree; fall back to in_reply_to_status_id
    # when it is missing or malformed, which happens in a handful of threads.
    edges_by_id: list[tuple[str, str]] = []
    struct_path = thread / "structure.json"
    if struct_path.exists():
        try:
            def walk(d, parent=None):
                for k, v in d.items():
                    if parent is not None:
                        edges_by_id.append((parent, k))
                    if isinstance(v, dict):
                        walk(v, k)
            walk(json.loads(struct_path.read_text()))
        except (json.JSONDecodeError, AttributeError):
            edges_by_id = []
    if not edges_by_id:
        for tid, t in tweets.items():
            p = t.get("in_reply_to_status_id")
            if p is not None and str(p) in tweets:
                edges_by_id.append((str(p), tid))

    t0 = _twitter_time(src.get("created_at"))
    order = [src_id] + sorted(tid for tid in tweets if tid != src_id)
    index = {tid: i for i, tid in enumerate(order)}

    uids, times, feats = [], [], []
    for tid in order:
        t = tweets[tid]
        user = t.get("user") or {}
        uids.append(str(user.get("id", "?")))
        ts = _twitter_time(t.get("created_at"))
        times.append(0.0 if (ts is None or t0 is None) else (ts - t0) / 60.0)
        feats.append(_twitter_user_features(user))

    edges = [(index[p], index[c]) for p, c in edges_by_id
             if p in index and c in index and index[p] != index[c]]

    cascade = Cascade(uids, np.asarray(times, dtype=np.float64), edges,
                      np.asarray(feats, dtype=np.float32))
    return src.get("text", ""), cascade


def _twitter_time(s):
    if not s:
        return None
    from datetime import datetime
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%a %b %d %H:%M:%S %Y"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except (ValueError, TypeError):
            continue
    return None


def _twitter_user_features(user: dict) -> list[float]:
    """The profile features of Sect. 4.2, in the order used throughout.

    Counts are log1p-compressed: follower counts span six orders of magnitude,
    and an untransformed count would dominate every distance in a similarity
    graph and every gradient in the first layer.
    """
    followers = float(user.get("followers_count") or 0)
    friends = float(user.get("friends_count") or 0)
    return [
        np.log1p(followers),
        np.log1p(friends),
        followers / (friends + 1.0),                       # follower/friend ratio
        np.log1p(float(user.get("statuses_count") or 0)),  # history tweets
        np.log1p(float(user.get("listed_count") or 0)),
        1.0 if user.get("verified") else 0.0,
        1.0 if user.get("description") else 0.0,
    ]


# --------------------------------------------------------------------------
# CED (Sina Weibo)
# --------------------------------------------------------------------------

def load_ced(root: Path | None = None) -> DisinfoDataset:
    """CED (Song et al.), the Chinese rumour corpus used by Table 2's Xu et al.

    Reused from ``data/raw/thunlp_rumor``, already fetched by
    ``scripts/get_data.sh`` for the entropia replication.

    Note this is **not** the "Sina Weibo" row of Table 3: that row cites Ma et
    al. (2016) and gives 4664 claims, while CED holds 3387. Table 2 lists Weibo
    and CED as separate datasets for Xu et al., consistent with them being
    different corpora.

    Reposts carry ``parent``/``mid`` message ids, giving the propagation tree,
    but their ``text`` field is empty in the release -- only the original
    microblog has text. User profile features come from the original post's
    author, so every node in a cascade shares them; this is a limitation of the
    corpus, not of the loader, and it makes CED a structure-only benchmark.
    """
    base = Path(root or DATA_RAW) / "thunlp_rumor" / "CED_Dataset"
    items = []

    for kind, label in (("rumor-repost", "false"), ("non-rumor-repost", "true")):
        for path in sorted((base / kind).glob("*.json")):
            orig_path = base / "original-microblog" / path.name
            if not orig_path.exists():
                continue
            try:
                reposts = json.loads(path.read_text(encoding="utf-8"))
                orig = json.loads(orig_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            cascade = _parse_ced_cascade(reposts, orig)
            items.append(Item(id=path.stem, text=orig.get("text", ""),
                              label=label,
                              meta={"kind": kind,
                                    "user": orig.get("user", {}),
                                    "pics": orig.get("pics", 0),
                                    "comments": orig.get("comments", 0)},
                              cascade=cascade))

    return DisinfoDataset("CED", items, ["true", "false"])


def _parse_ced_cascade(reposts: list[dict], orig: dict) -> Cascade:
    from datetime import datetime

    def parse_date(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
        except (ValueError, TypeError):
            return None

    # A few original-microblog records store `user` as a string rather than an
    # object; treat those as having no profile.
    user = orig.get("user")
    user = user if isinstance(user, dict) else {}

    # Node 0 is the source; reposts follow in file order.
    mids = ["__source__"]
    uids = [str(user.get("id", "source"))]
    stamps: list[float | None] = [None]

    for r in reposts:
        mids.append(str(r.get("mid", "")))
        uids.append(str(r.get("uid", "?")))
        stamps.append(parse_date(r.get("date")))

    index = {m: i for i, m in enumerate(mids)}
    known = [s for s in stamps if s is not None]
    t0 = min(known) if known else 0.0
    times = np.array([0.0 if s is None else (s - t0) / 60.0 for s in stamps])

    edges = []
    for i, r in enumerate(reposts, start=1):
        parent = str(r.get("parent") or "")
        p = index.get(parent, 0)       # missing/blank parent attaches to source
        if p != i:
            edges.append((p, i))

    feat = np.asarray([
        np.log1p(float(user.get("followers") or 0)),
        np.log1p(float(user.get("friends") or 0)),
        float(user.get("followers") or 0) / (float(user.get("friends") or 0) + 1.0),
        np.log1p(float(user.get("messages") or 0)),
        0.0,
        1.0 if user.get("verified") else 0.0,
        1.0 if user.get("description") else 0.0,
    ], dtype=np.float32)
    # Only the source author's profile is released, so it is broadcast to every
    # node. Documented in the loader docstring; do not read this as per-user data.
    feats = np.repeat(feat[None, :], len(mids), axis=0)

    return Cascade(uids, times, edges, feats)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_LOADERS = {
    "liar": load_liar,
    "twitter15": lambda **kw: load_twitter("twitter15", **kw),
    "twitter16": lambda **kw: load_twitter("twitter16", **kw),
    "pheme": load_pheme,
    "ced": load_ced,
}


def load_dataset(name: str, **kwargs) -> DisinfoDataset:
    key = name.lower().replace(" ", "")
    if key not in _LOADERS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(_LOADERS)}")
    return _LOADERS[key](**kwargs)


def dataset_sizes() -> dict[str, int]:
    """Item counts keyed by Table 3's dataset names, for ``verify_table3``."""
    out = {}
    for key, table3_name in (("liar", "LIAR"), ("twitter15", "Twitter15"),
                             ("twitter16", "Twitter16"), ("pheme", "PHEME")):
        try:
            out[table3_name] = len(load_dataset(key))
        except (FileNotFoundError, OSError):
            pass
    return out
