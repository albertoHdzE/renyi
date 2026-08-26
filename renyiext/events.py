"""P0 -- the data layer: snowflake decoding and per-account event series.

Cresci-2015, in the TwiBot-22 four-file schema, stores tweet nodes carrying only
``id`` and ``text``. ``botsage/text.py`` concludes from this that "the Cresci-15
release carries no tweet timestamps anyway".

It does. Twitter snowflake IDs, in use since 2010-11-04, pack

    | 41 bits milliseconds since epoch | 10 bits machine | 12 bits sequence |

so the posting time is recoverable by a shift and an add (decision D1)::

    timestamp_ms = (tweet_id >> 22) + 1_288_834_974_657

**This module does not assert that the decode is correct.** Asserting it is the
job of :mod:`renyiext.checks` and of the P0 gate, and the standard is
``docs/06-STANDING-RULES.md`` S1: counts and a date range are not agreement.
The discriminating evidence is elementwise and comes from an independent field
-- user nodes carry ``created_at``, and a user's tweets cannot predate their own
account. See :func:`account_age_violations`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import DATA_RAW, TWITTER_EPOCH_MS, FIRST_SNOWFLAKE_ID  # noqa: F401

__all__ = [
    "decode_snowflake", "decode_snowflake_array", "has_snowflake_form",
    "CresciEvents", "load_cresci_events", "account_age_violations",
    "counter_null_timestamps", "POST_TYPE", "classify_post_type",
    "save_events", "load_events_cached", "load_cresci_text_side",
    "load_tb20_text_side",
]

# Twitter's own ``created_at`` format, as it appears in the user nodes.
_CREATED_FMT = "%a %b %d %H:%M:%S %z %Y"

# The four-symbol behavioural alphabet (decision D5).
POST_TYPE = {"original": 0, "reply": 1, "retweet": 2, "quote": 3}

_RETWEET = re.compile(r"^\s*RT\s+@\w+")
_REPLY = re.compile(r"^\s*@\w+")
# A quote tweet carries a link back to another status. In this corpus the
# t.co-shortened form is all that survives, so the marker is a trailing URL on
# a tweet that is not already a retweet.
_STATUS_URL = re.compile(r"https?://t\.co/\w+\s*$")


# ---------------------------------------------------------------------------
# Snowflake decoding
# ---------------------------------------------------------------------------

def has_snowflake_form(tweet_id: int) -> bool:
    """Whether ``tweet_id`` carries a recoverable timestamp (decision D9).

    Snowflakes were introduced 2010-11-04. Ids issued before that are
    *sequential* and carry no time information at all -- but they are large
    numbers (~3e10), so they survive any naive lower bound such as ``2**22``
    and decode to a few milliseconds past the epoch. In Cresci-2015 that put
    **63,830 tweets on a single millisecond**, which the G1 render showed as a
    spike at day 0 and an artefactual peak at hour 1 (the epoch is 01:42:54Z).
    G2 could not see it: those fake timestamps are still later than the 2007-9
    account-creation dates they were checked against.

    The threshold is the first snowflake id, not a bit width.
    """
    return int(tweet_id) >= FIRST_SNOWFLAKE_ID


def decode_snowflake(tweet_id: int | str) -> datetime:
    """Posting time of one tweet id, in UTC (decision D1).

    ``tweet_id`` may carry this corpus's ``t`` prefix.
    """
    n = int(str(tweet_id).lstrip("t"))
    ms = (n >> 22) + TWITTER_EPOCH_MS
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def decode_snowflake_array(ids: np.ndarray) -> np.ndarray:
    """Vectorised decode. Returns milliseconds since the Unix epoch, int64.

    Kept in integer milliseconds rather than ``datetime64`` because the
    downstream object is an inter-arrival series, and float seconds lose
    resolution on bursty accounts -- which is exactly the population whose
    burstiness we are trying to measure.
    """
    ids = np.asarray(ids, dtype=np.int64)
    return (ids >> 22) + TWITTER_EPOCH_MS


def counter_null_timestamps(ids: np.ndarray) -> np.ndarray:
    """G4 control: what the decode would look like if ids were a **counter**.

    The gate-4 question for decision D1 is *what would this number be under a
    process known to contain nothing?* -- here, if the top 41 bits were not a
    clock but a monotone counter incremented once per tweet globally.

    Under that null, decoded "time" is an affine function of the rank of the id,
    so this returns ranks rescaled onto the same span as the real decode. Any
    structure that survives in the real decode but not here (circadian rhythm,
    inter-arrival heavy tails) is evidence the top bits are a clock.
    """
    ids = np.asarray(ids, dtype=np.int64)
    real = decode_snowflake_array(ids)
    order = np.argsort(np.argsort(ids))
    lo, hi = real.min(), real.max()
    if len(ids) < 2:
        return real.copy()
    return (lo + (order / (len(ids) - 1)) * (hi - lo)).astype(np.int64)


# ---------------------------------------------------------------------------
# Post type
# ---------------------------------------------------------------------------

def classify_post_type(text: str) -> int:
    """Map a tweet to the four-symbol alphabet (decision D5).

    Cresci-2015's conversion carries no explicit type field -- open item 2 of
    bitacora 00 -- so the type is inferred from surface markers, in this order:

    1. ``RT @user`` prefix              -> retweet
    2. leading ``@user``                -> reply
    3. trailing t.co link, not a RT     -> quote
    4. otherwise                        -> original

    The order matters: a retweet of a reply begins ``RT @a: @b ...`` and must
    read as a retweet. This is a lossy inference and is recorded as such; the
    proportion falling into each class is reported by the P0 gate so the reader
    can judge it.
    """
    if not text:
        return POST_TYPE["original"]
    if _RETWEET.match(text):
        return POST_TYPE["retweet"]
    if _REPLY.match(text):
        return POST_TYPE["reply"]
    if _STATUS_URL.search(text):
        return POST_TYPE["quote"]
    return POST_TYPE["original"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class CresciEvents:
    """Per-account event series for Cresci-2015.

    ``user_ids[i]`` owns the events at ``offsets[i]:offsets[i+1]`` of the
    flat ``ts_ms`` / ``post_type`` arrays, sorted ascending in time.
    """

    user_ids: np.ndarray          # (n_users,) str
    labels: np.ndarray            # (n_users,) int, 1 = bot
    created_ms: np.ndarray        # (n_users,) int64, account creation
    offsets: np.ndarray           # (n_users + 1,) int64
    ts_ms: np.ndarray             # (n_events,) int64, sorted within account
    post_type: np.ndarray         # (n_events,) int8
    n_dropped_nonsnowflake: int = 0
    # Pre-snowflake tweets discarded per account (decision D9). This exclusion
    # is class-dependent -- older accounts lose more -- so it is carried per
    # user rather than as a single total, and reported with every result.
    dropped_per_user: np.ndarray | None = None

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_events(self) -> int:
        return len(self.ts_ms)

    def counts(self) -> np.ndarray:
        """Events per account."""
        return np.diff(self.offsets)

    def events_of(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """``(timestamps_ms, post_types)`` for account ``i``."""
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.ts_ms[a:b], self.post_type[a:b]

    def inter_arrival_ms(self, i: int) -> np.ndarray:
        """Inter-arrival times for account ``i``, in milliseconds.

        Zero-length gaps are kept, not clipped: simultaneous posting is a real
        and highly informative behaviour for automated accounts, and clipping
        it would remove the signal before it is measured.
        """
        ts, _ = self.events_of(i)
        return np.diff(ts) if len(ts) > 1 else np.empty(0, dtype=np.int64)

    def __repr__(self) -> str:
        c = self.counts()
        return (f"CresciEvents(users={self.n_users}, events={self.n_events}, "
                f"bot={int(self.labels.sum())}, "
                f"events/user median={int(np.median(c))})")


def load_cresci_events(root: Path | None = None) -> CresciEvents:
    """Build the per-account event series from the Cresci-2015 conversion.

    Reads ``node.json`` (users and tweets), ``edge.csv`` (the ``post`` relation
    linking them) and ``label.csv``.
    """
    base = Path(root) if root else DATA_RAW / "bot" / "cresci-2015"

    with open(base / "node.json") as fh:
        nodes = json.load(fh)

    users: dict[str, dict] = {}
    tweet_text: dict[str, str] = {}
    for obj in nodes:
        nid = obj.get("id", "")
        if nid.startswith("u"):
            users[nid] = obj
        elif nid.startswith("t"):
            tweet_text[nid] = obj.get("text") or ""
    del nodes

    labels_raw: dict[str, int] = {}
    with open(base / "label.csv") as fh:
        next(fh)
        for line in fh:
            uid, lab = line.rstrip("\n").split(",")[:2]
            labels_raw[uid] = 1 if lab.strip() == "bot" else 0

    # The post relation: source user -> target tweet.
    posts: dict[str, list[str]] = {}
    with open(base / "edge.csv") as fh:
        header = next(fh).rstrip("\n").split(",")
        i_src, i_rel, i_tgt = (header.index("source_id"), header.index("relation"),
                               header.index("target_id"))
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) <= i_tgt or p[i_rel] != "post":
                continue
            posts.setdefault(p[i_src], []).append(p[i_tgt])

    uids = sorted(u for u in users if u in labels_raw)

    user_ids, labels, created, offs = [], [], [], [0]
    ts_all, type_all, dropped = [], [], []
    n_dropped = 0

    for uid in uids:
        tids = posts.get(uid, [])
        raw = []
        n_drop_u = 0
        for tid in tids:
            n = int(tid.lstrip("t"))
            if not has_snowflake_form(n):
                n_dropped += 1
                n_drop_u += 1
                continue
            raw.append((n, tid))
        dropped.append(n_drop_u)
        if not raw:
            # Keep the account with zero events: the exclusion is a bias and
            # must be visible downstream, not silently dropped here.
            user_ids.append(uid)
            labels.append(labels_raw[uid])
            created.append(_created_ms(users[uid]))
            offs.append(offs[-1])
            continue

        ids = np.array([r[0] for r in raw], dtype=np.int64)
        ts = decode_snowflake_array(ids)
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        types = np.array(
            [classify_post_type(tweet_text.get(raw[j][1], "")) for j in order],
            dtype=np.int8,
        )

        user_ids.append(uid)
        labels.append(labels_raw[uid])
        created.append(_created_ms(users[uid]))
        ts_all.append(ts)
        type_all.append(types)
        offs.append(offs[-1] + len(ts))

    return CresciEvents(
        user_ids=np.array(user_ids),
        labels=np.array(labels, dtype=np.int8),
        created_ms=np.array(created, dtype=np.int64),
        offsets=np.array(offs, dtype=np.int64),
        ts_ms=(np.concatenate(ts_all) if ts_all else np.empty(0, np.int64)),
        post_type=(np.concatenate(type_all) if type_all else np.empty(0, np.int8)),
        n_dropped_nonsnowflake=n_dropped,
        dropped_per_user=np.array(dropped, dtype=np.int64),
    )


def _created_ms(user: dict) -> int:
    raw = user.get("created_at")
    if not raw:
        return -1
    try:
        return int(datetime.strptime(raw, _CREATED_FMT).timestamp() * 1000)
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# The gate-2 check for decision D1
# ---------------------------------------------------------------------------

def account_age_violations(ev: CresciEvents, tol_ms: int = 0) -> dict:
    """Elementwise test of the snowflake decode against an independent field.

    Gate G2 of the datasaurus rule: a claim of agreement is settled by comparing
    **elements in a common coordinate**, not by counts and a plausible range.

    User nodes carry ``created_at``, which never passed through the decoder. A
    tweet cannot predate the account that posted it, so every decoded timestamp
    must satisfy ``ts >= created``. That is a per-event, independently sourced
    constraint on 2.8M elements, and it is what licenses D1 -- or refutes it.

    Returns the violation count, the fraction, *where the violations live*
    (per account, and how far back), and the same statistic under the counter
    null for comparison.
    """
    viol_per_user, worst_ms, n_checked = [], [], 0
    for i in range(ev.n_users):
        c = ev.created_ms[i]
        ts, _ = ev.events_of(i)
        if c < 0 or len(ts) == 0:
            viol_per_user.append(0)
            worst_ms.append(0)
            continue
        bad = ts < (c - tol_ms)
        n_checked += len(ts)
        viol_per_user.append(int(bad.sum()))
        worst_ms.append(int((c - ts[bad]).max()) if bad.any() else 0)

    viol = np.array(viol_per_user)
    return {
        "n_events_checked": int(n_checked),
        "n_violations": int(viol.sum()),
        "frac_violations": float(viol.sum() / max(n_checked, 1)),
        "n_users_with_any_violation": int((viol > 0).sum()),
        "n_users_checked": int((ev.counts() > 0).sum()),
        "worst_backdate_days": float(max(worst_ms) / 86_400_000.0),
        "violations_per_user_max": int(viol.max()) if len(viol) else 0,
    }


# ---------------------------------------------------------------------------
# Caching -- loading node.json costs ~2 min and P2 sweeps the binning grid
# ---------------------------------------------------------------------------

def save_events(ev: CresciEvents, path: Path) -> None:
    """Persist an event series. Arrays only; no pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        user_ids=ev.user_ids, labels=ev.labels, created_ms=ev.created_ms,
        offsets=ev.offsets, ts_ms=ev.ts_ms, post_type=ev.post_type,
        dropped_per_user=ev.dropped_per_user,
        n_dropped_nonsnowflake=np.array([ev.n_dropped_nonsnowflake]),
    )


def load_events_cached(path: Path, root: Path | None = None) -> CresciEvents:
    """Load from ``path`` if present, else build and write it.

    The cache key is the file name only. Anything that changes the decode --
    ``FIRST_SNOWFLAKE_ID`` (D9), the post-type rules (D5) -- must be paired with
    a new file name, or a stale cache will silently outlive the decision that
    invalidated it.
    """
    path = Path(path)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return CresciEvents(
            user_ids=z["user_ids"], labels=z["labels"],
            created_ms=z["created_ms"], offsets=z["offsets"],
            ts_ms=z["ts_ms"], post_type=z["post_type"],
            n_dropped_nonsnowflake=int(z["n_dropped_nonsnowflake"][0]),
            dropped_per_user=z["dropped_per_user"],
        )
    ev = load_cresci_events(root)
    save_events(ev, path)
    return ev


def load_cresci_text_side(ev: CresciEvents, kept: np.ndarray,
                          age_reference_ms: int) -> tuple[np.ndarray, dict, list]:
    """Text-side corpus read for the behavioural/text fronts: per-account
    ``(post_type, text)`` sequences, META-lite rows and usernames for the
    kept accounts.

    Reads the same ``edge.csv``/``node.json`` sources as
    :func:`load_cresci_events` and rebuilds its exact per-account ordering
    (decode, stable time sort) so text aligns with the cached ``post_type``
    slice -- asserted elementwise per account. META-lite is botsage's
    read-only recipe (followers, following, tweet_count, age vs
    ``age_reference_ms``; ``listed_count`` stays dropped per the WP-B
    alignment).

    Provenance: introduced by WP-I, generalising the script-local loader the
    WP-H producer carries; that copy is left untouched (its phase is closed
    and its artefact predates this function).

    Returns ``(meta, username, seq)`` with rows ordered as ``ev.user_ids``
    restricted to ``kept``.
    """
    base = DATA_RAW / "bot" / "cresci-2015"
    kept_ids = set(ev.user_ids[kept])

    with open(base / "edge.csv") as fh:
        header = next(fh).rstrip("\n").split(",")
        i_src, i_rel, i_tgt = (header.index("source_id"),
                               header.index("relation"),
                               header.index("target_id"))
        posts: dict[str, list[str]] = {}
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) <= i_tgt or p[i_rel] != "post":
                continue
            posts.setdefault(p[i_src], []).append(p[i_tgt])

    with open(base / "node.json") as fh:
        nodes = json.load(fh)
    username: dict[str, str] = {}
    pm: dict[str, tuple[float, float, float]] = {}
    born: dict[str, int | None] = {}
    texts: dict[str, str] = {}
    for obj in nodes:
        oid = obj.get("id", "")
        if oid.startswith("u") and oid in kept_ids:
            u = obj.get("public_metrics") or {}
            username[oid] = obj.get("username") or ""
            pm[oid] = (float(u.get("followers_count") or 0),
                       float(u.get("following_count") or 0),
                       float(u.get("tweet_count") or 0))
            try:
                born[oid] = int(datetime.strptime(
                    obj.get("created_at"), _CREATED_FMT).timestamp() * 1000)
            except (TypeError, ValueError):
                born[oid] = None
        elif oid.startswith("t"):
            texts[oid] = obj.get("text") or ""
    del nodes

    meta_rows, seq = [], []
    for i in np.where(kept)[0]:
        uid = str(ev.user_ids[i])
        f, g, tc = pm[uid]
        age = ((age_reference_ms - born[uid]) / 86_400_000.0
               if born[uid] is not None else np.nan)
        meta_rows.append([f, g, tc, age])
        raw = [tid for tid in posts.get(uid, [])
               if has_snowflake_form(int(tid.lstrip("t")))]
        ids = np.array([int(tid.lstrip("t")) for tid in raw], dtype=np.int64)
        ts = decode_snowflake_array(ids)
        order = np.argsort(ts, kind="stable")
        a, b = int(ev.offsets[i]), int(ev.offsets[i + 1])
        # ts[order] == the cache's time-sorted slice proves raw[order] is
        # exactly the loader's id order (same stable sort on the same ids)
        assert np.array_equal(ts[order], ev.ts_ms[a:b]), uid
        seq.append([(int(ev.post_type[a + k]), texts[raw[order[k]]])
                    for k in order])
    del texts
    return (np.array(meta_rows, dtype=np.float64), username, seq)


def load_tb20_text_side(age_reference_ms: int):
    """Raw TwiBot-20 labelled release (bitacoras 22-23): per-user texts,
    META_raw rows and labels for the labelled population.

    Reads ``data/raw/bot/twibot-20-raw/{train,dev,test}.json`` -- the
    release schema (ID/profile/tweet/neighbor/domain/label), texts kept RAW
    (D4). Rows are ordered train -> dev -> test (ids unique across files,
    asserted). META rows follow the aligned four-field recipe
    ``[followers_count, following_count(friends), statuses_count, age_days
    vs age_reference_ms]``; profile values arrive whitespace-padded and are
    stripped before parsing.

    Returns ``(meta, labels, texts)``: meta float64 (n, 4) with NaN age
    where created_at is unparseable, labels int8 (n,), texts list of n
    lists of raw strings (empty where the user ships no tweet field).
    """
    base = DATA_RAW / "bot" / "twibot-20-raw"
    meta_rows, labels, texts = [], [], []
    seen = set()
    for split in ("train", "dev", "test"):
        with open(base / f"{split}.json") as fh:
            users = json.load(fh)
        for u in users:
            uid = str(u["ID"]).strip()
            assert uid not in seen, f"duplicate id across splits: {uid}"
            seen.add(uid)
            p = u.get("profile") or {}
            try:
                followers = float(p.get("followers_count"))
            except (TypeError, ValueError):
                followers = 0.0
            try:
                friends = float(p.get("friends_count"))
            except (TypeError, ValueError):
                friends = 0.0
            try:
                statuses = float(p.get("statuses_count"))
            except (TypeError, ValueError):
                statuses = 0.0
            created = p.get("created_at")
            try:
                born = int(datetime.strptime(created.strip(), _CREATED_FMT)
                           .timestamp() * 1000)
            except (AttributeError, TypeError, ValueError):
                born = None
            age = ((age_reference_ms - born) / 86_400_000.0
                   if born is not None else np.nan)
            meta_rows.append([followers, friends, statuses, age])
            labels.append(int(u.get("label")))
            t = u.get("tweet")
            texts.append(list(t) if isinstance(t, list) else [])
    return (np.array(meta_rows, dtype=np.float64),
            np.array(labels, dtype=np.int8), texts)
