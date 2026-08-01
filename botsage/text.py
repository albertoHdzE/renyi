"""The text branch: tweet cleaning (Sect. 3.1.2) and BERT embeddings (Sect. 3.2).

The paper's appendix (Listing A.1) pins this stage down more precisely than any
other, so it is reproduced literally:

* tokenizer with ``max_length=50``, ``truncation=True``, ``padding=True``,
  ``add_special_tokens=True``;
* the **mean over token vectors** of the last hidden state gives one 768-vector
  per tweet -- not the ``[CLS]`` vector, which is the more usual choice;
* the **mean over a user's tweets** gives one 768-vector per user;
* a user with no tweets gets ``zeros(768)``.

Listing A.1 also sorts by timestamp and keeps the 15 most recent tweets. That is
for TwiBot-22 only; Sect. 3.1.2 says "For Cresci, all tweets of each user are
considered", and the Cresci-15 release carries no tweet timestamps anyway, so
``max_tweets=None`` there is both faithful and forced.

One consequence worth stating plainly: averaging token vectors and then
averaging over hundreds of tweets is a lot of averaging. A user's 768-vector
ends up close to the corpus mean, and the *between-user* variance that a
classifier needs is small. ``embedding_diagnostics`` measures exactly that, so
the notebook can show it rather than assert it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

__all__ = ["clean_tweet", "STOPWORDS", "TweetEncoder", "user_text_embeddings",
           "embedding_diagnostics"]


# A small English stop-word list. Sect. 3.1.2 says "common words (e.g. 'and',
# 'the')" without naming a source; this is NLTK's English list, which is the
# usual choice in this setting and is reproduced inline to avoid an extra
# download dependency.
STOPWORDS = frozenset("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll
these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down in
out on off over under again further then once here there when where why how all
any both each few more most other some such no nor not only own same so than too
very s t can will just don don't should should've now d ll m o re ve y ain aren
aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven
haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't
shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split())

_URL = re.compile(r"http\S+|www\.\S+")
_MENTION_HASHTAG = re.compile(r"[@#]\w+")
_NON_ASCII = re.compile(r"[^\x00-\x7F]+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")


def clean_tweet(text: str, remove_stopwords: bool = True) -> str:
    """Sect. 3.1.2, in the order the paper lists the steps.

    Lowercase; strip URLs, hashtags, mentions; drop non-ASCII (emoji and
    symbols); remove numbers and remaining special characters; remove stop
    words; collapse whitespace.

    Order matters: URLs must go before non-alphabetic stripping, or ``http``
    and the domain fragments survive as tokens and become some of the most
    frequent "words" in the corpus.
    """
    if not text:
        return ""
    t = text.lower()
    t = _URL.sub(" ", t)
    t = _MENTION_HASHTAG.sub(" ", t)
    t = _NON_ASCII.sub(" ", t)
    t = _NON_ALPHA.sub(" ", t)          # removes numbers and punctuation
    if remove_stopwords:
        t = " ".join(w for w in t.split() if w not in STOPWORDS)
    return _WS.sub(" ", t).strip()


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TweetEncoder:
    """DistilBERT/BERT feature extraction, matching Listing A.1.

    ``pooling="mean"`` reproduces the appendix (mean over token vectors).
    ``"cls"`` is offered for the notebook's comparison, since the choice is
    rarely stated and materially changes the embedding.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 max_length: int = 50, device: str | None = None,
                 pooling: str = "mean", batch_size: int = 256):
        from transformers import AutoModel, AutoTokenizer

        self.model_name, self.max_length = model_name, max_length
        self.pooling, self.batch_size = pooling, batch_size
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode_batch(self, batch: list[str]) -> torch.Tensor:
        """One vector per string, for a single batch."""
        enc = self.tokenizer(batch, padding=True, truncation=True,
                             max_length=self.max_length,
                             add_special_tokens=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).last_hidden_state           # (B, T, 768)
        if self.pooling == "cls":
            vec = hidden[:, 0]
        else:
            # Mean over *real* tokens only. Listing A.1 averages over the padded
            # axis, but it encodes one tweet at a time, so no padding exists
            # there. Batching without masking would make a tweet's embedding
            # depend on the longest tweet in its batch -- a batching artefact,
            # not a property of the text.
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return vec.float().cpu()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """One vector per input string.

        Materialises every vector, so it is for small inputs only -- Cresci-15
        has 2.7 million tweets, which at 768 float32 is 8.4 GB. Use
        ``user_text_embeddings``, which never holds more than one batch.
        """
        if not texts:
            return torch.zeros(0, self.dim)
        return torch.cat([self.encode_batch(texts[i:i + self.batch_size])
                          for i in range(0, len(texts), self.batch_size)])


def user_text_embeddings(tweets: list[list[str]], encoder: TweetEncoder,
                         max_tweets: int | None = None,
                         clean: bool = True, cache: Path | None = None,
                         quiet: bool = True) -> np.ndarray:
    """Sect. 3.2: mean over a user's tweet embeddings; zeros when there are none.

    Tweets from all users are encoded in one flat stream so the GPU batches stay
    full -- Listing A.1's per-tweet calls are most of why the paper reports 100+
    hours in Table 1 -- but each batch is **accumulated straight into the
    per-user running sum** and then discarded.

    That matters at this scale rather than being mere tidiness: Cresci-15 has
    2.7 million tweets, so holding every tweet vector before averaging costs
    2.7e6 x 768 x 4 B = **8.4 GB**, and TwiBot-22's 80 million tweets would be
    250 GB. Streaming keeps the peak at one batch.
    """
    if cache is not None and Path(cache).exists():
        return np.load(cache)

    flat: list[str] = []
    owner: list[int] = []
    for u, tw in enumerate(tweets):
        picked = tw[:max_tweets] if max_tweets else tw
        for t in picked:
            c = clean_tweet(t) if clean else t
            if c:
                flat.append(c)
                owner.append(u)

    n_users, dim = len(tweets), encoder.dim
    if not quiet:
        print(f"  encoding {len(flat):,} cleaned tweets from {n_users:,} "
              f"users on {encoder.device}", flush=True)

    out = torch.zeros(n_users, dim)
    counts = torch.zeros(n_users, 1)
    owner_t = torch.as_tensor(owner, dtype=torch.long)
    bs = encoder.batch_size
    n_batches = (len(flat) + bs - 1) // bs

    import time
    t0 = time.time()
    for b in range(n_batches):
        lo, hi = b * bs, min((b + 1) * bs, len(flat))
        vec = encoder.encode_batch(flat[lo:hi])
        idx = owner_t[lo:hi]
        out.index_add_(0, idx, vec)
        counts.index_add_(0, idx, torch.ones(hi - lo, 1))
        if not quiet and (b % 200 == 0 or b == n_batches - 1):
            done = hi / max(len(flat), 1)
            rate = hi / max(time.time() - t0, 1e-9)
            eta = (len(flat) - hi) / max(rate, 1e-9) / 60
            print(f"    {done:5.1%}  {hi:>9,}/{len(flat):,} tweets  "
                  f"{rate:6.0f}/s  eta {eta:5.1f} min", flush=True)

    out = out / counts.clamp(min=1)        # users with no tweets stay at zeros

    arr = out.numpy().astype(np.float32)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
    return arr


def embedding_diagnostics(emb: np.ndarray, labels: np.ndarray | None = None) -> dict:
    """How much signal survives the double averaging of Sect. 3.2.

    ``between_over_within`` is the ratio of between-class to within-class
    standard deviation, averaged over dimensions -- a direct read on whether the
    768 text dimensions separate bots from humans at all.
    """
    out = {
        "shape": tuple(emb.shape),
        "zero_rows": int((np.abs(emb).sum(1) == 0).sum()),
        "mean_norm": float(np.linalg.norm(emb, axis=1).mean()),
        "dim_std_mean": float(emb.std(axis=0).mean()),
    }
    centred = emb - emb.mean(0, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    out["centred_norm_mean"] = float(norms.mean())
    # Effective dimensionality: how many PCs hold 95% of the variance.
    if emb.shape[0] > 2:
        s = np.linalg.svd(centred, compute_uv=False)
        var = s ** 2
        out["pcs_for_95pct_variance"] = int(
            np.searchsorted(np.cumsum(var) / var.sum(), 0.95) + 1)

    if labels is not None:
        m = labels >= 0
        y, e = labels[m], emb[m]
        if len(np.unique(y)) == 2:
            a, b = e[y == 0], e[y == 1]
            between = np.abs(a.mean(0) - b.mean(0))
            within = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-12
            out["between_over_within"] = float((between / within).mean())
    return out
