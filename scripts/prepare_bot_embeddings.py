#!/usr/bin/env python
"""Precompute the per-user BERT / DistilBERT embeddings of Sect. 3.2.

This is the expensive stage -- the paper spends Tables 1-3 on how expensive --
so it is a separate script and its output is cached under ``data/processed/bot``.

    P=01-info-propagation/bot-detection-paper/.venv/bin/python
    $P scripts/prepare_bot_embeddings.py --model distilbert-base-uncased
    $P scripts/prepare_bot_embeddings.py --model bert-base-uncased

The paper runs both on Cresci-15 (Sect. 2.4.2), which is why both are needed to
reproduce its Table 4.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from botsage.data import load_dataset  # noqa: E402
from botsage.text import (TweetEncoder, embedding_diagnostics,  # noqa: E402
                          user_text_embeddings)

CACHE = Path(__file__).resolve().parent.parent / "data" / "processed" / "bot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cresci-15")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max-tweets", type=int, default=None,
                    help="Sect. 2.4.1 uses 15 for Twibot-22; Cresci uses all")
    ap.add_argument("--max-length", type=int, default=50)   # Listing A.1
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--pooling", default="mean")            # Listing A.1
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    if args.max_tweets:
        tag += f"_top{args.max_tweets}"
    out = CACHE / f"{args.dataset}_{tag}_{args.pooling}.npy"
    # Only labelled users are ever classified, and on Cresci-15 they are 5,301
    # of 1.29M nodes -- so the full matrix is 3.7 GB of mostly zeros. The
    # compact file is what everything downstream loads.
    compact = out.with_name(out.stem + "_labelled.npy")

    if compact.exists():
        print(f"already cached -> {compact}")
        return

    print(f"loading {args.dataset}")
    ds = load_dataset(args.dataset)
    if ds.tweets is None:
        raise SystemExit(f"{ds.name} has no raw tweets to encode")
    n_tweets = sum(len(t) for t in ds.tweets)
    print(f"  {len(ds):,} users, {n_tweets:,} tweets")

    enc = TweetEncoder(args.model, max_length=args.max_length,
                       device=args.device, pooling=args.pooling,
                       batch_size=args.batch_size)
    print(f"  encoder {args.model} on {enc.device}, dim {enc.dim}")

    t0 = time.time()
    emb = user_text_embeddings(ds.tweets, enc, max_tweets=args.max_tweets,
                               cache=out, quiet=args.quiet)
    dt = time.time() - t0
    print(f"  encoded in {dt / 60:.1f} min "
          f"({n_tweets / max(dt, 1):.0f} tweets/s)")

    lab = ds.labelled
    np.save(compact, emb[lab])
    out.unlink(missing_ok=True)
    print(f"  {len(lab):,} labelled rows -> {compact}")

    diag = embedding_diagnostics(emb[lab], ds.labels[lab])
    for k, v in diag.items():
        print(f"    {k:26s} {v}")


if __name__ == "__main__":
    main()
