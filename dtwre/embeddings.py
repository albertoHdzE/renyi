"""Node2Vec embeddings (Section 3.2.2).

Second-order biased random walks (Grover & Leskovec) followed by Skip-Gram
with negative sampling, implemented directly in PyTorch so the notebook has no
gensim/DGL dependency and every step stays inspectable.

The paper fixes only the embedding dimension (64) and states that the graph is
converted to an undirected NetworkX graph first; walk length, walk count,
context window and the return/in-out parameters p, q are not reported and use
the reference defaults from the Node2Vec paper.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

__all__ = ["build_csr", "random_walks", "skipgram_embeddings", "node2vec"]


def build_csr(graph: nx.Graph, num_nodes: int):
    """Compressed adjacency (indptr, indices) over a fixed node range."""
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    neigh = [[] for _ in range(num_nodes)]
    for u, v in graph.edges():
        if u < num_nodes and v < num_nodes:
            neigh[u].append(v)
            neigh[v].append(u)
    for i in range(num_nodes):
        indptr[i + 1] = indptr[i] + len(neigh[i])
    indices = np.fromiter((x for lst in neigh for x in lst),
                          dtype=np.int64, count=int(indptr[-1]))
    return indptr, indices


def random_walks(indptr, indices, num_nodes, num_walks=10, walk_length=80,
                 p=1.0, q=1.0, rng=None) -> np.ndarray:
    """Generate ``num_walks`` biased walks of length ``walk_length`` per node.

    With ``p == q == 1`` the walk is a plain uniform random walk (DeepWalk);
    otherwise each step reweights neighbours by 1/p (return to previous),
    1 (stay in the 1-hop neighbourhood) or 1/q (move outward).
    """
    rng = rng or np.random.default_rng(0)
    uniform = (abs(p - 1.0) < 1e-12) and (abs(q - 1.0) < 1e-12)

    starts = np.repeat(np.arange(num_nodes, dtype=np.int64), num_walks)
    rng.shuffle(starts)
    walks = np.full((len(starts), walk_length), -1, dtype=np.int64)
    walks[:, 0] = starts

    if uniform:
        # p == q == 1 reduces to a first-order uniform walk, which steps every
        # walk forward simultaneously instead of looping in Python. Same
        # distribution, orders of magnitude faster on large graphs.
        deg_all = np.diff(indptr)
        cur = starts.copy()
        alive = deg_all[cur] > 0
        for step in range(1, walk_length):
            if not np.any(alive):
                break
            d = np.where(deg_all[cur] > 0, deg_all[cur], 1)
            offset = (rng.random(len(cur)) * d).astype(np.int64)
            # Dead-end nodes have no CSR slot; their index would run past the
            # end of `indices`. Clamp, then discard via the `alive` mask.
            idx = np.minimum(indptr[cur] + offset, len(indices) - 1)
            nxt = indices[idx]
            walks[alive, step] = nxt[alive]
            cur = np.where(alive, nxt, cur)
            alive &= deg_all[cur] > 0
        return walks

    for i, s in enumerate(starts):
        cur = s
        prev = -1
        for step in range(1, walk_length):
            lo, hi = indptr[cur], indptr[cur + 1]
            if hi <= lo:
                break                       # dead end: truncate the walk
            nbrs = indices[lo:hi]
            if uniform or prev < 0:
                nxt = nbrs[rng.integers(0, len(nbrs))]
            else:
                plo, phi = indptr[prev], indptr[prev + 1]
                prev_nbrs = indices[plo:phi]
                w = np.ones(len(nbrs), dtype=np.float64)
                w[nbrs == prev] = 1.0 / p
                mask = np.isin(nbrs, prev_nbrs, assume_unique=False)
                w[mask & (nbrs != prev)] = 1.0
                w[~mask & (nbrs != prev)] = 1.0 / q
                w /= w.sum()
                nxt = nbrs[rng.choice(len(nbrs), p=w)]
            walks[i, step] = nxt
            prev, cur = cur, nxt
    return walks


def _skipgram_pairs(walks: np.ndarray, window: int, rng,
                    max_pairs: int | None = 20_000_000) -> np.ndarray:
    """Centre/context pairs from walks, skipping the -1 truncation padding.

    Vectorised over the whole walk matrix: for every offset ``d <= window``
    the pairs are the columns shifted by ``d`` against themselves, kept in
    both directions. Equivalent to the nested-loop formulation but ~2 orders
    of magnitude faster, which matters because the walk matrix yields tens of
    millions of pairs.

    ``max_pairs`` caps memory by uniformly subsampling. The pair count grows as
    ``num_nodes * num_walks * walk_length * window``, so on a 110k-node graph
    the uncapped set exceeds 3e9 rows and is OOM-killed. Uniform subsampling
    leaves the Skip-Gram objective unbiased -- it just sees fewer samples.
    """
    if walks.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    chunks = []
    for d in range(1, window + 1):
        if d >= walks.shape[1]:
            break
        a = walks[:, :-d].ravel()
        b = walks[:, d:].ravel()
        keep = (a >= 0) & (b >= 0)
        if not np.any(keep):
            continue
        a, b = a[keep], b[keep]
        chunks.append(np.stack([a, b], axis=1))
        chunks.append(np.stack([b, a], axis=1))

    if not chunks:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.concatenate(chunks, axis=0).astype(np.int64)

    if max_pairs is not None and len(pairs) > max_pairs:
        keep = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = pairs[keep]
    return pairs


def skipgram_embeddings(pairs, num_nodes, dim=64, epochs=5, lr=0.025,
                        negative=5, batch_size=8192, device="cpu", seed=0):
    """Train Skip-Gram with negative sampling on (centre, context) pairs."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)

    emb_in = nn.Embedding(num_nodes, dim)
    emb_out = nn.Embedding(num_nodes, dim)
    nn.init.uniform_(emb_in.weight, -0.5 / dim, 0.5 / dim)
    nn.init.zeros_(emb_out.weight)
    emb_in, emb_out = emb_in.to(device), emb_out.to(device)

    if len(pairs) == 0:
        return emb_in.weight.detach().cpu().numpy()

    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()),
                           lr=lr)
    pairs_t = torch.as_tensor(pairs, dtype=torch.long)

    # Unigram^0.75 noise distribution, as in the original Skip-Gram.
    counts = np.bincount(pairs[:, 1], minlength=num_nodes).astype(np.float64)
    noise = torch.as_tensor((counts ** 0.75) / max((counts ** 0.75).sum(), 1e-12),
                            dtype=torch.float)
    if float(noise.sum()) <= 0:
        noise = torch.ones(num_nodes) / num_nodes

    for _ in range(epochs):
        perm = torch.randperm(len(pairs_t), generator=g)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            centre = pairs_t[idx, 0].to(device)
            context = pairs_t[idx, 1].to(device)

            v_c = emb_in(centre)                              # (B, d)
            v_o = emb_out(context)                            # (B, d)
            pos_score = torch.sum(v_c * v_o, dim=1)
            pos_loss = nn.functional.logsigmoid(pos_score)

            neg_idx = torch.multinomial(noise, len(idx) * negative,
                                        replacement=True).to(device)
            v_n = emb_out(neg_idx).view(len(idx), negative, -1)
            neg_score = torch.bmm(v_n, v_c.unsqueeze(2)).squeeze(2)
            neg_loss = nn.functional.logsigmoid(-neg_score).sum(1)

            loss = -(pos_loss + neg_loss).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    return emb_in.weight.detach().cpu().numpy()


def node2vec(graph: nx.Graph, num_nodes: int, dim=64, num_walks=10,
             walk_length=80, window=10, p=1.0, q=1.0, epochs=5, lr=0.025,
             negative=5, seed=0, device="cpu") -> np.ndarray:
    """End-to-end Node2Vec: walks -> Skip-Gram -> (num_nodes, dim) matrix."""
    rng = np.random.default_rng(seed)
    indptr, indices = build_csr(graph, num_nodes)
    walks = random_walks(indptr, indices, num_nodes, num_walks=num_walks,
                         walk_length=walk_length, p=p, q=q, rng=rng)
    pairs = _skipgram_pairs(walks, window, rng)
    return skipgram_embeddings(pairs, num_nodes, dim=dim, epochs=epochs, lr=lr,
                               negative=negative, device=device, seed=seed)
