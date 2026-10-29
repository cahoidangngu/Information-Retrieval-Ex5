"""Pairwise training-data construction for Ranking SVM.

Rules (guarded by assertions in build_pairs and tests/):
* pairs are formed ONLY inside one query (rows of a query are contiguous, groups = qid runs);
* only pairs with DIFFERENT labels (equal labels carry no preference);
* each pair is stored once as (x_hi - x_lo, +1) and then a random, EXACTLY half of the pairs
  is flipped to (x_lo - x_hi, -1) so the two classes are balanced (no intercept is fitted).
"""
import numpy as np


def group_bounds(qids):
    qids = np.asarray(qids)
    b = np.flatnonzero(np.r_[True, qids[1:] != qids[:-1], True])
    return list(zip(b[:-1], b[1:]))


def count_pairs(y, qids):
    """Number of valid (same query, label_i > label_j) pairs, without materialising them."""
    y = np.asarray(y); tot = 0
    for a, b in group_bounds(qids):
        g = y[a:b]
        tot += int((g[:, None] > g[None, :]).sum())
    return tot


def estimate_memory_gb(n_pairs, n_feat):
    """Dense float64 pair matrix + liblinear's internal copy (16 B per feature_node, d+2 nodes/row)."""
    return n_pairs * (8 * n_feat + 16 * (n_feat + 2)) / 1024 ** 3


def build_pairs(X, y, qids, bm25_rank, mode, rng, cap=None):
    """mode: 'plain' (weights None) | 'gap' (w=|y_i-y_j|) | 'dndcg' (rank-aware, extra variant).
    cap: None = all valid pairs; int = uniform sample of at most `cap` pairs per query.
    Returns Xd, yd(+-1), w(or None), (I, J) global row indices (after orientation I=higher label)."""
    y = np.asarray(y); qids = np.asarray(qids)
    I, J = [], []
    for a, b in group_bounds(qids):
        g = y[a:b]
        ii, jj = np.nonzero(g[:, None] > g[None, :])
        if cap is not None and ii.size > cap:
            sel = rng.choice(ii.size, size=cap, replace=False)
            ii, jj = ii[sel], jj[sel]
        I.append(ii + a); J.append(jj + a)
    I = np.concatenate(I); J = np.concatenate(J)
    assert np.all(qids[I] == qids[J]), "pair crosses queries"
    assert np.all(y[I] > y[J]), "pair with equal/inverted labels"
    w = None
    if mode == "gap":
        w = (y[I] - y[J]).astype(float)
    elif mode == "dndcg":
        disc = lambda r: 1.0 / np.log2(1.0 + r)
        w = np.abs(2.0 ** y[I] - 2.0 ** y[J]) * np.abs(disc(bm25_rank[I]) - disc(bm25_rank[J]))
        w = w / w.mean()
    n = I.size
    flip = np.zeros(n, dtype=bool); flip[rng.permutation(n)[: n // 2]] = True
    sign = np.where(flip, -1.0, 1.0)
    Xd = (X[I] - X[J]) * sign[:, None]
    return Xd, sign.astype(int), w, (I, J)
