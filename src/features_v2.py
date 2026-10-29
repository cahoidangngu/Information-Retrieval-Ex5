"""Feature set v2 = v1 features + 5 new text-only features (v1 code and cache are NOT modified).

New features (no labels involved, computed on the SAME Top-100 candidate rows as v1):
  bm25_stem_comb     BM25 (k1=1.2, b=.75) on Porter-stemmed title+text
  bigram_match_count number of distinct query bigrams (adjacent query tokens after stop-word removal)
                     that occur as adjacent tokens in the document title or in the document text
  min_term_dist      smallest token distance between two DIFFERENT matched query terms in title+text
  avg_term_dist      mean over pairs of different matched query terms of their smallest pairwise distance
                     (min_/avg_ are set to the document length in tokens when fewer than 2 distinct query
                     terms match, i.e. "no proximity evidence"; distance = |pos_i - pos_j| in the
                     title+text token sequence, title first)
  bm25_k2_b05_comb   BM25 with k1=2.0, b=0.5 on title+text (unstemmed)
"""
import itertools
import numpy as np
import pandas as pd
from bm25 import tokenize, build_vocab, count_matrix, bm25_idf, bm25_weight_matrix, binary_query_matrix
from porter import stem
from features import FEATURE_NAMES as V1_NAMES

V2_NEW = ["bm25_stem_comb", "bigram_match_count", "min_term_dist", "avg_term_dist", "bm25_k2_b05_comb"]
V2_NAMES = V1_NAMES + V2_NEW


def bigrams(tokens):
    return set(zip(tokens, tokens[1:]))


def proximity(positions, terms, doc_len):
    """positions: {term: sorted np.array of token positions}; terms: distinct query terms.
    Returns (min_dist, avg_dist) over pairs of matched distinct terms; doc_len if < 2 matched."""
    matched = [t for t in terms if t in positions]
    if len(matched) < 2:
        return float(doc_len), float(doc_len)
    dists = []
    for a, b in itertools.combinations(matched, 2):
        pa, pb = positions[a], positions[b]
        idx = np.searchsorted(pb, pa)
        left = np.abs(pa - pb[np.clip(idx - 1, 0, len(pb) - 1)])
        right = np.abs(pa - pb[np.clip(idx, 0, len(pb) - 1)])
        dists.append(float(np.minimum(left, right).min()))
    return min(dists), float(np.mean(dists))


def compute_v2_new(v1, titles, texts, query_ids, query_texts):
    """v1: DataFrame with columns qid, doc_idx (candidate rows). Returns DataFrame of the new columns, row-aligned."""
    tt = [tokenize(t) for t in titles]; bt = [tokenize(x) for x in texts]
    ct = [a + b for a, b in zip(tt, bt)]
    qtok = {q: tokenize(t) for q, t in zip(query_ids, query_texts)}
    qi_of = {q: i for i, q in enumerate(query_ids)}
    # BM25 k1=2, b=.5 (unstemmed) and stemmed BM25 (k1=1.2, b=.75) over the whole corpus
    vocab = build_vocab(ct); tf = count_matrix(ct, vocab)
    W2 = bm25_weight_matrix(tf, 2.0, 0.5, bm25_idf(tf))
    Q = binary_query_matrix([qtok[q] for q in query_ids], vocab)
    S2 = np.asarray((Q @ W2.T).todense())
    st_ct = [[stem(t) for t in d] for d in ct]
    svocab = build_vocab(st_ct); stf = count_matrix(st_ct, svocab)
    WS = bm25_weight_matrix(stf, 1.2, 0.75, bm25_idf(stf))
    QS = binary_query_matrix([[stem(t) for t in qtok[q]] for q in query_ids], svocab)
    SS = np.asarray((QS @ WS.T).todense())
    # per-document caches for bigrams and token positions
    big_cache, pos_cache = {}, {}

    def doc_big(d):
        if d not in big_cache:
            big_cache[d] = bigrams(tt[d]) | bigrams(bt[d])
        return big_cache[d]

    def doc_pos(d):
        if d not in pos_cache:
            p = {}
            for i, t in enumerate(ct[d]):
                p.setdefault(t, []).append(i)
            pos_cache[d] = {t: np.array(v) for t, v in p.items()}
        return pos_cache[d]

    out = np.zeros((len(v1), 5))
    for r, (qid, d) in enumerate(zip(v1["qid"].values, v1["doc_idx"].values)):
        qt = qtok[qid]; qi = qi_of[qid]
        out[r, 0] = SS[qi, d]
        out[r, 1] = len(bigrams(qt) & doc_big(d))
        out[r, 2], out[r, 3] = proximity(doc_pos(d), list(dict.fromkeys(qt)), len(ct[d]))
        out[r, 4] = S2[qi, d]
    return pd.DataFrame(out, columns=V2_NEW, index=v1.index)
