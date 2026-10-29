"""Transparent BM25 / TF-IDF implementation on scipy sparse matrices.

Tokenisation: lower-case, [a-z0-9]+ tokens, scikit-learn English stop-word list removed,
NO stemming. Statistics (df, avgdl, collection frequencies) are computed from corpus text
only (unsupervised, no labels).
BM25 (per field): idf(t) = ln(1 + (N - df + .5)/(df + .5));
  score(q,d) = sum_{t in UNIQUE(q)} idf(t) * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)).
Query term multiplicity is ignored (each unique query term counts once).
"""
import re
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return [t for t in _TOK.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def build_vocab(token_lists):
    vocab = {}
    for toks in token_lists:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def count_matrix(token_lists, vocab):
    """docs x V term-frequency CSR matrix (out-of-vocab tokens dropped)."""
    indptr, indices, data = [0], [], []
    for toks in token_lists:
        c = {}
        for t in toks:
            j = vocab.get(t)
            if j is not None:
                c[j] = c.get(j, 0) + 1
        indices.extend(c.keys()); data.extend(c.values()); indptr.append(len(indices))
    return sp.csr_matrix((np.array(data, dtype=np.float64), np.array(indices), np.array(indptr)),
                         shape=(len(token_lists), len(vocab)))


def bm25_idf(tf):
    N = tf.shape[0]
    df = np.asarray((tf > 0).sum(axis=0)).ravel()
    return np.log(1.0 + (N - df + 0.5) / (df + 0.5))


def bm25_weight_matrix(tf, k1, b, idf=None):
    """Doc x V matrix W such that score(q, d) = (binary query vector) . W[d]."""
    tf = tf.tocsr(); idf = bm25_idf(tf) if idf is None else idf
    dl = np.asarray(tf.sum(axis=1)).ravel()
    avgdl = dl.mean()
    W = tf.copy()
    rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
    denom = W.data + k1 * (1 - b + b * dl[rows] / avgdl)
    W.data = idf[W.indices] * W.data * (k1 + 1) / denom
    return W


def tfidf_normalized(tf):
    """Sublinear tf (1+ln tf) * smoothed idf, L2-normalised rows."""
    tf = tf.tocsr(); N = tf.shape[0]
    df = np.asarray((tf > 0).sum(axis=0)).ravel()
    idf = np.log((1 + N) / (1 + df)) + 1.0
    M = tf.copy(); M.data = (1 + np.log(M.data)) * idf[M.indices]
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel()); norms[norms == 0] = 1
    M = sp.diags(1.0 / norms) @ M
    return M.tocsr(), idf


def binary_query_matrix(query_token_lists, vocab):
    rows, cols = [], []
    for i, toks in enumerate(query_token_lists):
        for j in {vocab[t] for t in toks if t in vocab}:
            rows.append(i); cols.append(j)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(query_token_lists), len(vocab)))


def top_k(score_row, k):
    """Indices of the k highest scores with score>0; ties broken by doc index (stable)."""
    order = np.argsort(-score_row, kind="stable")[:k]
    return order[score_row[order] > 0]
