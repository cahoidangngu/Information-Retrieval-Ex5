"""Stage 2: query-document features computed ONLY from raw text (no labels anywhere).

compute_features() intentionally has no label/qrels argument.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from bm25 import (tokenize, build_vocab, count_matrix, bm25_idf, bm25_weight_matrix,
                  tfidf_normalized, binary_query_matrix, top_k)

FEATURE_NAMES = ["bm25_title", "bm25_body", "bm25_comb", "tfidf_cos_title", "tfidf_cos_body",
                 "lm_dirichlet", "query_len", "doc_len", "title_len", "matched_terms",
                 "coverage_ratio", "sum_idf_matched", "max_idf_matched", "bm25_rank"]


class TextIndex:
    def __init__(self, titles, texts, k1, b, analyzer=tokenize):
        """analyzer: text -> token list, used for documents AND queries (default = v1 tokenizer, unchanged)."""
        self.analyzer = analyzer
        self.tt = [analyzer(t) for t in titles]
        self.bt = [analyzer(x) for x in texts]
        self.ct = [a + c for a, c in zip(self.tt, self.bt)]
        self.vocab = build_vocab(self.ct)
        self.tf = {"title": count_matrix(self.tt, self.vocab), "body": count_matrix(self.bt, self.vocab)}
        self.tf["comb"] = (self.tf["title"] + self.tf["body"]).tocsr()
        self.idf = {f: bm25_idf(m) for f, m in self.tf.items()}
        self.W = {f: bm25_weight_matrix(m, k1, b, self.idf[f]) for f, m in self.tf.items()}
        self.tfidf = {f: tfidf_normalized(self.tf[f])[0] for f in ("title", "body")}
        self.tfidf_idf = {f: tfidf_normalized(self.tf[f])[1] for f in ("title", "body")}
        self.doc_len_body = np.array([len(t) for t in self.bt], dtype=float)
        self.doc_len_title = np.array([len(t) for t in self.tt], dtype=float)
        self.doc_len_comb = self.doc_len_title + self.doc_len_body
        cf = np.asarray(self.tf["comb"].sum(axis=0)).ravel()
        self.p_c = cf / cf.sum()                       # collection language model


def stage1_and_features(index, query_ids, query_texts, k, mu):
    """BM25(comb) Top-k candidates + feature table. Returns DataFrame (rows grouped by query)."""
    qtok = [index.analyzer(t) for t in query_texts]
    Q = binary_query_matrix(qtok, index.vocab)
    S = {f: np.asarray((Q @ index.W[f].T).todense()) for f in ("title", "body", "comb")}
    qn = {}
    for f in ("title", "body"):
        Qt = Q.multiply(index.tfidf_idf[f][None, :]).tocsr()
        nr = np.sqrt(np.asarray(Qt.multiply(Qt).sum(axis=1)).ravel()); nr[nr == 0] = 1
        qn[f] = np.asarray(((sp.diags(1.0 / nr) @ Qt) @ index.tfidf[f].T).todense())
    tfc = index.tf["comb"].tocsc(); idf_c = index.idf["comb"]
    rows = []
    for qi, qid in enumerate(query_ids):
        cand = top_k(S["comb"][qi], k)
        if cand.size == 0:
            continue
        terms = Q[qi].indices
        sub = tfc[:, terms][cand].toarray()               # cand x |terms| term freqs
        dl = index.doc_len_comb[cand]
        pc = index.p_c[terms]
        lm = np.log((sub + mu * pc[None, :]) / (dl[:, None] + mu)).sum(axis=1) if terms.size else np.zeros(cand.size)
        matched = sub > 0
        n_u = len(set(qtok[qi]))
        idfm = np.where(matched, idf_c[terms][None, :], 0.0)
        rows.append(pd.DataFrame({
            "qid": qid, "doc_idx": cand,
            "bm25_title": S["title"][qi, cand], "bm25_body": S["body"][qi, cand],
            "bm25_comb": S["comb"][qi, cand],
            "tfidf_cos_title": qn["title"][qi, cand], "tfidf_cos_body": qn["body"][qi, cand],
            "lm_dirichlet": lm, "query_len": float(len(qtok[qi])),
            "doc_len": index.doc_len_body[cand], "title_len": index.doc_len_title[cand],
            "matched_terms": matched.sum(axis=1).astype(float),
            "coverage_ratio": matched.sum(axis=1) / max(n_u, 1),
            "sum_idf_matched": idfm.sum(axis=1),
            "max_idf_matched": idfm.max(axis=1) if terms.size else np.zeros(cand.size),
            "bm25_rank": np.arange(1, cand.size + 1, dtype=float)}))
    return pd.concat(rows, ignore_index=True)
