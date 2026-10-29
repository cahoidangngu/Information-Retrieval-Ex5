"""Helpers for the stemmed-Stage-1 experiment: stemming analyzer, Stage-1 recall summary, oracle upper bound."""
import numpy as np
import pandas as pd
import metrics as M
from bm25 import tokenize
from porter import stem


def stem_analyzer(text):
    """v1 tokenizer (lower-case, [a-z0-9]+, stop-words removed) followed by Porter stemming of every token."""
    return [stem(t) for t in tokenize(text)]


def recall_summary(cr, top_k=100):
    """cr: per-query DataFrame [split, n_cand, n_rel_total, n_rel_in_cand]. One row per split."""
    rows = []
    for s, g in cr.groupby("split", sort=False):
        rec = g["n_rel_in_cand"] / g["n_rel_total"]
        rows.append({"split": s, "n_queries": len(g), "recall@100_micro": g["n_rel_in_cand"].sum() / g["n_rel_total"].sum(),
                     "recall@100_macro": rec.mean(), "frac_queries_no_relevant_candidate": float((g["n_rel_in_cand"] == 0).mean()),
                     "n_queries_no_relevant_candidate": int((g["n_rel_in_cand"] == 0).sum()),
                     "frac_queries_lt100_cand": float((g["n_cand"] < top_k).mean())})
    return pd.DataFrame(rows)


def oracle_and_stage1(frame, qrels_split, score_col="bm25_comb"):
    """Oracle re-ranking (true labels) and the Stage-1 ranking itself, for one split's candidate rows.
    Queries without candidates count as 0. Returns dict of mean metrics."""
    q, y, r = frame["qid"].values, frame["label"].values, frame["bm25_rank"].values
    orc = pd.DataFrame(M.evaluate_scores(q, y, r, y.astype(float), qrels_split))
    s1 = pd.DataFrame(M.evaluate_scores(q, y, r, frame[score_col].values, qrels_split))
    return {"oracle_NDCG@10": orc["ndcg10"].mean(), "oracle_MAP": orc["ap"].mean(),
            "stage1_NDCG@10": s1["ndcg10"].mean(), "stage1_MAP": s1["ap"].mean()}
