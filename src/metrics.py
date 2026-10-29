"""Hand-written ranking metrics.

Conventions (stated in the report):
* rel >= 1 is relevant for MAP (binarised); NDCG uses the graded labels.
* NDCG gain: 'exp' -> 2^rel - 1 (primary), 'lin' -> rel. Discount 1/log2(rank+1).
* IDCG is computed from the FULL qrels of the query (not only the candidate list), so a
  re-ranker is penalised for relevant documents BM25 failed to retrieve.
* AP = sum_k P@k * rel_k / (#relevant in FULL qrels), over the ranked list (<=100 docs).
* Unjudged documents are non-relevant (label 0).
* Pairwise accuracy: over pairs (i,j) of the same query with label_i > label_j, a pair is
  correct if score_i > score_j; ties count 0.5.
"""
import numpy as np


def _gain(rel, gain):
    rel = np.asarray(rel, dtype=float)
    return (2.0 ** rel - 1.0) if gain == "exp" else rel


def dcg_at_k(rels_in_rank_order, k, gain="exp"):
    r = np.asarray(rels_in_rank_order)[:k]
    if r.size == 0:
        return 0.0
    return float(np.sum(_gain(r, gain) / np.log2(np.arange(2, r.size + 2))))


def ndcg_at_k(rels_in_rank_order, all_rels, k, gain="exp"):
    ideal = np.sort(np.asarray(all_rels))[::-1]
    idcg = dcg_at_k(ideal, k, gain)
    return 0.0 if idcg == 0 else dcg_at_k(rels_in_rank_order, k, gain) / idcg


def average_precision(rels_in_rank_order, n_relevant_total):
    if n_relevant_total == 0:
        return 0.0
    b = (np.asarray(rels_in_rank_order) >= 1).astype(float)
    if b.size == 0:
        return 0.0
    prec = np.cumsum(b) / np.arange(1, b.size + 1)
    return float(np.sum(prec * b) / n_relevant_total)


def pairwise_counts(scores, labels):
    """(correct, total, n_ties) over pairs with different labels (ties = 0.5 correct)."""
    s = np.asarray(scores, dtype=float); y = np.asarray(labels)
    hi = y[:, None] > y[None, :]
    total = int(hi.sum())
    if total == 0:
        return 0.0, 0, 0
    d = s[:, None] - s[None, :]
    correct = float((hi & (d > 0)).sum()) + 0.5 * float((hi & (d == 0)).sum())
    return correct, total, int((hi & (d == 0)).sum())


def rank_order(scores, tiebreak_rank):
    """Indices sorted by score desc; ties broken by original BM25 rank (deterministic)."""
    return np.lexsort((np.asarray(tiebreak_rank), -np.asarray(scores, dtype=float)))


def evaluate_scores(qids, labels, bm25_rank, scores, qrels_split):
    """Per-query metrics for one scoring of the candidate rows (rows of a query contiguous).

    qrels_split: {qid: {docid: rel}} FULL judgments of the split.
    """
    qids = np.asarray(qids); out = []
    bounds = np.flatnonzero(np.r_[True, qids[1:] != qids[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        q = qids[a]
        y = np.asarray(labels[a:b]); s = np.asarray(scores[a:b])
        order = rank_order(s, bm25_rank[a:b])
        ry = y[order]
        full = np.array(list(qrels_split[q].values()))
        nrel = int((full >= 1).sum())
        c, t, ties = pairwise_counts(s, y)
        out.append({"qid": q, "n_cand": int(b - a), "n_rel_total": nrel, "n_rel_in_cand": int((y >= 1).sum()),
                    "ndcg5": ndcg_at_k(ry, full, 5), "ndcg10": ndcg_at_k(ry, full, 10),
                    "ndcg10_lin": ndcg_at_k(ry, full, 10, "lin"),
                    "ap": average_precision(ry, nrel),
                    "pair_correct": c, "pair_total": t, "pair_ties": ties,
                    "pair_acc": (c / t) if t else np.nan})
    # queries with NO BM25 candidate (no term match) still count: all metrics 0 for every model
    seen = {r["qid"] for r in out}
    for q, d in qrels_split.items():
        if q not in seen:
            nrel = sum(1 for r in d.values() if r >= 1)
            out.append({"qid": q, "n_cand": 0, "n_rel_total": nrel, "n_rel_in_cand": 0, "ndcg5": 0.0,
                        "ndcg10": 0.0, "ndcg10_lin": 0.0, "ap": 0.0, "pair_correct": 0.0,
                        "pair_total": 0, "pair_ties": 0, "pair_acc": np.nan})
    return out


def summarize(per_query_rows):
    import pandas as pd
    d = pd.DataFrame(per_query_rows)
    tot = d["pair_total"].sum()
    return {"MAP": d["ap"].mean(), "NDCG@5": d["ndcg5"].mean(), "NDCG@10": d["ndcg10"].mean(),
            "NDCG@10_lin": d["ndcg10_lin"].mean(),
            "pair_acc_micro": d["pair_correct"].sum() / tot if tot else np.nan,
            "pair_acc_macro": d["pair_acc"].mean(), "n_queries": len(d),
            "n_pairs": int(tot), "tie_frac": d["pair_ties"].sum() / tot if tot else np.nan}
