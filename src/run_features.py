"""Stage 1 (BM25 Top-100) + Stage 2 (features) + candidate-recall report.

Usage: python -I src/run_features.py [--smoke]
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C
import data as D
from features import TextIndex, stage1_and_features, FEATURE_NAMES
from util import dirs, Logger, env_info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    proc, res = dirs(args.smoke); log = Logger(res / "run_log.txt")
    log(f"=== run_features smoke={args.smoke} seed={C.SEED} | {env_info()}")
    sha = D.download(); log(f"nfcorpus.zip sha256={sha}")
    doc_ids, titles, texts, queries, qrels = D.load()
    stats = D.dataset_stats(doc_ids, titles, texts, queries, qrels)
    if args.smoke:
        qrels = D.subset_qrels(qrels, 60, C.SEED)
    D.assert_query_disjoint(qrels)
    (res / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    log("dataset stats: " + json.dumps(stats))

    t0 = time.perf_counter()
    index = TextIndex(titles, texts, C.BM25_K1, C.BM25_B)
    log(f"index built in {time.perf_counter()-t0:.1f}s | vocab={len(index.vocab)}")

    split_of = {q: s for s in D.SPLITS for q in qrels[s]}
    qids = [q for s in D.SPLITS for q in sorted(qrels[s])]
    t0 = time.perf_counter()
    df = stage1_and_features(index, qids, [queries[q] for q in qids], C.TOP_K, C.LM_MU)
    log(f"stage1+2 for {len(qids)} queries -> {len(df)} rows in {time.perf_counter()-t0:.1f}s")

    # labels attached AFTER feature computation (features never see them)
    did = np.array(doc_ids)
    df["doc_id"] = did[df["doc_idx"].values]
    df["split"] = df["qid"].map(split_of)
    df["label"] = [qrels[s][q].get(d, 0) for s, q, d in zip(df["split"], df["qid"], df["doc_id"])]
    df = df[["qid", "split", "doc_id", "doc_idx", "label"] + FEATURE_NAMES]
    df.to_pickle(proc / "features.pkl")

    # ---- candidate recall (course slides: do not ignore candidate-generation recall)
    rows = []
    n_by_q = df.groupby("qid").size()
    rel_in = df[df["label"] >= 1].groupby("qid").size()
    for s in D.SPLITS:
        for q, d in qrels[s].items():
            nrel = sum(1 for r in d.values() if r >= 1)
            k = int(rel_in.get(q, 0))
            rows.append({"qid": q, "split": s, "query": queries[q], "n_cand": int(n_by_q.get(q, 0)),
                         "n_rel_total": nrel, "n_rel_in_cand": k, "recall_at_100": k / nrel if nrel else np.nan})
    cr = pd.DataFrame(rows); cr.to_csv(res / "candidate_recall.csv", index=False)
    summ = []
    for s, g in cr.groupby("split"):
        summ.append({"split": s, "n_queries": len(g), "recall@100_macro": g["recall_at_100"].mean(),
                     "recall@100_micro": g["n_rel_in_cand"].sum() / g["n_rel_total"].sum(),
                     "frac_queries_zero_rel_in_cand": float((g["n_rel_in_cand"] == 0).mean()),
                     "n_queries_zero_rel_in_cand": int((g["n_rel_in_cand"] == 0).sum()),
                     "frac_queries_lt100_cand": float((g["n_cand"] < C.TOP_K).mean()),
                     "mean_rel_per_query": g["n_rel_total"].mean(),
                     "frac_queries_with_gt100_rel": float((g["n_rel_total"] > C.TOP_K).mean())})
    pd.DataFrame(summ).to_csv(res / "candidate_recall_summary.csv", index=False)
    log("candidate recall:\n" + pd.DataFrame(summ).to_string())
    log(f"label counts among candidates:\n{df.groupby('split')['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
