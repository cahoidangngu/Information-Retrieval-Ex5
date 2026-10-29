"""EXPERIMENT (small, additive): Porter-stemmed Stage 1 (BM25 Top-100) + v1 features recomputed on the new candidates.

Usage: python -I src/run_stemmed_stage1.py [--smoke]     Writes results/ext_stem_*.csv; nothing existing is modified.
The WHOLE stage-1/2 analyzer is the stemming one (query and document tokens are Porter-stemmed after stop-word removal), so all
14 v1 features, incl. bm25_comb / bm25_rank, are computed in stemmed space on the stemmed candidates.
Protocol identical to the main run: official split, scaler on train only, C on dev only (NDCG@10), test scored once.
Models: linear regression, logistic regression, sampled Ranking SVM (squared hinge), same seeds as the main run.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C
import data as D
import metrics as M
import stats as ST
import ltr_runner as LR
import stage1_eval as S1
from features import TextIndex, stage1_and_features, FEATURE_NAMES
from util import dirs, Logger, env_info

MODELS = ["linreg", "logreg", "ranksvm_sampled"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    proc, res = dirs(args.smoke); log = Logger(res / "run_log.txt")
    log(f"=== run_stemmed_stage1 smoke={args.smoke} | {env_info()}")
    doc_ids, titles, texts, queries, qrels = D.load()
    cr0 = pd.read_csv(res / "candidate_recall.csv")           # unstemmed Stage 1 (main run); defines the evaluated queries
    qrels = {s: {q: d for q, d in qrels[s].items() if q in set(cr0.loc[cr0["split"] == s, "qid"])} for s in D.SPLITS}
    split_of = {q: s for s in D.SPLITS for q in qrels[s]}
    qids = [q for s in D.SPLITS for q in sorted(qrels[s])]

    t0 = time.perf_counter()
    index = TextIndex(titles, texts, C.BM25_K1, C.BM25_B, analyzer=S1.stem_analyzer)
    df = stage1_and_features(index, qids, [queries[q] for q in qids], C.TOP_K, C.LM_MU)
    log(f"stemmed index+stage1+features: {len(df)} rows for {len(qids)} queries in {time.perf_counter()-t0:.1f}s | vocab={len(index.vocab)}")
    did = np.array(doc_ids)
    df["doc_id"] = did[df["doc_idx"].values]; df["split"] = df["qid"].map(split_of)
    df["label"] = [qrels[s][q].get(d, 0) for s, q, d in zip(df["split"], df["qid"], df["doc_id"])]
    df = df[["qid", "split", "doc_id", "doc_idx", "label"] + FEATURE_NAMES]
    df.to_pickle(proc / "features_stemmed_stage1.pkl")

    # ---- Stage-1 comparison (stemmed vs unstemmed), per split
    n_by_q = df.groupby("qid").size(); rel_in = df[df["label"] >= 1].groupby("qid").size()
    rows = []
    for s in D.SPLITS:
        for q, d in qrels[s].items():
            rows.append({"qid": q, "split": s, "n_cand": int(n_by_q.get(q, 0)), "n_rel_total": sum(1 for r in d.values() if r >= 1),
                         "n_rel_in_cand": int(rel_in.get(q, 0))})
    cr1 = pd.DataFrame(rows); cr1.to_csv(res / "ext_stem_candidate_recall.csv", index=False)
    old = pd.read_pickle(proc / "features.pkl")
    out = []
    for name, cr, frame in (("unstemmed (main)", cr0, old), ("Porter-stemmed", cr1, df)):
        summ = S1.recall_summary(cr)
        for _, r in summ.iterrows():
            o = S1.oracle_and_stage1(frame[frame["split"] == r["split"]], qrels[r["split"]])
            out.append({"stage1": name, **r.to_dict(), **o})
    pd.DataFrame(out).to_csv(res / "ext_stem_stage1_summary.csv", index=False)
    log("stage-1 comparison:\n" + pd.DataFrame(out).round(4).to_string())

    # ---- retrain on the stemmed candidates (same protocol)
    Sx = {s: df[df["split"] == s].reset_index(drop=True) for s in D.SPLITS}
    sc, (Xtr, Xdev, Xte) = LR.scale(Sx["train"], [Sx["train"], Sx["dev"], Sx["test"]], FEATURE_NAMES)
    g = lambda s: (Sx[s]["label"].values, Sx[s]["qid"].values, Sx[s]["bm25_rank"].values)
    (ytr, qtr, rtr), (yd, qd, rd), (yt, qt, rt) = g("train"), g("dev"), g("test")
    tun, mt, md_, pq, wr = [], [], [], [], []
    base_rows = M.evaluate_scores(qt, yt, rt, Sx["test"]["bm25_comb"].values, qrels["test"])       # stemmed BM25 alone (Stage-1 order)
    for x in base_rows:
        x["model"] = "stemmed_bm25"
    pq += base_rows
    mt.append({"model": "stemmed_bm25", "C": np.nan, **M.summarize(base_rows), "train_s": 0.0})
    md_.append({"model": "stemmed_bm25", "C": np.nan, **M.summarize(M.evaluate_scores(qd, yd, rd, Sx["dev"]["bm25_comb"].values, qrels["dev"]))})
    for name in MODELS:
        grid, models, best_c = LR.tune(name, Xtr, ytr, qtr, rtr, Xdev, yd, qd, rd, qrels["dev"], C.SEED, log)
        tun.append(grid); m = LR.get(models, best_c)
        md_.append({"model": name, "C": best_c, **M.summarize(LR.per_query_rows(m, Xdev, yd, qd, rd, qrels["dev"]))})
        r = LR.per_query_rows(m, Xte, yt, qt, rt, qrels["test"])
        for x in r:
            x["model"] = name
        pq += r
        mt.append({"model": name, "C": best_c, **M.summarize(r), "train_s": m["train_s"]})
        wr += [{"model": name, "feature": f, "weight": w} for f, w in zip(FEATURE_NAMES, m["w"])]
    pd.concat(tun).to_csv(res / "ext_stem_tuning_dev.csv", index=False)
    pd.DataFrame(mt).to_csv(res / "ext_stem_metrics_test.csv", index=False)
    pd.DataFrame(md_).to_csv(res / "ext_stem_metrics_dev.csv", index=False)
    pqd = pd.DataFrame(pq); pqd.to_csv(res / "ext_stem_per_query_test.csv", index=False)
    pd.DataFrame(wr).to_csv(res / "ext_stem_weights.csv", index=False)

    # ---- paired tests: (a) vs UNSTEMMED BM25 (main baseline), (b) vs STEMMED BM25 alone
    main_pq = pd.read_csv(res / "per_query_test.csv")
    base_un = main_pq[main_pq["model"] == "bm25"]
    a = ST.significance_table(pd.concat([base_un, pqd], ignore_index=True)); a["comparison"] = "(a) vs unstemmed BM25"
    st_only = pqd[pqd["model"] == "stemmed_bm25"].assign(model="bm25")
    b = ST.significance_table(pd.concat([st_only, pqd[pqd["model"] != "stemmed_bm25"]], ignore_index=True)); b["comparison"] = "(b) vs stemmed BM25 alone"
    pd.concat([a, b]).to_csv(res / "ext_stem_significance.csv", index=False)
    log("stemmed-stage1 test metrics:\n" + pd.DataFrame(mt)[["model", "C", "MAP", "NDCG@5", "NDCG@10", "pair_acc_micro"]].to_string())


if __name__ == "__main__":
    main()
