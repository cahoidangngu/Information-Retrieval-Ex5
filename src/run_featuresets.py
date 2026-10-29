"""EXTENSION 4: feature sets v1 / v1_reduced (VIF < 10) / v2 on the official split, same Top-100 candidates.

Usage: python -I src/run_featuresets.py [--smoke]   Writes results/ext_featuresets_*.csv and data/processed/<run>/features_v2.pkl.
Scaler fitted on train only; C tuned on dev only. v1 results must equal the main pipeline (asserted).
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
from analysis import EXPECTED
from features import FEATURE_NAMES
from features_v2 import V2_NAMES, V2_NEW, compute_v2_new
from util import dirs, Logger, env_info

MODELS = ["linreg", "logreg", "ranksvm_all", "ranksvm_sampled"]
EXPECTED_ALL = dict(EXPECTED, bm25_stem_comb=1, bigram_match_count=1, min_term_dist=-1, avg_term_dist=-1, bm25_k2_b05_comb=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    proc, res = dirs(args.smoke); log = Logger(res / "run_log.txt")
    log(f"=== run_featuresets smoke={args.smoke} | {env_info()}")
    df = pd.read_pickle(proc / "features.pkl")
    doc_ids, titles, texts, queries, qrels = D.load()
    cr = pd.read_csv(res / "candidate_recall.csv")
    qrels = {s: {q: d for q, d in qrels[s].items() if q in set(cr.loc[cr["split"] == s, "qid"])} for s in D.SPLITS}

    # ---- v2 features (cached next to, never inside, the v1 cache)
    f2 = proc / "features_v2.pkl"
    if f2.exists():
        df2 = pd.read_pickle(f2)
    else:
        t0 = time.perf_counter(); qids = list(dict.fromkeys(df["qid"]))
        new = compute_v2_new(df[["qid", "doc_idx"]], titles, texts, qids, [queries[q] for q in qids])
        df2 = pd.concat([df, new], axis=1); df2.to_pickle(f2)
        log(f"v2 features computed for {len(df2)} rows in {time.perf_counter()-t0:.1f}s")
    assert (df2[FEATURE_NAMES].values == df[FEATURE_NAMES].values).all() and (df2["qid"].values == df["qid"].values).all()
    assert not df2[V2_NEW].isna().any().any()

    tr_rows = df2[df2["split"] == "train"]
    kept, drops = LR.vif_prune(tr_rows[FEATURE_NAMES], 10.0)
    drops.to_csv(res / "ext_vif_pruning_v1.csv", index=False)
    log(f"v1_reduced keeps {kept}; dropped in order {list(drops['feature'])}")
    vif_rows = []
    for name, feats in (("v1", FEATURE_NAMES), ("v1_reduced", kept), ("v2", V2_NAMES)):
        v = LR.vif(tr_rows[feats])
        vif_rows += [{"feature_set": name, "feature": f, "VIF": v[f]} for f in feats]
    pd.DataFrame(vif_rows).to_csv(res / "ext_vif_by_set.csv", index=False)
    SETS = {"v1": FEATURE_NAMES, "v1_reduced": kept, "v2": V2_NAMES}

    S = {s: df2[df2["split"] == s].reset_index(drop=True) for s in D.SPLITS}
    cols = lambda s: (S[s]["label"].values, S[s]["qid"].values, S[s]["bm25_rank"].values)
    (ytr, qtr, rtr), (yd, qd, rd), (yt, qt, rt) = cols("train"), cols("dev"), cols("test")
    main_pq = pd.read_csv(res / "per_query_test.csv")
    main_t = pd.read_csv(res / "metrics_test.csv").set_index("model")

    tun, mtest, mdev, pq, wrows = [], [], [], [], []
    for sname, feats in SETS.items():
        log(f"--- feature set {sname} ({len(feats)} features)")
        sc, (Xtr, Xdev, Xte) = LR.scale(S["train"], [S["train"], S["dev"], S["test"]], feats)
        for name in MODELS:
            grid, models, best_c = LR.tune(name, Xtr, ytr, qtr, rtr, Xdev, yd, qd, rd, qrels["dev"], C.SEED, log)
            grid.insert(0, "feature_set", sname); tun.append(grid)
            m = LR.get(models, best_c)
            rd_rows = LR.per_query_rows(m, Xdev, yd, qd, rd, qrels["dev"]); mdev.append({"feature_set": sname, "model": name, "C": best_c, **M.summarize(rd_rows)})
            r = LR.per_query_rows(m, Xte, yt, qt, rt, qrels["test"])
            for x in r:
                x["model"] = f"{sname}:{name}"; x["feature_set"] = sname; x["base_model"] = name
            pq += r
            mtest.append({"feature_set": sname, "model": name, "C": best_c, **M.summarize(r), "train_s": m["train_s"]})
            wrows += [{"feature_set": sname, "model": name, "feature": f, "weight": w} for f, w in zip(feats, m["w"])]
            if sname == "v1":    # reproducibility regression check against the main pipeline
                ref = main_t.loc[name]
                assert abs(ref["NDCG@10"] - mtest[-1]["NDCG@10"]) < 1e-9 and abs(ref["MAP"] - mtest[-1]["MAP"]) < 1e-9, \
                    f"v1/{name} differs from main results"
    # single-feature reranks (fixed hyper-parameters, nothing tuned): is the v2 gain just stemming / BM25(k1=2,b=.5)?
    sf = []
    for feat in ("bm25_stem_comb", "bm25_k2_b05_comb"):
        for split, (y_, q_, r_) in (("test", (yt, qt, rt)), ("dev", (yd, qd, rd))):
            rr = M.evaluate_scores(q_, y_, r_, S[split][feat].values, qrels[split])
            if split == "test":
                for x in rr:
                    x["model"] = f"rerank:{feat}"
                pq += rr
            sf.append({"split": split, "model": f"rerank:{feat}", **M.summarize(rr)})
    pd.DataFrame(sf).to_csv(res / "ext_single_feature_reranks.csv", index=False)
    pd.concat(tun).to_csv(res / "ext_featuresets_tuning_dev.csv", index=False)
    pd.DataFrame(mtest).to_csv(res / "ext_featuresets_metrics_test.csv", index=False)
    pd.DataFrame(mdev).to_csv(res / "ext_featuresets_metrics_dev.csv", index=False)
    pqd = pd.DataFrame(pq); pqd.to_csv(res / "ext_featuresets_per_query_test.csv", index=False)
    pd.DataFrame(wrows).to_csv(res / "ext_featuresets_weights.csv", index=False)

    # ---- sign sanity check (expected signs fixed a priori, see analysis.EXPECTED + v2 additions)
    wdf = pd.DataFrame(wrows)
    sg = []
    for (s, f), g in wdf.groupby(["feature_set", "feature"], sort=False):
        row = {"feature_set": s, "feature": f, "expected_sign": EXPECTED_ALL[f]}
        for m_ in ("linreg", "logreg", "ranksvm_all"):
            w = float(g[g["model"] == m_]["weight"].iloc[0]); row[f"w_{m_}"] = w
            row[f"ok_{m_}"] = "n/a" if EXPECTED_ALL[f] == 0 or w == 0 else ("yes" if np.sign(w) == EXPECTED_ALL[f] else "NO")
        sg.append(row)
    pd.DataFrame(sg).to_csv(res / "ext_featuresets_sign_check.csv", index=False)

    # ---- significance. (i) each (set, model) vs BM25; Holm across all 12 per metric
    base = main_pq[main_pq["model"] == "bm25"]
    sig = ST.significance_table(pd.concat([base, pqd], ignore_index=True)); sig["comparison"] = "vs BM25"
    outs = [sig]
    # (i-b) v2 models vs the stemmed-BM25-only rerank (does v2 add anything beyond stemming?)
    for m_ in MODELS:
        a = pqd[pqd["model"] == "rerank:bm25_stem_comb"].assign(model="bm25"); b = pqd[pqd["model"] == f"v2:{m_}"]
        s_ = ST.significance_table(pd.concat([a, b], ignore_index=True)); s_["comparison"] = f"v2:{m_} vs rerank:bm25_stem_comb"
        outs.append(s_)
    # (ii) v2 / v1_reduced vs v1 with the same model (paired on queries)
    for other in ("v2", "v1_reduced"):
        for m_ in MODELS:
            a = pqd[pqd["model"] == f"v1:{m_}"].assign(model="bm25"); b = pqd[pqd["model"] == f"{other}:{m_}"]
            s = ST.significance_table(pd.concat([a, b], ignore_index=True)); s["comparison"] = f"{other}:{m_} vs v1:{m_}"
            outs.append(s)
    pd.concat(outs).to_csv(res / "ext_featuresets_significance.csv", index=False)
    log("feature-set test metrics:\n" + pd.DataFrame(mtest)[["feature_set", "model", "C", "MAP", "NDCG@10", "pair_acc_micro"]].to_string())


if __name__ == "__main__":
    main()
