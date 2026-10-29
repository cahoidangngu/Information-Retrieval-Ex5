"""Train / tune / evaluate all rankers on the SAME Top-100 candidate sets.

Usage: python -I src/run_models.py [--smoke]
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config as C
import data as D
import metrics as M
import models as MD
import pairs as P
import stats as ST
from features import FEATURE_NAMES
from util import dirs, Logger, env_info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    proc, res = dirs(args.smoke); log = Logger(res / "run_log.txt")
    log(f"=== run_models smoke={args.smoke} seed={C.SEED} | {env_info()}")
    df = pd.read_pickle(proc / "features.pkl")
    _, _, _, _, qrels = D.load()
    cr = pd.read_csv(res / "candidate_recall.csv")   # all evaluated queries (incl. 0-candidate)
    qrels = {s: {q: d for q, d in qrels[s].items() if q in set(cr.loc[cr["split"] == s, "qid"])} for s in D.SPLITS}
    S = {s: df[df["split"] == s].reset_index(drop=True) for s in D.SPLITS}
    qsets = {s: set(S[s]["qid"]) for s in S}
    assert not (qsets["train"] & qsets["dev"]) and not (qsets["train"] & qsets["test"]) and not (qsets["dev"] & qsets["test"]), "query leakage between splits"
    # Query-level split guard: a query's rows are contiguous and live in exactly one split
    for s in S:
        assert S[s]["qid"].is_monotonic_increasing or S[s]["qid"].nunique() == (S[s]["qid"] != S[s]["qid"].shift()).sum()

    # ---- standardisation: fitted on TRAIN candidate rows only
    scaler = StandardScaler().fit(S["train"][FEATURE_NAMES].values)
    assert np.allclose(scaler.mean_, S["train"][FEATURE_NAMES].values.mean(axis=0))
    X = {s: scaler.transform(S[s][FEATURE_NAMES].values) for s in S}
    pd.DataFrame({"feature": FEATURE_NAMES, "train_mean": scaler.mean_, "train_std": scaler.scale_}).to_csv(res / "scaler_stats.csv", index=False)
    y = {s: S[s]["label"].values for s in S}
    q = {s: S[s]["qid"].values for s in S}
    rk = {s: S[s]["bm25_rank"].values for s in S}

    def per_query(split, scores):
        return M.evaluate_scores(q[split], y[split], rk[split], scores, qrels[split])

    # ---- pair statistics / memory control
    n_all = P.count_pairs(y["train"], q["train"]); d = len(FEATURE_NAMES)
    est = P.estimate_memory_gb(n_all, d)
    fits = est <= C.PAIR_MEM_BUDGET_GB
    gb = lambda n: P.estimate_memory_gb(n, d)
    pair_stats = {"n_train_rows": len(S["train"]), "n_train_queries": len(qsets["train"]), "n_features": d,
                  "n_valid_pairs_all": n_all, "est_train_memory_gb_all_pairs": est,
                  "memory_budget_gb": C.PAIR_MEM_BUDGET_GB, "all_pairs_fit_budget": bool(fits),
                  "sample_cap_per_query": C.PAIR_CAP_PER_QUERY}
    log("pair stats: " + json.dumps(pair_stats))
    if not fits:
        log("ALL-PAIRS DOES NOT FIT the memory budget -> all-pairs variant is replaced by capped sampling "
            "(uniform without replacement, cap per query as above); this is stated in the report.")
    cap_all = None if fits else C.PAIR_CAP_PER_QUERY

    # ---- variants
    variants = [  # name, kind, pair mode, cap
        ("ranksvm_all", "svm", "plain", cap_all),
        ("ranksvm_sampled", "svm", "plain", C.PAIR_CAP_PER_QUERY),
        ("ranksvm_gapweighted", "svm", "gap", cap_all),
        ("ranksvm_dndcgweighted_extra", "svm", "dndcg", cap_all)]
    ranker_names = ["bm25", "linreg", "logreg"] + [v[0] for v in variants]

    tuning, fitted = [], {}      # fitted[(name, C)] = model dict

    def dev_record(name, Cval, m):
        sc = MD.score(m, X["dev"]); rows = per_query("dev", sc); sm = M.summarize(rows)
        tuning.append({"model": name, "C": Cval, **{k: sm[k] for k in ("MAP", "NDCG@5", "NDCG@10", "pair_acc_micro")},
                       "train_s": m["train_s"], "peak_mb": m["peak_mb"], "no_converge": m["no_converge"],
                       "n_pairs": m.get("n_pairs", np.nan)})
        fitted[(name, Cval)] = m
        log(f"  {name:30s} C={Cval!s:7s} dev NDCG@10={sm['NDCG@10']:.4f} MAP={sm['MAP']:.4f} "
            f"pairacc={sm['pair_acc_micro']:.4f} train={m['train_s']:.1f}s peak={m['peak_mb']:.0f}MB conv_warn={m['no_converge']}")

    log("tuning on DEV (never on test)")
    m = MD.fit_linreg(X["train"], y["train"].astype(float)); dev_record("linreg", np.nan, m)
    for Cv in C.C_GRID:
        m = MD.fit_logreg(X["train"], (y["train"] >= 1).astype(int), Cv); dev_record("logreg", Cv, m)
    for k, (name, _, mode, cap) in enumerate(variants):
        rng = np.random.default_rng(C.SEED + k)
        t0 = time.perf_counter()
        Xd, yd, w, (I, J) = P.build_pairs(X["train"], y["train"], q["train"], rk["train"], mode, rng, cap)
        bal = float((yd == 1).mean())
        assert abs(bal - 0.5) <= 1.0 / max(len(yd), 1) + 1e-9, "pair directions unbalanced"
        log(f"{name}: {len(yd)} pairs ({Xd.nbytes/1024**2:.0f} MB dense), +1 fraction={bal:.4f}, built in {time.perf_counter()-t0:.1f}s")
        pair_stats[f"n_pairs_{name}"] = int(len(yd))
        for Cv in C.C_GRID:
            m = MD.fit_ranksvm(Xd, yd, w, Cv); m["n_pairs"] = len(yd); dev_record(name, Cv, m)
        del Xd, yd, w, I, J
    (res / "pair_stats.json").write_text(json.dumps(pair_stats, indent=2))
    tun = pd.DataFrame(tuning); tun.to_csv(res / "tuning_dev.csv", index=False)

    # ---- choose C per model on DEV; evaluate on TEST
    chosen = {}
    for name in ranker_names[1:]:
        t = tun[tun["model"] == name]
        chosen[name] = t.loc[t[C.SELECT_METRIC].idxmax(), "C"]     # first max -> smallest C on ties
    log("chosen C (dev " + C.SELECT_METRIC + "): " + json.dumps({k: (None if pd.isna(v) else v) for k, v in chosen.items()}))

    def get_model(name):
        Cv = chosen[name]
        return fitted[(name, Cv if not pd.isna(Cv) else np.nan)] if not pd.isna(Cv) else fitted[(name, np.nan)]

    score_cols = {"bm25": {s: S[s]["bm25_comb"].values for s in ("dev", "test")}}
    meta = {"bm25": {"C": np.nan, "train_s": 0.0, "peak_mb": np.nan, "predict_s": np.nan, "n_pairs": np.nan}}
    for name in ranker_names[1:]:
        m = get_model(name)
        score_cols[name] = {s: MD.score(m, X[s]) for s in ("dev", "test")}
        meta[name] = {"C": chosen[name], "train_s": m["train_s"], "peak_mb": m["peak_mb"],
                      "predict_s": MD.predict_time(m, X["test"]), "n_pairs": m.get("n_pairs", np.nan)}
    for split, fn in (("dev", "metrics_dev.csv"), ("test", "metrics_test.csv")):
        rows, pq = [], []
        for name in ranker_names:
            r = per_query(split, score_cols[name][split])
            for x in r:
                x["model"] = name
            pq += r
            rows.append({"model": name, **M.summarize(r), **meta[name]})
        pd.DataFrame(rows).to_csv(res / fn, index=False)
        if split == "test":
            pqd = pd.DataFrame(pq); pqd.to_csv(res / "per_query_test.csv", index=False)
            log("TEST metrics:\n" + pd.DataFrame(rows)[["model", "MAP", "NDCG@5", "NDCG@10", "pair_acc_micro", "C", "train_s", "predict_s"]].to_string())
        else:
            pd.DataFrame(pq).to_csv(res / "per_query_dev.csv", index=False)

    # ---- scores, weights, best-LTR-by-DEV
    out = S["test"][["qid", "doc_id", "label", "bm25_rank"]].copy()
    for name in ranker_names:
        out[name] = score_cols[name]["test"]
    out.to_csv(res / "test_scores.csv.gz", index=False)
    wr = []
    for name in ranker_names[1:]:
        m = get_model(name)
        wr += [{"model": name, "feature": f, "weight": w} for f, w in zip(FEATURE_NAMES, m["w"])]
    pd.DataFrame(wr).to_csv(res / "weights.csv", index=False)
    dev_m = pd.read_csv(res / "metrics_dev.csv").set_index("model")
    ltr = [n for n in ranker_names if n != "bm25"]
    best = dev_m.loc[ltr, C.SELECT_METRIC].idxmax()
    (res / "best_model.json").write_text(json.dumps({"best_ltr_by_dev_" + C.SELECT_METRIC: best, "chosen_C": {k: (None if pd.isna(v) else float(v)) for k, v in chosen.items()}}, indent=2))
    log(f"best LTR model on DEV by {C.SELECT_METRIC}: {best}")

    sig = ST.significance_table(pd.read_csv(res / "per_query_test.csv"))
    sig.to_csv(res / "significance.csv", index=False)
    log("significance vs BM25 (test):\n" + sig.round(4).to_string())
    pd.DataFrame(S["train"][FEATURE_NAMES].corr()).to_csv(res / "feature_correlation_train.csv")
    log("done")


if __name__ == "__main__":
    main()
