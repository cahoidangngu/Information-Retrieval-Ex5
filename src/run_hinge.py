"""EXTENSION 2: standard-hinge Ranking SVM (dual CD) next to the squared-hinge one. Feature set v1, official split.

Usage: python -I src/run_hinge.py [--smoke]    Writes results/ext_hinge_*.csv (existing results are not touched).
C grid tuned on DEV only; test evaluated once with the chosen C.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C
import data as D
import metrics as M
import models as MD
import stats as ST
import ltr_runner as LR
from features import FEATURE_NAMES
from util import dirs, Logger, env_info

HINGE = ["ranksvm_hinge_all", "ranksvm_hinge_sampled"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    proc, res = dirs(args.smoke); log = Logger(res / "run_log.txt")
    log(f"=== run_hinge smoke={args.smoke} | {env_info()}")
    df = pd.read_pickle(proc / "features.pkl"); _, _, _, _, qrels = D.load()
    cr = pd.read_csv(res / "candidate_recall.csv")
    qrels = {s: {q: d for q, d in qrels[s].items() if q in set(cr.loc[cr["split"] == s, "qid"])} for s in D.SPLITS}
    S = {s: df[df["split"] == s].reset_index(drop=True) for s in D.SPLITS}
    sc, (Xtr, Xdev, Xte) = LR.scale(S["train"], [S["train"], S["dev"], S["test"]], FEATURE_NAMES)
    cols = lambda s: (S[s]["label"].values, S[s]["qid"].values, S[s]["bm25_rank"].values)
    (ytr, qtr, rtr), (yd, qd, rd), (yt, qt, rt) = cols("train"), cols("dev"), cols("test")

    grids, rows, pq = [], [], []
    for name in HINGE:
        grid, models, best_c = LR.tune(name, Xtr, ytr, qtr, rtr, Xdev, yd, qd, rd, qrels["dev"], C.SEED, log)
        grids.append(grid); m = LR.get(models, best_c)
        r = LR.per_query_rows(m, Xte, yt, qt, rt, qrels["test"])
        for x in r:
            x["model"] = name
        pq += r
        rows.append({"model": name, "loss": "hinge (dual CD)", "C": best_c, **M.summarize(r),
                     "train_s": m["train_s"], "predict_s": MD.predict_time(m, Xte), "peak_mb": m["peak_mb"],
                     "n_pairs": m["n_pairs"], "converged": not m["no_converge"], "n_iter": m.get("n_iter", np.nan)})
        for f, w in zip(FEATURE_NAMES, m["w"]):
            rows[-1][f"w_{f}"] = w
    pd.concat(grids).to_csv(res / "ext_hinge_tuning_dev.csv", index=False)
    met = pd.DataFrame(rows); met.to_csv(res / "ext_hinge_metrics_test.csv", index=False)
    pqd = pd.DataFrame(pq); pqd.to_csv(res / "ext_hinge_per_query_test.csv", index=False)

    # significance: (i) vs BM25, (ii) vs the squared-hinge counterpart, using the main run's per-query file
    main_pq = pd.read_csv(res / "per_query_test.csv")
    base = main_pq[main_pq["model"] == "bm25"]
    sig = ST.significance_table(pd.concat([base, pqd], ignore_index=True)); sig["comparison"] = "vs BM25"
    outs = [sig]
    for h, sqr in (("ranksvm_hinge_all", "ranksvm_all"), ("ranksvm_hinge_sampled", "ranksvm_sampled")):
        sub = pd.concat([main_pq[main_pq["model"] == sqr].assign(model="bm25"), pqd[pqd["model"] == h]], ignore_index=True)
        s = ST.significance_table(sub); s["comparison"] = f"vs {sqr} (squared hinge)"; outs.append(s)
    pd.concat(outs).to_csv(res / "ext_hinge_significance.csv", index=False)
    log("hinge test metrics:\n" + met[["model", "C", "MAP", "NDCG@5", "NDCG@10", "pair_acc_micro", "train_s", "converged"]].to_string())


if __name__ == "__main__":
    main()
