"""EXTENSION 3: query-level 5-fold cross-validation over ALL queries (train+dev+test pooled), feature set v1.

Usage: python -I src/run_cv.py [--smoke]     Writes results/ext_cv_*.csv. The official-split results stay the main table.

Protocol per fold k (folds are assigned by QUERY id, never by rows):
  test fold      = fold k queries (never used for fitting, scaling or tuning)
  training folds = the other 4 folds; from them 20 % of the QUERIES (seeded) form an inner validation split
  scaler         = StandardScaler fitted on the training-fold rows only
  C              = chosen on the inner validation split by NDCG@10 (inner-train fits), then the model is REFIT on all
                   training-fold rows with that C and evaluated once on the test fold
Labels note: pooled qrels mix the train qrels (grade 1 only) with dev/test qrels (grades 1 and 2).
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C
import data as D
import metrics as M
import stats as ST
import ltr_runner as LR
from features import FEATURE_NAMES
from util import dirs, Logger, env_info

N_FOLDS = 5
INNER_VAL_FRAC = 0.2
MODELS = ["linreg", "logreg", "ranksvm_all", "ranksvm_sampled", "ranksvm_gapweighted", "ranksvm_dndcgweighted_extra"]


def assign_folds(qids, n_folds, seed):
    """Query-level fold assignment: seeded permutation of the sorted query ids, round-robin (sizes differ by <= 1)."""
    qids = sorted(qids)
    perm = np.random.default_rng(seed).permutation(len(qids))
    fold = {}
    for rank, i in enumerate(perm):
        fold[qids[i]] = rank % n_folds
    return fold


def inner_split(pool_qids, frac, seed):
    """Query-level inner validation split of the training-fold queries -> (inner_train_set, inner_val_set)."""
    pool = sorted(pool_qids)
    perm = np.random.default_rng(seed).permutation(len(pool))
    n_val = int(round(frac * len(pool)))
    val = {pool[i] for i in perm[:n_val]}
    return set(pool) - val, val


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    proc, res = dirs(args.smoke); log = Logger(res / "run_log.txt")
    log(f"=== run_cv smoke={args.smoke} folds={N_FOLDS} | {env_info()}")
    df = pd.read_pickle(proc / "features.pkl"); _, _, _, _, qrels = D.load()
    cr = pd.read_csv(res / "candidate_recall.csv")
    all_q = list(cr["qid"]); assert len(all_q) == len(set(all_q)), "a query appears in two splits"
    qrels_all = {}
    for s in D.SPLITS:
        qrels_all.update({q: d for q, d in qrels[s].items() if q in set(all_q)})
    assert set(qrels_all) == set(all_q)
    fold_of = assign_folds(all_q, N_FOLDS, C.SEED)
    pd.DataFrame({"qid": list(fold_of), "fold": list(fold_of.values())}).to_csv(res / "ext_cv_folds.csv", index=False)
    df["fold"] = df["qid"].map(fold_of)
    log(f"{len(all_q)} queries pooled; fold sizes {np.bincount(list(fold_of.values())).tolist()}")

    sub = lambda qs: {q: qrels_all[q] for q in qs}
    fold_rows, pq_rows = [], []
    for k in range(N_FOLDS):
        seed = C.SEED + 1000 * (k + 1)
        test_q = {q for q, f in fold_of.items() if f == k}; pool_q = set(all_q) - test_q
        in_tr, in_va = inner_split(pool_q, INNER_VAL_FRAC, seed)
        assert not (test_q & pool_q) and not (in_tr & in_va) and (in_tr | in_va) == pool_q
        rows = lambda qs: df[df["qid"].isin(qs)].reset_index(drop=True)
        pool, test, itr, iva = rows(pool_q), rows(test_q), rows(in_tr), rows(in_va)
        sc, (Xpool, Xtest, Xitr, Xiva) = LR.scale(pool, [pool, test, itr, iva], FEATURE_NAMES)   # fit on training folds only
        g = lambda d: (d["label"].values, d["qid"].values, d["bm25_rank"].values)
        (yp, qp, rp), (yte, qte, rte), (yi, qi, ri), (yv, qv, rv) = g(pool), g(test), g(itr), g(iva)
        log(f"--- fold {k}: test {len(test_q)} queries ({len(test)} rows), train folds {len(pool_q)} (inner train {len(in_tr)} / val {len(in_va)})")
        # BM25 baseline on the test fold
        base = M.evaluate_scores(qte, yte, rte, test["bm25_comb"].values, sub(test_q))
        for x in base:
            x.update(model="bm25", fold=k)
        pq_rows += base
        fold_rows.append({"fold": k, "model": "bm25", "C": np.nan, **M.summarize(base)})
        for name in MODELS:
            grid, _, best_c = LR.tune(name, Xitr, yi, qi, ri, Xiva, yv, qv, rv, sub(in_va), seed)
            m = LR.fit_one(name, best_c, Xpool, yp, qp, rp, seed)       # refit on all training folds
            r = LR.per_query_rows(m, Xtest, yte, qte, rte, sub(test_q))
            for x in r:
                x.update(model=name, fold=k)
            pq_rows += r
            fold_rows.append({"fold": k, "model": name, "C": best_c, **M.summarize(r), "train_s": m["train_s"]})
            log(f"  {name:28s} C={best_c!s:7s} fold-{k} test NDCG@10={fold_rows[-1]['NDCG@10']:.4f} MAP={fold_rows[-1]['MAP']:.4f}")
    fr = pd.DataFrame(fold_rows); fr.to_csv(res / "ext_cv_fold_metrics.csv", index=False)
    pqd = pd.DataFrame(pq_rows); pqd.to_csv(res / "ext_cv_per_query.csv", index=False)
    assert pqd.groupby("model")["qid"].nunique().eq(len(all_q)).all(), "every query must be scored exactly once per model"

    metrics = ["MAP", "NDCG@5", "NDCG@10", "pair_acc_micro"]
    summ = fr.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index(); summ.to_csv(res / "ext_cv_summary.csv", index=False)
    pooled = pqd.groupby("model", sort=False).agg(MAP_pooled=("ap", "mean"), NDCG10_pooled=("ndcg10", "mean")).reset_index()
    pooled.to_csv(res / "ext_cv_pooled_oof.csv", index=False)
    sig = ST.significance_table(pqd.drop(columns=["fold"]))
    sig.to_csv(res / "ext_cv_significance.csv", index=False)
    log("CV summary (mean over folds):\n" + summ.round(4).to_string())
    log("CV significance (pooled out-of-fold queries):\n" + sig[["model", "metric", "mean_delta", "ci95_lo", "ci95_hi", "p_ttest", "p_wilcoxon"]].round(4).to_string())


if __name__ == "__main__":
    main()
