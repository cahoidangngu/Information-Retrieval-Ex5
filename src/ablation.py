"""Ablations that back the discussion questions (NOT used for model selection or tuning).

C is fixed to the value chosen on DEV for the sampled RankSVM; dev AND test are reported.
  A) within-query pairs (reference)   B) cross-query pairs (pitfall)   C) with intercept   D) without intercept
Usage: python -I src/ablation.py [--smoke]
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import config as C
import data as D
import metrics as M
import pairs as P
from features import FEATURE_NAMES
from util import dirs, Logger


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    proc, res = dirs(args.smoke); log = Logger(res / "run_log.txt")
    df = pd.read_pickle(proc / "features.pkl"); _, _, _, _, qrels = D.load()
    cr = pd.read_csv(res / "candidate_recall.csv")
    qrels = {s: {q: d for q, d in qrels[s].items() if q in set(cr.loc[cr["split"] == s, "qid"])} for s in D.SPLITS}
    S = {s: df[df["split"] == s].reset_index(drop=True) for s in D.SPLITS}
    sc = StandardScaler().fit(S["train"][FEATURE_NAMES].values)
    X = {s: sc.transform(S[s][FEATURE_NAMES].values) for s in S}
    y, q, rk = ({s: S[s][c].values for s in S} for c in ("label", "qid", "bm25_rank"))
    Cv = json.loads((res / "best_model.json").read_text())["chosen_C"]["ranksvm_sampled"]
    rng = np.random.default_rng(C.SEED + 100)

    Xd, yd, _, _ = P.build_pairs(X["train"], y["train"], q["train"], rk["train"], "plain", rng, C.PAIR_CAP_PER_QUERY)
    n = len(yd)
    # cross-query pairs: random (pos, neg) rows drawn from DIFFERENT queries, same number of pairs, balanced directions
    pos = np.flatnonzero(y["train"] > 0); neg = np.flatnonzero(y["train"] == 0)
    I = rng.choice(pos, n); J = rng.choice(neg, n)
    bad = q["train"][I] == q["train"][J]
    while bad.any():
        J[bad] = rng.choice(neg, bad.sum()); bad = q["train"][I] == q["train"][J]
    assert np.all(q["train"][I] != q["train"][J])
    sign = np.where(rng.permutation(n) < n // 2, -1.0, 1.0)
    Xc, yc = (X["train"][I] - X["train"][J]) * sign[:, None], sign.astype(int)

    rows = []
    for name, (A, b, icpt) in {"A within-query pairs, no intercept (reference)": (Xd, yd, False),
                               "B cross-query pairs, no intercept (pitfall)": (Xc, yc, False),
                               "C within-query pairs, WITH intercept": (Xd, yd, True)}.items():
        m = LinearSVC(C=Cv, fit_intercept=icpt, dual=False, tol=1e-4, max_iter=1000, random_state=C.SEED).fit(A, b)
        w = m.coef_.ravel(); r = {"variant": name, "C": Cv, "n_pairs": len(b), "intercept": float(m.intercept_[0]) if icpt else 0.0}
        for s in ("dev", "test"):
            sm = M.summarize(M.evaluate_scores(q[s], y[s], rk[s], X[s] @ w, qrels[s]))
            r[f"{s}_NDCG@10"], r[f"{s}_MAP"], r[f"{s}_pair_acc"] = sm["NDCG@10"], sm["MAP"], sm["pair_acc_micro"]
        r["train_pair_acc_on_own_pairs"] = float((np.sign(A @ w + (m.intercept_[0] if icpt else 0)) == b).mean())
        rows.append(r)
    # D) a single-class pair set (all +1, i.e. unbalanced directions) cannot even be fitted without an intercept trick
    try:
        LinearSVC(fit_intercept=False, dual=False).fit(Xd * yd[:, None], np.ones(len(yd), int))   # all pairs oriented as +1
        unb = "fitted"
    except ValueError as e:
        unb = "fails: " + str(e)[:90]
    out = pd.DataFrame(rows); out["unbalanced_directions_(all +1)"] = unb
    out.to_csv(res / "ablation.csv", index=False)
    log("ablation:\n" + out.round(4).to_string())


if __name__ == "__main__":
    main()
