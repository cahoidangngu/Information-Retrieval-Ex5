"""Shared train / tune / evaluate helpers for the extension experiments (hinge, feature sets, CV).

The original pipeline (run_models.py) is NOT modified; the helpers here reproduce its logic (same pair
construction, same seeds per variant) so that feature set v1 on the official split gives identical numbers
(checked by an assertion in run_featuresets.py).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config as C
import metrics as M
import models as MD
import pairs as P

# name -> (pair mode, 'all'|'sampled', solver, seed offset). Offsets of the first four equal run_models.py.
VARIANTS = {
    "ranksvm_all": ("plain", "all", "sq", 0),
    "ranksvm_sampled": ("plain", "sampled", "sq", 1),
    "ranksvm_gapweighted": ("gap", "all", "sq", 2),
    "ranksvm_dndcgweighted_extra": ("dndcg", "all", "sq", 3),
    "ranksvm_hinge_all": ("plain", "all", "hinge", 10),
    "ranksvm_hinge_sampled": ("plain", "sampled", "hinge", 11),
}
POINTWISE = ("linreg", "logreg")


def scale(train_df, frames, names):
    """StandardScaler fitted on train_df ONLY; returns (scaler, [X for each frame])."""
    sc = StandardScaler().fit(train_df[names].values)
    assert np.allclose(sc.mean_, train_df[names].values.mean(axis=0))
    return sc, [sc.transform(f[names].values) for f in frames]


def make_pairs(name, X, y, q, rk, seed):
    mode, kind, _, off = VARIANTS[name]
    fits = P.estimate_memory_gb(P.count_pairs(y, q), X.shape[1]) <= C.PAIR_MEM_BUDGET_GB
    cap = None if (kind == "all" and fits) else C.PAIR_CAP_PER_QUERY
    return P.build_pairs(X, y, q, rk, mode, np.random.default_rng(seed + off), cap)


def fit_one(name, Cval, X, y, q, rk, seed, pairs=None):
    """Fit one model; for SVM variants `pairs` may be a pre-built (Xd, yd, w, _) tuple."""
    if name == "linreg":
        return MD.fit_linreg(X, y.astype(float))
    if name == "logreg":
        return MD.fit_logreg(X, (y >= 1).astype(int), Cval)
    Xd, yd, w, _ = pairs if pairs is not None else make_pairs(name, X, y, q, rk, seed)
    fit = MD.fit_ranksvm_hinge if VARIANTS[name][2] == "hinge" else MD.fit_ranksvm
    m = fit(Xd, yd, w, Cval)
    m["n_pairs"] = len(yd)
    return m


def per_query_rows(m, X, y, q, rk, qrels):
    return M.evaluate_scores(q, y, rk, MD.score(m, X), qrels)


def tune(name, Xtr, ytr, qtr, rktr, Xva, yva, qva, rkva, qrels_va, seed, log=None):
    """Grid over C on the validation split. Returns (grid DataFrame, {C: model}). Linear regression has no C."""
    grid, models = [], {}
    pairs = None if name in POINTWISE else make_pairs(name, Xtr, ytr, qtr, rktr, seed)
    for Cv in ([np.nan] if name == "linreg" else C.C_GRID):
        m = fit_one(name, Cv, Xtr, ytr, qtr, rktr, seed, pairs)
        sm = M.summarize(per_query_rows(m, Xva, yva, qva, rkva, qrels_va))
        grid.append({"model": name, "C": Cv, "MAP": sm["MAP"], "NDCG@5": sm["NDCG@5"], "NDCG@10": sm["NDCG@10"],
                     "pair_acc_micro": sm["pair_acc_micro"], "train_s": m["train_s"], "peak_mb": m["peak_mb"],
                     "no_converge": m["no_converge"], "n_pairs": m.get("n_pairs", np.nan)})
        models[Cv if not pd.isna(Cv) else "nan"] = m
        if log:
            log(f"  {name:28s} C={Cv!s:7s} val NDCG@10={sm['NDCG@10']:.4f} MAP={sm['MAP']:.4f} train={m['train_s']:.1f}s conv_warn={m['no_converge']}")
    grid = pd.DataFrame(grid)
    best_c = grid.loc[grid[C.SELECT_METRIC].idxmax(), "C"]      # first max -> smallest C on ties
    return grid, models, best_c


def get(models, Cv):
    return models["nan" if pd.isna(Cv) else Cv]


def vif(df_features):
    """VIF_j = diag of the inverse correlation matrix (label-free)."""
    corr = np.corrcoef(df_features.values, rowvar=False)
    return pd.Series(np.diag(np.linalg.inv(corr)), index=df_features.columns)


def vif_prune(df_features, threshold=10.0):
    """Drop the feature with the largest VIF, recompute, repeat until all VIF < threshold.
    Returns (kept names in original order, DataFrame of drops [feature, vif_at_drop])."""
    cols = list(df_features.columns); drops = []
    while len(cols) > 1:
        v = vif(df_features[cols])
        if v.max() < threshold:
            break
        worst = v.idxmax(); drops.append({"feature": worst, "vif_at_drop": float(v.max())}); cols.remove(worst)
    return cols, pd.DataFrame(drops, columns=["feature", "vif_at_drop"])
