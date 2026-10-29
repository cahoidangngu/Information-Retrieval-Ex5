"""Paired significance tests of per-query differences vs the BM25 baseline."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import config as C


def paired_stats(delta, rng, n_boot=C.N_BOOT):
    d = np.asarray(delta, dtype=float); n = d.size
    sd = d.std(ddof=1)
    t = stats.ttest_rel(d, np.zeros(n)) if sd > 0 else None
    nz = int((d != 0).sum())
    try:
        w = stats.wilcoxon(d, zero_method="wilcox")      # drops zero differences
        wp = float(w.pvalue)
    except ValueError:
        wp = np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_queries": n, "mean_delta": d.mean(), "ci95_lo": lo, "ci95_hi": hi,
            "t_stat": float(t.statistic) if t else np.nan, "p_ttest": float(t.pvalue) if t else np.nan,
            "p_wilcoxon": wp, "cohen_dz": d.mean() / sd if sd > 0 else np.nan,
            "n_improved": int((d > 0).sum()), "n_harmed": int((d < 0).sum()),
            "n_unchanged": int((d == 0).sum()), "n_nonzero": nz}


def holm(p):
    p = np.asarray(p, dtype=float); order = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def significance_table(per_query, baseline="bm25", metrics=("ndcg10", "ap"), seed=C.SEED):
    """per_query: DataFrame [model, qid, <metrics>] on the TEST queries."""
    rng = np.random.default_rng(seed)
    base = per_query[per_query["model"] == baseline].set_index("qid")
    rows = []
    for m in metrics:
        sub = []
        for model, g in per_query.groupby("model", sort=False):
            if model == baseline:
                continue
            g = g.set_index("qid").loc[base.index]
            r = paired_stats((g[m] - base[m]).values, rng); r.update({"model": model, "metric": m})
            sub.append(r)
        pt = holm([r["p_ttest"] for r in sub])
        pw = holm([r["p_wilcoxon"] for r in sub])
        for r, a, b in zip(sub, pt, pw):
            r["p_ttest_holm"] = a; r["p_wilcoxon_holm"] = b
        rows += sub
    cols = ["model", "metric", "n_queries", "mean_delta", "ci95_lo", "ci95_hi", "p_ttest", "p_ttest_holm",
            "p_wilcoxon", "p_wilcoxon_holm", "cohen_dz", "n_improved", "n_harmed", "n_unchanged", "n_nonzero", "t_stat"]
    return pd.DataFrame(rows)[cols]
