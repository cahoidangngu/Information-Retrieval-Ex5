"""Analyses a)-d): feature weights, improved/harmed queries, pairwise-vs-NDCG, figures.

Usage: python -I src/analysis.py [--smoke]
Everything is computed from files written by run_features.py / run_models.py.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import config as C
import data as D
import metrics as M
from features import FEATURE_NAMES
from util import dirs, Logger

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#6b6b6b"       # Okabe-Ito, colour-blind safe
LABEL = {"bm25": "BM25", "linreg": "Linear regression", "logreg": "Logistic regression",
         "ranksvm_all": "RankSVM all-pairs", "ranksvm_sampled": "RankSVM sampled",
         "ranksvm_gapweighted": "RankSVM gap-weighted", "ranksvm_dndcgweighted_extra": "RankSVM dNDCG-weighted (extra)"}
# a-priori expected sign of each feature's effect on relevance (stated BEFORE looking at weights);
# 0 = no a-priori expectation
EXPECTED = {"bm25_title": 1, "bm25_body": 1, "bm25_comb": 1, "tfidf_cos_title": 1, "tfidf_cos_body": 1,
            "lm_dirichlet": 1, "query_len": 0, "doc_len": 0, "title_len": 0, "matched_terms": 1,
            "coverage_ratio": 1, "sum_idf_matched": 1, "max_idf_matched": 1, "bm25_rank": -1}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)


def md(df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                                      else ("" if (isinstance(v, float) and pd.isna(v)) else str(v)) for v in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    proc, res = dirs(args.smoke); log = Logger(res / "run_log.txt"); fig = res / "figures"
    log("=== analysis")
    best = json.loads((res / "best_model.json").read_text())
    best_name = [v for k, v in best.items() if k.startswith("best_ltr")][0]
    df = pd.read_pickle(proc / "features.pkl")
    sc = pd.read_csv(res / "scaler_stats.csv")
    W = pd.read_csv(res / "weights.csv").pivot(index="feature", columns="model", values="weight").loc[FEATURE_NAMES]
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"].reset_index(drop=True)
    ts = pd.read_csv(res / "test_scores.csv.gz")
    assert (ts["qid"].values == te["qid"].values).all() and (ts["doc_id"].values == te["doc_id"].values).all()
    pq = pd.read_csv(res / "per_query_test.csv")
    cr = pd.read_csv(res / "candidate_recall.csv"); cr_te = cr[cr["split"] == "test"].set_index("qid")
    _, _, _, queries, qrels = D.load()
    doc_ids, titles, _, _, _ = D.load()
    title_of = dict(zip(doc_ids, titles))

    # ------------------------------------------------------------------ a) feature weights
    corr = tr[FEATURE_NAMES].corr()
    vif = pd.Series(np.diag(np.linalg.inv(corr.values)), index=FEATURE_NAMES)
    uni = pd.Series({f: stats.spearmanr(tr[f], tr["label"])[0] for f in FEATURE_NAMES})
    # within-query version: correlation after centring each feature and label inside its query
    cen = tr.groupby("qid")[FEATURE_NAMES + ["label"]].transform(lambda x: x - x.mean())
    uni_wq = pd.Series({f: stats.spearmanr(cen[f], cen["label"])[0] if cen[f].std() > 0 else np.nan for f in FEATURE_NAMES})
    wt = pd.DataFrame({"feature": FEATURE_NAMES,
                       "ranksvm_all": W["ranksvm_all"].values, "ranksvm_sampled": W["ranksvm_sampled"].values,
                       "linreg": W["linreg"].values, "logreg": W["logreg"].values,
                       "expected_sign": [EXPECTED[f] for f in FEATURE_NAMES],
                       "spearman_with_label_train": uni.values, "spearman_within_query_train": uni_wq.values,
                       "VIF": vif.values})
    for m in ("ranksvm_all", "linreg", "logreg"):
        wt[f"sign_ok_{m}"] = [("n/a" if e == 0 else ("yes" if np.sign(w) == e else "NO")) for e, w in zip(wt["expected_sign"], wt[m])]
    wt.sort_values("ranksvm_all", key=np.abs, ascending=False).to_csv(res / "weights_table.csv", index=False)
    corr.to_csv(res / "feature_correlation_train.csv")

    fig1, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    order = wt.reindex(wt["ranksvm_all"].abs().sort_values().index)["feature"].tolist()
    for ax, m in zip(axes, ["ranksvm_all", "linreg", "logreg"]):
        v = W.loc[order, m]
        ax.barh(order, v.values, color=[BLUE if x >= 0 else ORANGE for x in v.values], height=0.65)
        ax.axvline(0, color="#333", lw=0.8); ax.set_title(LABEL[m], fontsize=11, loc="left"); style(ax)
        ax.set_xlabel("weight on standardised feature")
    fig1.suptitle("Learned weights (blue = positive, orange = negative). Magnitudes comparable only within a panel.", fontsize=10, x=0.01, ha="left")
    fig1.tight_layout(); fig1.savefig(fig / "feature_weights.png", dpi=160); plt.close(fig1)

    # ------------------------------------------------------------------ oracle headroom (analysis only)
    rows_or = M.evaluate_scores(te["qid"].values, te["label"].values, te["bm25_rank"].values,
                                te["label"].values.astype(float), {q: qrels["test"][q] for q in cr_te.index})
    orc = pd.DataFrame(rows_or)
    pd.DataFrame([{"ranker": "oracle re-ranking of the same candidates (uses test labels; upper bound only)",
                   "MAP": orc["ap"].mean(), "NDCG@10": orc["ndcg10"].mean()}]).to_csv(res / "oracle_upper_bound.csv", index=False)

    # ------------------------------------------------------------------ b) per-query improved / harmed
    base = pq[pq["model"] == "bm25"].set_index("qid"); lt = pq[pq["model"] == best_name].set_index("qid")
    qd = pd.DataFrame({"query": [queries[q] for q in base.index], "n_rel_total": base["n_rel_total"],
                       "n_cand": base["n_cand"], "n_rel_in_cand": base["n_rel_in_cand"],
                       "recall_at_100": cr_te.loc[base.index, "recall_at_100"],
                       "ndcg10_bm25": base["ndcg10"], "ndcg10_ltr": lt["ndcg10"],
                       "delta_ndcg10": lt["ndcg10"] - base["ndcg10"],
                       "delta_pairacc": lt["pair_acc"] - base["pair_acc"], "delta_ap": lt["ap"] - base["ap"]})
    qd.index.name = "qid"; qd.to_csv(res / "query_deltas_best_model.csv")
    imp = qd.sort_values("delta_ndcg10", ascending=False).head(10)
    har = qd.sort_values("delta_ndcg10", ascending=True).head(10)
    imp.to_csv(res / "top10_improved.csv"); har.to_csv(res / "top10_harmed.csv")

    w_best = W[best_name].values
    Xs = (te[FEATURE_NAMES].values - sc["train_mean"].values) / sc["train_std"].values
    contrib = Xs * w_best[None, :]
    te["ltr_score"] = ts[best_name].values
    blocks, ev_rows = [], []
    for tag, tab in (("IMPROVED", imp), ("HARMED", har)):
        for qid in tab.index:
            idx = np.flatnonzero(te["qid"].values == qid)
            g = te.iloc[idx]
            ordr = M.rank_order(g["ltr_score"].values, g["bm25_rank"].values)
            ltr_rank = np.empty(len(g), dtype=int); ltr_rank[ordr] = np.arange(1, len(g) + 1)
            bm_rank = g["bm25_rank"].values.astype(int)
            ent = np.flatnonzero((ltr_rank <= 10) & (bm_rank > 10)); lef = np.flatnonzero((ltr_rank > 10) & (bm_rank <= 10))
            diff = contrib[idx][ent].mean(axis=0) - contrib[idx][lef].mean(axis=0) if len(ent) and len(lef) else np.zeros(len(FEATURE_NAMES))
            top = np.argsort(-np.abs(diff))[:3]
            fmt = lambda i: f"[rel={g['label'].values[i]}] BM25#{bm_rank[i]}->LTR#{ltr_rank[i]} \"{title_of[g['doc_id'].values[i]][:70]}\""
            r = tab.loc[qid]
            relidx = np.flatnonzero(g["label"].values >= 1)
            rel_moves = ", ".join(f"rel={g['label'].values[i]} BM25#{bm_rank[i]}->LTR#{ltr_rank[i]}" for i in relidx[np.argsort(bm_rank[relidx])])
            ev_rows.append({"qid": qid, "group": tag, "query": r["query"], "delta_ndcg10": r["delta_ndcg10"],
                            "n_entered": len(ent), "n_entered_relevant": int((g["label"].values[ent] >= 1).sum()),
                            "n_left": len(lef), "n_left_relevant": int((g["label"].values[lef] >= 1).sum()),
                            "top_feature_1": FEATURE_NAMES[top[0]], "top_feature_2": FEATURE_NAMES[top[1]], "top_feature_3": FEATURE_NAMES[top[2]],
                            "relevant_candidate_rank_moves": rel_moves})
            blocks.append(
                f"### {tag} {qid}: \"{r['query']}\"\n"
                f"NDCG@10 {r['ndcg10_bm25']:.3f} -> {r['ndcg10_ltr']:.3f} (delta {r['delta_ndcg10']:+.3f}); "
                f"relevant total={int(r['n_rel_total'])}, in candidates={int(r['n_rel_in_cand'])}, "
                f"recall@100={r['recall_at_100']:.3f}, candidates={int(r['n_cand'])}\n\n"
                f"Entered LTR top-10 ({len(ent)}):\n" + "\n".join("- " + fmt(i) for i in ent) +
                f"\n\nLeft top-10 ({len(lef)}):\n" + "\n".join("- " + fmt(i) for i in lef) +
                "\n\nFeatures with the largest score contribution difference (entered minus left, w*z): " +
                ", ".join(f"{FEATURE_NAMES[i]} ({diff[i]:+.2f})" for i in top) + "\n")
    pd.DataFrame(ev_rows).to_csv(res / "query_evidence.csv", index=False)
    (res / "query_evidence.md").write_text(f"# Per-query evidence for best LTR model on DEV: {best_name}\n\n" + "\n".join(blocks), encoding="utf8")

    # group analysis of deltas
    qd["recall_bucket"] = pd.cut(qd["recall_at_100"], [-0.01, 0.0, 0.1, 0.3, 1.0], labels=["0", "(0,0.1]", "(0.1,0.3]", ">0.3"])
    grp = qd.groupby("recall_bucket", observed=True).agg(n_queries=("delta_ndcg10", "size"), mean_ndcg10_bm25=("ndcg10_bm25", "mean"),
                                                         mean_delta_ndcg10=("delta_ndcg10", "mean"),
                                                         n_improved=("delta_ndcg10", lambda x: int((x > 0).sum())),
                                                         n_harmed=("delta_ndcg10", lambda x: int((x < 0).sum()))).reset_index()
    grp.to_csv(res / "query_groups_by_recall.csv", index=False)
    nz = qd[qd["delta_ndcg10"] != 0]
    extra = {"n_queries_changed": int(len(nz)), "n_queries_unchanged": int((qd["delta_ndcg10"] == 0).sum()),
             "unchanged_with_zero_recall": int(((qd["delta_ndcg10"] == 0) & (qd["n_rel_in_cand"] == 0)).sum()),
             "spearman_delta_vs_bm25_ndcg": float(stats.spearmanr(qd["delta_ndcg10"], qd["ndcg10_bm25"])[0]),
             "spearman_delta_ndcg_vs_delta_pairacc": float(stats.spearmanr(qd["delta_ndcg10"], qd["delta_pairacc"], nan_policy="omit")[0]),
             "n_queries_ndcg_up_pairacc_down": int(((qd["delta_ndcg10"] > 0) & (qd["delta_pairacc"] < 0)).sum()),
             "n_queries_ndcg_down_pairacc_up": int(((qd["delta_ndcg10"] < 0) & (qd["delta_pairacc"] > 0)).sum()),
             "best_model": best_name}

    # ------------------------------------------------------------------ c) pairwise accuracy vs NDCG
    # restricted pair accuracy: pairs where at least one document is inside the BM25 top-10
    names = [n for n in LABEL if n in ts.columns]
    share_rows = []
    tot_pairs = 0; tot_top = 0
    acc = {n: [0.0, 0, 0.0, 0] for n in names}      # correct_top, n_top, correct_deep, n_deep
    for qid, g in ts.groupby("qid", sort=False):
        y = g["label"].values; r = g["bm25_rank"].values
        hi = y[:, None] > y[None, :]
        intop = (r[:, None] <= 10) | (r[None, :] <= 10)
        tot_pairs += hi.sum(); tot_top += (hi & intop).sum()
        for n in names:
            s = g[n].values; d = s[:, None] - s[None, :]
            ok = np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))
            acc[n][0] += (ok * (hi & intop)).sum(); acc[n][1] += (hi & intop).sum()
            acc[n][2] += (ok * (hi & ~intop)).sum(); acc[n][3] += (hi & ~intop).sum()
    mt = pd.read_csv(res / "metrics_test.csv").set_index("model")
    pa = pd.DataFrame([{"model": n, "NDCG@10": mt.loc[n, "NDCG@10"], "pair_acc_all_pairs": mt.loc[n, "pair_acc_micro"],
                        "pair_acc_pairs_with_a_BM25_top10_doc": acc[n][0] / acc[n][1],
                        "pair_acc_pairs_entirely_below_BM25_top10": acc[n][2] / acc[n][3]} for n in names])
    pa.to_csv(res / "pairacc_vs_ndcg.csv", index=False)
    extra["share_of_valid_pairs_with_a_bm25_top10_doc"] = float(tot_top / tot_pairs)
    (res / "analysis_extra.json").write_text(json.dumps(extra, indent=2))

    # ------------------------------------------------------------------ figures
    fig2, ax = plt.subplots(figsize=(7, 4)); d = qd["delta_ndcg10"]
    ax.hist(d[d != 0], bins=40, color=BLUE, edgecolor="white")
    ax.axvline(0, color="#333", lw=0.8); ax.set_xlabel(f"per-query delta NDCG@10 ({LABEL[best_name]} - BM25), test"); ax.set_ylabel("queries")
    ax.set_title(f"{int((d == 0).sum())} of {len(d)} queries unchanged (not shown)", fontsize=10, loc="left"); style(ax); ax.grid(axis="y", color="#e5e5e5")
    fig2.tight_layout(); fig2.savefig(fig / "delta_ndcg10_hist.png", dpi=160); plt.close(fig2)

    tun = pd.read_csv(res / "tuning_dev.csv")
    fig3, ax = plt.subplots(figsize=(7, 4.2))
    cols = {"logreg": ORANGE, "ranksvm_all": BLUE, "ranksvm_sampled": "#009E73", "ranksvm_dndcgweighted_extra": "#CC79A7"}
    for n, c in cols.items():
        t = tun[tun["model"] == n]; ax.plot(t["C"], t["NDCG@10"], marker="o", color=c, lw=1.8, label=LABEL[n])
    dm = pd.read_csv(res / "metrics_dev.csv").set_index("model")
    ax.axhline(dm.loc["bm25", "NDCG@10"], color=GREY, ls="--", lw=1.2, label="BM25 (dev)")
    ax.set_xscale("log"); ax.set_xlabel("C"); ax.set_ylabel("dev NDCG@10"); style(ax); ax.grid(axis="y", color="#e5e5e5")
    ax.legend(frameon=False, fontsize=8); fig3.tight_layout(); fig3.savefig(fig / "tuning_curves_dev.png", dpi=160); plt.close(fig3)

    fig4, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in pa.iterrows():
        ax.scatter(r["pair_acc_all_pairs"], r["NDCG@10"], color=GREY if r["model"] == "bm25" else BLUE, s=45, zorder=3)
        if r["model"] == "ranksvm_gapweighted":
            continue                       # identical to all-pairs (train grade gap is always 1); labelled once
        lab = "RankSVM all-pairs = gap-weighted" if r["model"] == "ranksvm_all" else LABEL[r["model"]]
        ax.annotate(lab, (r["pair_acc_all_pairs"], r["NDCG@10"]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("pairwise accuracy (all valid candidate pairs), test"); ax.set_ylabel("NDCG@10, test"); style(ax); ax.grid(axis="y", color="#e5e5e5")
    fig4.tight_layout(); fig4.savefig(fig / "pairacc_vs_ndcg10.png", dpi=160); plt.close(fig4)
    log("analysis done; best model " + best_name)


if __name__ == "__main__":
    main()
