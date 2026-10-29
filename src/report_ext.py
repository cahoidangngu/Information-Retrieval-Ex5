"""Section 9 of report.md (extensions). All numbers come from results/ext_*.csv; every directional statement is generated
from a fixed rule (see `verdict`) instead of being typed in, so it cannot contradict the data.

Rule used everywhere: a difference is called an improvement/degradation ONLY if the paired t-test AND the Wilcoxon test
both have raw p < 0.05 AND the 95 % bootstrap CI of the mean difference excludes 0. Holm-adjusted p-values are shown next
to it and mentioned when they disagree.
"""
import numpy as np
import pandas as pd
from analysis import md, LABEL
import config as C

R = C.RESULTS_DIR
LAB = dict(LABEL, ranksvm_hinge_all="RankSVM all-pairs, standard hinge", ranksvm_hinge_sampled="RankSVM sampled, standard hinge")
f4 = lambda x: f"{x:.4f}"


def verdict(r):
    ok = (r["p_ttest"] < 0.05) and (r["p_wilcoxon"] < 0.05) and (r["ci95_lo"] > 0 or r["ci95_hi"] < 0)
    if not ok:
        return "no supported difference"
    return "improvement supported" if r["mean_delta"] > 0 else "degradation supported"


def with_verdict(sig):
    s = sig.copy(); s["verdict"] = s.apply(verdict, axis=1)
    return s


def sig_table(sig, cols_first):
    s = with_verdict(sig)
    keep = cols_first + ["mean_delta", "ci95_lo", "ci95_hi", "p_ttest", "p_wilcoxon", "p_ttest_holm", "cohen_dz", "verdict"]
    return md(s[keep], "{:.4f}")


def pf(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def pm(m, s):
    return f"{m:.4f} ± {s:.4f}"


def ext_section(main_t, main_dev, main_tun, best_name):
    out = []
    # =============================================================== 9.0 overview
    out.append("""## 9. Extensions (added after review)

Everything in sections 0-8 is unchanged (the original pipeline and its files are untouched). The extensions below are separate scripts that write `results/ext_*.csv`.
**Decision rule for any claim in this section:** a difference counts as an improvement (or degradation) only if the paired t-test *and* the Wilcoxon test both give raw p < 0.05 *and* the 95 % bootstrap CI of the mean per-query difference excludes 0. Holm-adjusted p-values are always shown; where they disagree with the raw rule this is said explicitly. Nothing in this section was tuned on test data: C is always chosen on a validation split (dev, or the inner validation split in the cross-validation).
""")

    # =============================================================== 9.1 hinge
    ht = pd.read_csv(R / "ext_hinge_metrics_test.csv").set_index("model")
    htun = pd.read_csv(R / "ext_hinge_tuning_dev.csv")
    hsig = pd.read_csv(R / "ext_hinge_significance.csv")
    rows = []
    for name, loss in (("ranksvm_all", "squared hinge (sklearn default)"), ("ranksvm_hinge_all", "standard hinge"),
                       ("ranksvm_sampled", "squared hinge (sklearn default)"), ("ranksvm_hinge_sampled", "standard hinge")):
        if name in ht.index:
            r = ht.loc[name]; dev = htun[(htun.model == name) & (htun.C == r["C"])].iloc[0]
            rows.append({"model": LAB[name], "loss": loss, "C (dev)": f"{r['C']:g}", "dev NDCG@10": dev["NDCG@10"], "test MAP": r["MAP"],
                         "test NDCG@5": r["NDCG@5"], "test NDCG@10": r["NDCG@10"], "test pair acc": r["pair_acc_micro"],
                         "train s": r["train_s"], "converged": "yes" if r["converged"] else "NO (max_iter reached)"})
        else:
            r = main_t.loc[name]; dev = main_dev.loc[name]
            rows.append({"model": LAB[name], "loss": loss, "C (dev)": f"{r['C']:g}", "dev NDCG@10": dev["NDCG@10"], "test MAP": r["MAP"],
                         "test NDCG@5": r["NDCG@5"], "test NDCG@10": r["NDCG@10"], "test pair acc": r["pair_acc_micro"],
                         "train s": r["train_s"], "converged": "yes"})
    nconv = int(htun["no_converge"].sum()); nfit = len(htun)
    bad_c = htun[htun["C"] == 100].set_index("model")["NDCG@10"]
    best_dev = htun.groupby("model")["NDCG@10"].max()
    out.append(f"""### 9.1 Standard-hinge vs squared-hinge Ranking SVM (feature set v1, official split)

Which loss matches the course slides? The Ranking SVM objective in `Chap_8.pdf` (slides 91-92) is 1/2||w||^2 + C * sum max(0, 1 - w^T(x_i - x_j)), i.e. the **standard (L1) hinge**; the code snippet on slide 109 is `LinearSVC(C=1.0, fit_intercept=False)`, which in scikit-learn is the **squared** hinge (default `loss="squared_hinge"`). The main experiments used squared hinge (primal solver); here the standard hinge is added with `LinearSVC(loss="hinge", dual=True, fit_intercept=False)` (the only combination scikit-learn offers for hinge), same pair sets, same seeds, same C grid tuned on dev. Files: `ext_hinge_metrics_test.csv`, `ext_hinge_tuning_dev.csv`, `ext_hinge_significance.csv`.

{md(pd.DataFrame(rows), "{:.4f}")}

* **Convergence:** {nconv} of {nfit} hinge fits reached `max_iter=1000` without meeting the tolerance (liblinear dual coordinate descent; all squared-hinge fits converged: {int(main_tun[main_tun.model.str.startswith('ranksvm')]['no_converge'].sum())} warnings in `tuning_dev.csv`). Hinge solutions are therefore approximate; the chosen models are {'not converged' if not ht.loc['ranksvm_hinge_all','converged'] else 'converged'} (all-pairs) / {'not converged' if not ht.loc['ranksvm_hinge_sampled','converged'] else 'converged'} (sampled).
* **C = 100 is unusable for the hinge variants** (dev NDCG@10 {f4(bad_c['ranksvm_hinge_all'])} all-pairs and {f4(bad_c['ranksvm_hinge_sampled'])} sampled, versus a best of {f4(best_dev['ranksvm_hinge_all'])} and {f4(best_dev['ranksvm_hinge_sampled'])}); the dual solver stops far from the optimum there. Selecting C on dev avoids this.
* Training time of the chosen fits: hinge all-pairs {ht.loc['ranksvm_hinge_all','train_s']:.1f} s vs squared-hinge {main_t.loc['ranksvm_all','train_s']:.1f} s; the whole hinge C grid took far longer than the squared-hinge grid (see `ext_hinge_tuning_dev.csv`, column `train_s`).

Paired tests (test queries, per-query NDCG@10 and AP):

{sig_table(hsig[hsig.metric == 'ndcg10'], ['comparison', 'model'])}

{sig_table(hsig[hsig.metric == 'ap'], ['comparison', 'model'])}
""")

    # =============================================================== 9.2 CV
    cvs = pd.read_csv(R / "ext_cv_summary.csv").set_index("model")
    cvf = pd.read_csv(R / "ext_cv_fold_metrics.csv")
    cvsig = pd.read_csv(R / "ext_cv_significance.csv")
    cvp = pd.read_csv(R / "ext_cv_pooled_oof.csv").set_index("model")
    order = ["bm25", "linreg", "logreg", "ranksvm_all", "ranksvm_sampled", "ranksvm_gapweighted", "ranksvm_dndcgweighted_extra"]
    cv_rows = [{"model": LAB[n], "MAP": pm(cvs.loc[n, "MAP_mean"], cvs.loc[n, "MAP_std"]), "NDCG@5": pm(cvs.loc[n, "NDCG@5_mean"], cvs.loc[n, "NDCG@5_std"]),
                "NDCG@10": pm(cvs.loc[n, "NDCG@10_mean"], cvs.loc[n, "NDCG@10_std"]), "pair acc (micro)": pm(cvs.loc[n, "pair_acc_micro_mean"], cvs.loc[n, "pair_acc_micro_std"]),
                "pooled OOF NDCG@10": cvp.loc[n, "NDCG10_pooled"]} for n in order]
    piv = cvf.pivot(index="fold", columns="model", values="NDCG@10")[order].rename(columns=LAB).reset_index()
    cc = cvf[cvf.model != "bm25"].pivot(index="fold", columns="model", values="C")[order[1:]].rename(columns=LAB).reset_index()
    sv = with_verdict(cvsig)
    n_imp = int(((sv.metric == "ndcg10") & (sv.verdict == "improvement supported")).sum())
    n_deg = int(((sv.metric == "ndcg10") & (sv.verdict == "degradation supported")).sum())
    n_q = int(cvsig["n_queries"].iloc[0])
    cv_max = float(cvsig[cvsig.metric == "ndcg10"]["mean_delta"].abs().max())
    _ap = sv[sv.metric == "ap"]
    _dis = _ap[(_ap.p_ttest < 0.05) != (_ap.p_wilcoxon < 0.05)]
    ap_disagree = ("; ".join(f"{LAB[r.model]}: t-test p = {pf(r.p_ttest)} vs Wilcoxon p = {pf(r.p_wilcoxon)}" for r in _dis.itertuples())
                   if len(_dis) else "no model in this table has one test below and the other above 0.05 for AP")
    _hd = sv[(sv.verdict != "no supported difference") & (sv.p_ttest_holm >= 0.05)]
    holm_note = (f"{len(_hd)} row(s) labelled as supported have a Holm-adjusted t-test p >= 0.05 (" + "; ".join(f"{LAB[r.model]} / {r.metric}: {r.p_ttest_holm:.4f}" for r in _hd.itertuples()) + "), so they would not survive a multiplicity correction over the 6 models"
                 if len(_hd) else "all rows labelled as supported also have Holm-adjusted t-test p < 0.05")
    out.append(f"""### 9.2 Query-level 5-fold cross-validation over all {n_q:,} queries (feature set v1)

Protocol (`src/run_cv.py`, `results/ext_cv_folds.csv`): train+dev+test queries are pooled and assigned to 5 folds **by query id** (seeded permutation, fold sizes differ by at most 1). For each fold the other four folds are the training folds; 20 % of their *queries* form an inner validation split on which C is chosen by NDCG@10; the model is then refit on all training-fold rows with that C and scored once on the held-out fold. The `StandardScaler` is fitted on the training-fold rows only. Every query is therefore scored out-of-fold exactly once per model (asserted). Pooled labels mix train qrels (grade 1 only) with dev/test qrels (grades 1-2), so grade-gap weighting is only partly degenerate here. Hinge variants were not included in the CV (cost, see 9.1). **The official train/dev/test results (section 5) remain the main table.**

Mean ± std (n = 5 folds, ddof = 1) of the per-fold metrics:

{md(pd.DataFrame(cv_rows), "{:.4f}")}

Per-fold NDCG@10 (`ext_cv_fold_metrics.csv`; the std above mostly reflects fold-to-fold query difficulty, which affects BM25 equally):

{md(piv, "{:.4f}")}

C chosen on the inner validation split per fold:

{md(cc, "{:g}")}

Paired tests on the pooled out-of-fold per-query scores (n = {n_q:,}, vs BM25; Holm over the 6 models per metric):

{sig_table(cvsig[cvsig.metric == 'ndcg10'], ['model', 'metric'])}

{sig_table(cvsig[cvsig.metric == 'ap'], ['model', 'metric'])}

**Why the t-test and the Wilcoxon test can disagree (especially for AP), and what the verdict labels use.** The paired t-test tests the *mean* of the per-query differences; the Wilcoxon signed-rank test drops exact zeros and tests a rank-based shift of the non-zero differences. Per-query AP differences are small, skewed and contain many exact zeros (e.g. queries without any relevant candidate), so the two tests can return quite different p-values: {ap_disagree}. The verdict labels in these tables are based on the **raw (unadjusted) p-values** of both tests plus the bootstrap CI; Holm-adjusted p-values are printed beside them, and {holm_note}.

Reading: by the decision rule, **{n_imp} of 6** models improve and **{n_deg} of 6** degrade NDCG@10 relative to BM25 in the cross-validation ({'consistent with' if n_imp == 0 and n_deg == 0 else 'see the table; differs from'} the official-split finding of no supported difference). The larger sample ({n_q:,} vs 323 queries) gives tighter CIs, so small effects can become detectable; a statistically supported effect would still have to be compared with the practical threshold used in section 5.1 (|delta NDCG@10| < 0.01 is negligible). The largest mean NDCG@10 difference to BM25 in the CV is {cv_max:.4f}, i.e. **{'below' if cv_max < 0.01 else 'above'} the 0.01 practical threshold**: {'any supported improvement is statistically detectable but practically negligible' if cv_max < 0.01 else 'see the table'}.
""")

    # =============================================================== 9.3 / 9.4 feature sets
    fm = pd.read_csv(R / "ext_featuresets_metrics_test.csv")
    fd = pd.read_csv(R / "ext_featuresets_metrics_dev.csv")
    sf = pd.read_csv(R / "ext_single_feature_reranks.csv")
    fsig = with_verdict(pd.read_csv(R / "ext_featuresets_significance.csv"))
    vifs = pd.read_csv(R / "ext_vif_by_set.csv")
    drops = pd.read_csv(R / "ext_vif_pruning_v1.csv")
    sign = pd.read_csv(R / "ext_featuresets_sign_check.csv")
    fw = pd.read_csv(R / "ext_featuresets_weights.csv")
    base = main_t.loc["bm25"]
    frows = [{"feature set": "(BM25 baseline)", "model": "BM25 on v1 candidates", "C": "", "test MAP": base["MAP"], "test NDCG@5": base["NDCG@5"],
              "test NDCG@10": base["NDCG@10"], "test pair acc": base["pair_acc_micro"], "dev NDCG@10": main_dev.loc["bm25", "NDCG@10"]}]
    for _, r in sf[sf.split == "test"].iterrows():
        d = sf[(sf.split == "dev") & (sf.model == r["model"])].iloc[0]
        frows.append({"feature set": "single-feature rerank", "model": r["model"].replace("rerank:", "rerank by "), "C": "", "test MAP": r["MAP"],
                      "test NDCG@5": r["NDCG@5"], "test NDCG@10": r["NDCG@10"], "test pair acc": r["pair_acc_micro"], "dev NDCG@10": d["NDCG@10"]})
    for _, r in fm.iterrows():
        d = fd[(fd.feature_set == r["feature_set"]) & (fd.model == r["model"])].iloc[0]
        frows.append({"feature set": r["feature_set"], "model": LAB[r["model"]], "C": "" if pd.isna(r["C"]) else f"{r['C']:g}", "test MAP": r["MAP"],
                      "test NDCG@5": r["NDCG@5"], "test NDCG@10": r["NDCG@10"], "test pair acc": r["pair_acc_micro"], "dev NDCG@10": d["NDCG@10"]})

    def vsub(prefix, metric="ndcg10"):
        s = fsig[(fsig.metric == metric) & (fsig.comparison.str.startswith(prefix))]
        return s

    v2_vs_bm25 = vsub("vs BM25"); v2_vs_bm25 = v2_vs_bm25[v2_vs_bm25.model.str.startswith("v2:")]
    v2_vs_v1 = fsig[(fsig.metric == "ndcg10") & fsig.comparison.str.startswith("v2:") & fsig.comparison.str.contains(" vs v1:")]
    v2_vs_stem = fsig[(fsig.metric == "ndcg10") & fsig.comparison.str.contains("rerank:bm25_stem_comb")]
    n_v2_bm = int((v2_vs_bm25.verdict == "improvement supported").sum()); n_v2_v1 = int((v2_vs_v1.verdict == "improvement supported").sum())
    n_v2_stem = int((v2_vs_stem.verdict != "no supported difference").sum())
    stem_vs_bm = fsig[(fsig.metric == "ndcg10") & (fsig.comparison == "vs BM25") & (fsig.model == "rerank:bm25_stem_comb")].iloc[0]
    k2_vs_bm = fsig[(fsig.metric == "ndcg10") & (fsig.comparison == "vs BM25") & (fsig.model == "rerank:bm25_k2_b05_comb")].iloc[0]
    holm_v2 = v2_vs_bm25[v2_vs_bm25.verdict == "improvement supported"]
    holm_txt = ("; after Holm correction over the 14 'vs BM25' comparisons of this table none of them remains below 0.05" if (holm_v2["p_ttest_holm"] >= 0.05).all() and len(holm_v2) else
                ("; after Holm correction they stay below 0.05" if len(holm_v2) else ""))
    cmp_cols = ["comparison", "model"]
    out.append(f"""### 9.3 Feature set v2 vs v1 (same Top-100 candidates, official split)

`src/features_v2.py` (new file; v1 code and cache untouched) adds five **label-free** features to the 14 v1 features: `bm25_stem_comb` (BM25 on Porter-stemmed title+text; own Porter implementation, unit-tested on 64 examples from Porter's paper), `bigram_match_count` (distinct adjacent query-term pairs found adjacent in the title or text), `min_term_dist` / `avg_term_dist` (smallest / mean-of-pair-minimum token distance between different matched query terms; set to the document length when fewer than two terms match), and `bm25_k2_b05_comb` (BM25 with k1 = 2.0, b = 0.5). Stage 1 (candidates) is unchanged, so only the re-ranking features differ. The v1 run inside `run_featuresets.py` reproduces the main results exactly (assertion on MAP and NDCG@10 of all four models). Scaler on train only; C on dev only. Models: linear / logistic regression and the two squared-hinge Ranking SVMs (all-pairs, sampled).

{md(pd.DataFrame(frows), "{:.4f}")}

To separate "better features" from "just stemming", the table also contains **single-feature reranks** (rerank the same candidates by `bm25_stem_comb` or `bm25_k2_b05_comb` alone; hyper-parameters fixed in advance, nothing learned).

Paired tests, per-query NDCG@10 on the test queries (`ext_featuresets_significance.csv`; AP versions are in the CSV):

*v2 models vs BM25* (Holm over all 14 'vs BM25' comparisons, which also include the v1 / v1_reduced models and the single-feature reranks)

{sig_table(v2_vs_bm25, cmp_cols)}

*v2 vs v1, same model*

{sig_table(v2_vs_v1, cmp_cols)}

*v2 vs stemmed-BM25-only rerank (does v2 add anything beyond stemming?)*

{sig_table(v2_vs_stem, cmp_cols)}

*single-feature reranks vs BM25*

{sig_table(fsig[(fsig.metric == 'ndcg10') & (fsig.comparison == 'vs BM25') & fsig.model.str.startswith('rerank:')], cmp_cols)}

Reading (generated from the decision rule): v2 improves on v1 for **{n_v2_v1} of 4** models, so the new features help *relative to v1*. Against the plain BM25 baseline, **{n_v2_bm} of 4** v2 models are supported improvements{holm_txt}. Re-ranking by stemmed BM25 alone gives NDCG@10 {f4(sf[(sf.split=='test') & (sf.model=='rerank:bm25_stem_comb')]['NDCG@10'].iloc[0])} ({verdict(stem_vs_bm)} vs BM25), and **{n_v2_stem} of 4** v2 models differ from this stemmed-BM25-only rerank in a supported way; {'so the data do not show that the learned models with the extra features add anything beyond stemmed BM25 (absence of a supported difference is not proof of equality: any extra effect is small relative to the noise of 323 queries)' if n_v2_stem == 0 else 'see the table'}. BM25 with k1 = 2, b = 0.5 alone: {verdict(k2_vs_bm)} vs BM25. The test-set mean differences of the v2 models to BM25 are {v2_vs_bm25['mean_delta'].min():.4f} to {v2_vs_bm25['mean_delta'].max():.4f} NDCG@10 (practical threshold 0.01), the CIs are wide, and with 14 comparisons in the family these are exploratory results.
""")

    # ----- 9.4 reduced v1
    red = vifs[vifs.feature_set == "v1_reduced"]
    sg = sign[sign.feature_set.isin(["v1", "v1_reduced"])]
    def nfail(s):
        d = sg[sg.feature_set == s]
        return {m: list(d[d[f"ok_{m}"] == "NO"]["feature"]) for m in ("linreg", "logreg", "ranksvm_all")}
    f_v1, f_red = nfail("v1"), nfail("v1_reduced")
    persist = sorted(set(f_red["linreg"]) | set(f_red["logreg"]) | set(f_red["ranksvm_all"]))
    all3 = sorted(set(f_red["linreg"]) & set(f_red["logreg"]) & set(f_red["ranksvm_all"]))
    n_exp = int((sign[sign.feature_set == "v1_reduced"]["expected_sign"] != 0).sum())
    signtab = sg.rename(columns={"w_linreg": "w linreg", "w_logreg": "w logreg", "w_ranksvm_all": "w RankSVM", "ok_linreg": "ok linreg", "ok_logreg": "ok logreg", "ok_ranksvm_all": "ok RankSVM"})
    interpretable = len(persist) == 0
    red_vs_v1 = fsig[(fsig.metric == "ndcg10") & fsig.comparison.str.startswith("v1_reduced:") & fsig.comparison.str.contains(" vs v1:")]
    n_red_diff = int((red_vs_v1.verdict != "no supported difference").sum())
    out.append(f"""### 9.4 Reduced v1 feature set (drop features until every VIF < 10)

Procedure (label-free, training rows only; `ext_vif_pruning_v1.csv`): repeatedly drop the feature with the largest VIF and recompute. Dropped, in order: {', '.join(f"`{r.feature}` (VIF {r.vif_at_drop:.1f})" for r in drops.itertuples())}. Kept ({len(red)} features): {', '.join('`'+f+'`' for f in red['feature'])}; maximum VIF after pruning = {red['VIF'].max():.2f} (`ext_vif_by_set.csv`). Note that the procedure removed `bm25_comb`, the strongest single feature of v1; the BM25 signal remains through `bm25_title` and `bm25_body`.

Ranking quality (rows `v1_reduced` in the 9.3 table): reduced vs full v1 with the same model is {'**not** distinguishable' if n_red_diff == 0 else 'different in a supported way for ' + str(n_red_diff) + ' of 4 models'} by the decision rule (`ext_featuresets_significance.csv`, comparisons `v1_reduced:<model> vs v1:<model>`).

Sign check with the expected signs fixed before looking at any weights (+1 more is better, -1 less is better, 0 no expectation; `ext_featuresets_sign_check.csv`):

{md(signtab[['feature_set','feature','expected_sign','w linreg','ok linreg','w logreg','ok logreg','w RankSVM','ok RankSVM']], "{:.3f}")}

**Result:** in the full v1 set, features with a wrong sign (linear / logistic / RankSVM): {len(f_v1['linreg'])} / {len(f_v1['logreg'])} / {len(f_v1['ranksvm_all'])} of {int((sign[sign.feature_set=='v1']['expected_sign'] != 0).sum())} features with an expectation. In the reduced set: {len(f_red['linreg'])} / {len(f_red['logreg'])} / {len(f_red['ranksvm_all'])} of {n_exp}. {'The reduced weights ARE sign-interpretable in all three models.' if interpretable else 'So pruning to VIF < 10 did **not** make the weights sign-interpretable: ' + ', '.join('`'+f+'`' for f in persist) + ' still have an unexpected sign in at least one model' + (' (' + ', '.join('`'+f+'`' for f in all3) + ' in all three)' if all3 else '') + '.'} {'' if interpretable else 'A possible reason, which we did not test, is that a low VIF only removes *linear* redundancy: a feature can still receive a negative weight once the other features are held fixed. In any case per-feature signs should not be read as marginal effects.'} Weights come from one fit per model, without uncertainty estimates.
""")
    tldr = (f"* **Extensions (section 9; decision rule = paired t AND Wilcoxon p<0.05 AND CI excluding 0):** "
            f"(i) standard-hinge Ranking SVM: test NDCG@10 {f4(ht.loc['ranksvm_hinge_all','NDCG@10'])} (all pairs) / {f4(ht.loc['ranksvm_hinge_sampled','NDCG@10'])} (sampled) vs squared hinge "
            f"{f4(main_t.loc['ranksvm_all','NDCG@10'])} / {f4(main_t.loc['ranksvm_sampled','NDCG@10'])}; {nconv} of {nfit} hinge fits hit the iteration limit. "
            f"(ii) 5-fold query-level CV over {n_q:,} queries: {n_imp} of 6 models improve and {n_deg} of 6 degrade NDCG@10 vs BM25 (largest mean difference {cv_max:.4f}, {'below' if cv_max < 0.01 else 'above'} the 0.01 practical threshold). "
            f"(iii) Feature set v2: supported improvement over v1 for {n_v2_v1} of 4 models, over BM25 for {n_v2_bm} of 4, and over a stemmed-BM25-only rerank for {n_v2_stem} of 4 (a non-zero count there would mean a supported difference in either direction). "
            f"(iv) Reduced v1 (all VIF < 10): weights are {'sign-interpretable' if interpretable else 'still NOT sign-interpretable'}.")
    return "\n".join(out), tldr
