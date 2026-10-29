"""Builds report.md. Every number comes from files in results/ (tables are generated; prose cites values
through f-string lookups). Assertions guard each qualitative claim: if a re-run changes the outcome the
build fails and the prose must be revised instead of silently going stale.

Usage: python -I src/make_report.py
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C
from analysis import md, LABEL

R = C.RESULTS_DIR
f4 = lambda x: f"{x:.4f}"
f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{100 * x:.1f}%"

st = json.loads((R / "dataset_stats.json").read_text())
cr = pd.read_csv(R / "candidate_recall_summary.csv").set_index("split")
ps = json.loads((R / "pair_stats.json").read_text())
mt = pd.read_csv(R / "metrics_test.csv").set_index("model")
md_ = pd.read_csv(R / "metrics_dev.csv").set_index("model")
tun = pd.read_csv(R / "tuning_dev.csv")
sig = pd.read_csv(R / "significance.csv")
wt = pd.read_csv(R / "weights_table.csv").set_index("feature")
W = pd.read_csv(R / "weights.csv").pivot(index="feature", columns="model", values="weight")
ex = json.loads((R / "analysis_extra.json").read_text())
best = json.loads((R / "best_model.json").read_text())
best_name = ex["best_model"]
pa = pd.read_csv(R / "pairacc_vs_ndcg.csv").set_index("model")
ab = pd.read_csv(R / "ablation.csv")
cc = pd.read_csv(R / "crosscheck_metrics.csv")
orc = pd.read_csv(R / "oracle_upper_bound.csv").iloc[0]
grp = pd.read_csv(R / "query_groups_by_recall.csv")
evd = pd.read_csv(R / "query_evidence.csv")
imp = pd.read_csv(R / "top10_improved.csv"); har = pd.read_csv(R / "top10_harmed.csv")
qd = pd.read_csv(R / "query_deltas_best_model.csv")
runtime = (R / "total_runtime_seconds.txt").read_text().strip() if (R / "total_runtime_seconds.txt").exists() else "n/a"
log = (R / "run_log.txt").read_text()
sha = re.findall(r"sha256=(\w+)", log)[-1]
envline = re.findall(r"\| (python .*)", log)[-1]
ltr = [m for m in mt.index if m != "bm25"]
core = ["bm25", "linreg", "logreg", "ranksvm_all", "ranksvm_sampled", "ranksvm_gapweighted"]
ALL = core + ["ranksvm_dndcgweighted_extra"]

# ------------------------------------------------------------------ guards on qualitative claims
B = mt.loc["bm25"]
assert all(mt.loc[m, "NDCG@10"] < B["NDCG@10"] for m in ltr), "claim 'no LTR model beats BM25 on test NDCG@10' no longer true"
assert (sig["p_ttest"] > 0.05).all() and (sig["p_wilcoxon"] > 0.05).all(), "a difference became significant - rewrite section"
assert (sig["p_ttest_holm"] > 0.05).all()
for c in ("MAP", "NDCG@5", "NDCG@10", "pair_acc_micro"):
    assert np.isclose(mt.loc["ranksvm_all", c], mt.loc["ranksvm_gapweighted", c]), "all-pairs != gap-weighted"
assert ps["all_pairs_fit_budget"]
assert all(mt.loc[m, "pair_acc_micro"] > B["pair_acc_micro"] for m in ltr if m.startswith("ranksvm"))
assert mt.loc["ranksvm_all", "NDCG@10"] < B["NDCG@10"]
assert mt.loc["linreg", "pair_acc_micro"] < B["pair_acc_micro"] and mt.loc["logreg", "pair_acc_micro"] < B["pair_acc_micro"]
assert md_.loc[best_name, "NDCG@10"] > md_.loc["bm25", "NDCG@10"]       # LTR looks better on dev ...
assert mt.loc[best_name, "NDCG@10"] < B["NDCG@10"]                       # ... but not on test
assert abs(W.loc["query_len", "ranksvm_all"]) < 1e-9
assert wt.loc["lm_dirichlet", "sign_ok_ranksvm_all"] == "NO" and wt.loc["matched_terms", "sign_ok_ranksvm_all"] == "NO"
assert best_name == "ranksvm_sampled"
assert ab.loc[1, "test_pair_acc"] < ab.loc[0, "test_pair_acc"]
assert abs(ab.loc[2, "test_NDCG@10"] - ab.loc[0, "test_NDCG@10"]) < 1e-3
assert cc["max_abs_diff"].max() < 1e-9
assert ((sig["ci95_lo"] < 0) & (sig["ci95_hi"] > 0)).all(), "a CI excludes 0 - rewrite section 5.1/TL;DR"
for _f in ("lm_dirichlet", "matched_terms"):
    assert (W.loc[_f, ["ranksvm_all", "linreg", "logreg"]] < 0).all() and wt.loc[_f, "spearman_with_label_train"] > 0
assert (W.loc["bm25_body", ["ranksvm_all", "linreg", "logreg"]] < 0).all()
assert mt.loc["ranksvm_sampled", "NDCG@10"] >= mt.loc["ranksvm_all", "NDCG@10"] and mt.loc["ranksvm_sampled", "MAP"] >= mt.loc["ranksvm_all", "MAP"]
assert mt.loc["ranksvm_sampled", "pair_acc_micro"] > mt.loc["ranksvm_all", "pair_acc_micro"]
assert mt.loc["ranksvm_sampled", "train_s"] < mt.loc["ranksvm_all", "train_s"] and mt.loc["ranksvm_sampled", "peak_mb"] < mt.loc["ranksvm_all", "peak_mb"]
assert mt.loc["linreg", "train_s"] * 10 < mt.loc["ranksvm_all", "train_s"] and mt.loc["logreg", "train_s"] * 10 < mt.loc["ranksvm_all", "train_s"]
assert ab.loc[1, "dev_NDCG@10"] < ab.loc[0, "dev_NDCG@10"] and ab.loc[1, "test_NDCG@10"] > ab.loc[0, "test_NDCG@10"]   # 'opposite directions on dev/test'
assert ex["spearman_delta_ndcg_vs_delta_pairacc"] > 0 and ex["n_queries_ndcg_down_pairacc_up"] > ex["n_queries_ndcg_up_pairacc_down"]
assert pa.loc["ranksvm_all", "pair_acc_pairs_entirely_below_BM25_top10"] - pa.loc["bm25", "pair_acc_pairs_entirely_below_BM25_top10"] > \
       pa.loc["ranksvm_all", "pair_acc_pairs_with_a_BM25_top10_doc"] - pa.loc["bm25", "pair_acc_pairs_with_a_BM25_top10_doc"]

# ------------------------------------------------------------------ hand-written, evidence-based per-query explanations
# (written after reading results/query_evidence.md; they reference only what that file shows)
EXPL = {
 "PLAIN-371": "Only 2 relevant documents. The rel=2 document moves BM25#11 -> LTR#7 and a rel=1 document #4 -> #3; the document it replaces, \"An update on bioactive plant lignans\", matches only the generic query word *update*. The gain rests on one document, so it is fragile.",
 "PLAIN-430": "Two rel=2 documents (BM25#15 and #23) enter the top-10 at LTR#5 and #9, pushing out three non-relevant diet/pollutant papers. The entering documents have a *lower* bm25_comb and fewer matched terms than the ones they replace, so this is not a 'better lexical match' effect; it looks incidental to the learned weight combination.",
 "PLAIN-1537": "BM25's top-10 contained no relevant document (NDCG 0). Two relevant papers whose titles contain \"Low-Carbohydrate Diets\" (BM25#11, #21) rise to LTR#3 and #4, displacing off-topic *diet* papers (protein, flaxseed, fat). Driver: bm25_title. Three of the five entering documents are non-relevant, so the gain is two documents.",
 "PLAIN-207": "A relevant document enters at LTR#9 (BM25#13) and relevant documents already in the top-10 move up (BM25#3->#1, #9->#4). One entering non-relevant document is on-topic (heterocyclic amines) and may simply be unjudged. Feature differences are small (largest 0.34).",
 "PLAIN-1453": "4 of the 5 entering documents are relevant (food additives, children's food intake) while the 5 leaving ones are food-safety/handling papers that match *food* but not *junk*. Only 25 of 245 relevant documents are in the candidates, so a few promotions matter a lot. No single feature explains it (all contribution differences <= 0.07).",
 "PLAIN-280": "No new relevant document enters; relevant documents already in the top-10 move up (BM25#2->#1, #3->#2, #7->#5, #12->#8). The gain is a pure re-ordering; largest contribution differences are bm25_comb (negative) and matched_terms.",
 "PLAIN-2991": "The only relevant document in the top-10 (rel=2) moves from BM25#7 to LTR#2, while all five entering documents are non-relevant. The entering/leaving comparison does not explain why this particular document moved (its own feature vector was not inspected), so no causal explanation is claimed.",
 "PLAIN-2730": "Two rel=2 documents with *angiogenesis* in the title enter at LTR#4 and #5 (from BM25#14, #20); two off-topic Campylobacter papers also enter, and other relevant documents shuffle (BM25#4->#8, #6->#10, #7->#3). Net positive but mixed. Entering documents have lower bm25_comb and fewer matched terms.",
 "PLAIN-3332": "A rel=1 document enters (BM25#11 -> #9) and a rel=2 document moves #5 -> #2; two entering documents are non-relevant. A small effect from one or two documents.",
 "PLAIN-3141": "BM25's top-10 had no relevant document (NDCG 0). One rel=2 document (title about estrogen excretion) moves BM25#11 -> #4; the four other entering documents are non-relevant. The entire gain is this single document (driver: bm25_title).",
 "PLAIN-3085": "Both rel=2 documents (BM25#3 and #10) fall to #22 and #12; four non-relevant vitamin-D papers (e.g. \"Vitamin D: extraskeletal health\") take their place. These are topical, so part of the harm may be a judgement artefact (unjudged = non-relevant). Driver: lm_dirichlet, whose learned weight is negative - the entering documents have a *lower* LM score.",
 "PLAIN-3026": "A single relevant document (rel=2) falls from BM25#8 to #14; the two entering documents are non-relevant and longer (doc_len contribution +0.11). With one relevant document NDCG@10 goes 0.315 -> 0 on a one-position-pair swap.",
 "PLAIN-2460": "The rel=2 \"Banting Memorial Lecture\" falls BM25#1 -> #6 and another rel=2 document #5 -> #12; six non-relevant diabetes-diet papers enter. Driver: matched_terms, whose weight is negative (entering documents match *fewer* query terms).",
 "PLAIN-2500": "Two rel=2 documents drop (BM25#5 -> #30, #9 -> #18) while one rel=1 document rises to #4; five non-relevant saturated-fat papers enter. Driver: matched_terms (negative weight) again.",
 "PLAIN-2700": "The rel=2 \"Pathobiological determinants of atherosclerosis in youth\" falls BM25#2 -> #18; entering documents are about *childhood* soy/dairy/leukemia, i.e. they match one query word. Driver: matched_terms (entering documents match fewer terms).",
 "PLAIN-2354": "Very short query, two relevant documents; a rel=1 document moves BM25#1 -> #5 while one non-relevant document enters. Drivers: bm25_title, tfidf_cos_title. A small re-ordering that costs a lot only because NDCG@10 of this query is already based on 2 documents.",
 "PLAIN-3171": "Generic query (\"Convergence of Evidence\"). The LTR model promotes documents with *evidence* in the title (bm25_title, tfidf_cos_title) and pushes three relevant documents from BM25#2, #8, #9 to #39, #29, #28. Lexical signal is weak (recall@100 0.235), and the title-field features overweight the generic word.",
 "PLAIN-2680": "Two rel=2 documents (BM25#5, #8: flavonols / AhR ligands, i.e. *indirect* relevance) fall to #13, #14; three literal dioxin-exposure papers, judged non-relevant, enter. Relevance in NFCorpus is based on citation links, so literal topical matches are not necessarily relevant.",
 "PLAIN-2540": "Generic words of the title query (*does ... matter*) pull in \"Human gut microbiota: does diet matter?\" at LTR#4; the rel=2 document at BM25#1 falls to #8. Contribution differences are small (largest 0.16).",
 "PLAIN-102": "Only 1 of 24 relevant documents is in the candidates (recall@100 0.042); it falls BM25#5 -> #22 and three documents containing *childhood* enter. Nothing was available to rerank well; the harm is one document.",
}
assert set(EXPL) == set(imp["qid"]) | set(har["qid"]), "explanations do not match the current top-10 lists"

# ------------------------------------------------------------------ tables
def metric_table(m, mark=None):
    rows = []
    for n in ALL:
        r = m.loc[n]
        rows.append({"model": LABEL[n] + (" (dev-selected)" if n == mark else ""), "MAP": r["MAP"], "NDCG@5": r["NDCG@5"], "NDCG@10": r["NDCG@10"],
                     "pair acc (micro)": r["pair_acc_micro"], "pair acc (macro)": r["pair_acc_macro"],
                     "C": "" if pd.isna(r["C"]) else f"{r['C']:g}"})
    return md(pd.DataFrame(rows))

cost = md(pd.DataFrame([{"model": LABEL[n], "train pairs": "" if pd.isna(mt.loc[n, "n_pairs"]) else f"{int(mt.loc[n, 'n_pairs']):,}",
                         "train time (s)": mt.loc[n, "train_s"], "predict time, 32k rows (ms)": mt.loc[n, "predict_s"] * 1000 if not pd.isna(mt.loc[n, "predict_s"]) else np.nan,
                         "peak process RSS during fit (MB)": mt.loc[n, "peak_mb"]} for n in ALL]), "{:.2f}")
tun_tab = md(tun.assign(C=tun["C"].map(lambda c: "" if pd.isna(c) else f"{c:g}"))[["model", "C", "MAP", "NDCG@10", "pair_acc_micro", "train_s", "no_converge"]].rename(columns={"pair_acc_micro": "dev pair acc"}))
sig_tab = md(sig.assign(model=sig["model"].map(LABEL))[["model", "metric", "mean_delta", "ci95_lo", "ci95_hi", "p_ttest", "p_ttest_holm", "p_wilcoxon", "p_wilcoxon_holm", "cohen_dz", "n_improved", "n_harmed", "n_unchanged"]])
wtab = md(wt.reset_index()[["feature", "ranksvm_all", "ranksvm_sampled", "linreg", "logreg", "expected_sign", "spearman_with_label_train", "spearman_within_query_train", "VIF", "sign_ok_ranksvm_all", "sign_ok_linreg", "sign_ok_logreg"]], "{:.3f}")
rec_tab = md(cr.reset_index()[["split", "n_queries", "recall@100_macro", "recall@100_micro", "n_queries_zero_rel_in_cand", "frac_queries_zero_rel_in_cand", "frac_queries_lt100_cand", "mean_rel_per_query", "frac_queries_with_gt100_rel"]])
pa_tab = md(pa.reset_index().assign(model=lambda d: d["model"].map(LABEL)))
ab_tab = md(ab[["variant", "C", "n_pairs", "intercept", "dev_NDCG@10", "dev_pair_acc", "test_NDCG@10", "test_MAP", "test_pair_acc"]])
cc_tab = md(cc[["model", "ndcg10_lin_ours", "ndcg10_lin_ref", "ndcg10_exp_ours", "ndcg10_exp_ref", "map_ours", "map_ref", "max_abs_diff"]], "{:.6f}")
grp_tab = md(grp)

def qtable(tab):
    out = []
    for _, r in tab.iterrows():
        out.append(f"| {r['qid']} | {r['query']} | {r['ndcg10_bm25']:.3f} -> {r['ndcg10_ltr']:.3f} ({r['delta_ndcg10']:+.3f}) | {int(r['n_rel_total'])} / {int(r['n_rel_in_cand'])} / {r['recall_at_100']:.3f} | {EXPL[r['qid']]} |")
    return "| qid | query | NDCG@10 BM25 -> LTR (delta) | #rel total / in candidates / recall@100 | evidence-based explanation |\n|---|---|---|---|---|\n" + "\n".join(out)

harm_matched = int(sum("matched_terms" in (r.top_feature_1, r.top_feature_2, r.top_feature_3) for r in evd[evd.group == "HARMED"].itertuples()))
imp_matched = int(sum("matched_terms" in (r.top_feature_1, r.top_feature_2, r.top_feature_3) for r in evd[evd.group == "IMPROVED"].itertuples()))
harm_left_rel = int(evd[evd.group == "HARMED"]["n_left_relevant"].sum()); harm_enter_rel = int(evd[evd.group == "HARMED"]["n_entered_relevant"].sum())
imp_left_rel = int(evd[evd.group == "IMPROVED"]["n_left_relevant"].sum()); imp_enter_rel = int(evd[evd.group == "IMPROVED"]["n_entered_relevant"].sum())
sg = lambda m, met, col: sig[(sig.model == m) & (sig.metric == met)].iloc[0][col]
mde = 2.8 * (sg(best_name, 'ndcg10', 'ci95_hi') - sg(best_name, 'ndcg10', 'ci95_lo')) / (2 * 1.96)
sv = mt.loc["ranksvm_all"]; ss = mt.loc["ranksvm_sampled"]; lr = mt.loc["linreg"]; lg = mt.loc["logreg"]; dn = mt.loc["ranksvm_dndcgweighted_extra"]
n_tr_q_all = st["train"]["n_queries"]

report = f"""# Two-stage Learning-to-Rank on BEIR NFCorpus: BM25 Top-100 -> pointwise vs. Ranking-SVM re-ranking

*All numbers in this report are generated from files in `results/` by `src/make_report.py`; nothing is typed in by hand
except the per-query explanations in section 7b (which cite `results/query_evidence.md`). Re-running the pipeline regenerates the report,
and the generator aborts if a qualitative claim below stops being true.*

## 0. TL;DR (honest summary)

* On the held-out **test** queries no learned re-ranker beats the BM25 baseline: NDCG@10 BM25 = **{f4(B['NDCG@10'])}**, the model selected on **dev** ({LABEL[best_name]}) = {f4(mt.loc[best_name,'NDCG@10'])}; the seven LTR models range from {f4(mt.loc[ltr,'NDCG@10'].min())} to {f4(mt.loc[ltr,'NDCG@10'].max())} (shown for completeness, never used for selection); MAP BM25 = {f4(B['MAP'])}, dev-selected model MAP = {f4(mt.loc[best_name,'MAP'])}.
  All paired tests (t-test, Wilcoxon, also Holm-corrected) are non-significant (smallest raw p = {f3(min(sig.p_ttest.min(), sig.p_wilcoxon.min()))}); all bootstrap CIs contain 0.
* On the **dev** split the best LTR model ({LABEL[best_name]}) looked slightly better than BM25 (NDCG@10 {f4(md_.loc[best_name,'NDCG@10'])} vs {f4(md_.loc['bm25','NDCG@10'])}); this did not carry over to test - a reminder that tiny dev gains after selection are optimistic.
* Pairwise accuracy of every Ranking SVM is *higher* than BM25's ({f4(sv['pair_acc_micro'])} vs {f4(B['pair_acc_micro'])} for all-pairs) while its NDCG@10 is *lower*; the pointwise models have the lowest pairwise accuracy. Section 7c explains this with the data.
* The main reason for the limited headroom is **candidate generation**: only {pct(cr.loc['test','recall@100_micro'])} (micro) of the relevant test documents are in BM25's Top-100 and {pct(cr.loc['test','frac_queries_zero_rel_in_cand'])} of test queries have *no* relevant document among their candidates. An oracle that re-ranks the same candidates with the true labels reaches only NDCG@10 = {f4(orc['NDCG@10'])}, MAP = {f4(orc['MAP'])}.
* Grade-gap weighting is *degenerate* here: the BEIR NFCorpus **train** qrels contain only grade 1, so every training pair has gap 1 and the gap-weighted SVM is numerically identical to the all-pairs SVM. A rank-aware extra variant (dNDCG-weighted) is reported separately.

## 1. Setup and reproducibility

* Environment: `{envline}`, Windows 10, 13.9 GB RAM, no GPU. Libraries: numpy, pandas, scipy, scikit-learn, matplotlib (+ `ranx` only for the metric cross-check; `pytrec_eval` could not be built on this machine because it needs a C compiler).
* Seed: `{C.SEED}` everywhere (pair sampling, flips, bootstrap, solvers). Seeds per component are `SEED + k` (see `src/run_models.py`).
* Data: BEIR NFCorpus zip (`sha256 {sha}`), downloaded from `{C.DATA_URL}`.
* Full pipeline wall-clock (`bash run_all.sh`, including tests): **{runtime} s**. Per-step timings and every command/version/seed are logged in `results/run_log.txt`.
* BM25 parameters k1={C.BM25_K1}, b={C.BM25_B} (defaults, **not tuned**); LM-Dirichlet mu={C.LM_MU:g} (fixed, not tuned); Top-K={C.TOP_K}.

## 2. Dataset

BEIR NFCorpus: {st['n_docs']:,} documents (title + abstract; median {st['doc_words_median']:.0f} words), {st['n_queries_total']:,} queries (NutritionFacts article/video titles).
The official train/dev/test split exists and is **query-disjoint** (asserted in code and tests), so the official split is the main evaluation. (A query-level 5-fold cross-validation over all queries is added as a robustness check in section 9.2; it does not replace the official split.)

| split | #queries | #judgments | grade counts | mean judged docs / query |
|---|---|---|---|---|
| train | {st['train']['n_queries']:,} | {st['train']['n_judgments']:,} | {st['train']['levels']} | {st['train']['mean_judged_per_query']:.1f} |
| dev | {st['dev']['n_queries']:,} | {st['dev']['n_judgments']:,} | {st['dev']['levels']} | {st['dev']['mean_judged_per_query']:.1f} |
| test | {st['test']['n_queries']:,} | {st['test']['n_judgments']:,} | {st['test']['levels']} | {st['test']['mean_judged_per_query']:.1f} |

Observations that matter for the experiment:

1. **Train qrels have only grade 1; dev/test have grades 1 and 2.** Graded labels are used where they exist (NDCG, pointwise regression target), but the training signal is effectively binary. (Source: `results/dataset_stats.json`.)
2. Only judged documents carry labels: **unjudged documents are treated as non-relevant** everywhere (training labels, MAP, NDCG, pairwise accuracy). In NFCorpus relevance comes from citation links, so topical but unlinked documents count as non-relevant; this penalises every system.
3. {st['n_duplicate_text_groups']} groups of documents ({st['n_docs_in_duplicate_groups']} docs) have identical title+text under different ids. They are left as they are (they are different corpus entries with their own judgments).
4. The corpus is shared across splits, as in the BEIR benchmark: {pct(st['dev_judged_docs_also_judged_in_train'])} of dev-judged and {pct(st['test_judged_docs_also_judged_in_train'])} of test-judged documents are also judged for some train query. This is not label leakage, because no feature uses document identity or labels (verified by test `test_features_do_not_depend_on_labels_and_have_no_label_argument`), but length-like features could in principle let a model memorise documents that were often relevant. Corpus statistics (df, avgdl, collection frequencies) use all documents' text but never labels.
5. MAP uses binarised relevance (rel >= 1). NDCG uses graded labels.

## 3. Method

### 3.1 Stage 1 - BM25 Top-100
Own implementation (`src/bm25.py`): lower-casing, `[a-z0-9]+` tokens, scikit-learn English stop-word list, **no stemming**. For each field, `idf(t) = ln(1 + (N-df+0.5)/(df+0.5))`,
`score(q,d) = sum_(t in unique(q)) idf(t) * tf(k1+1)/(tf + k1(1-b+b*dl/avgdl))` (query-term multiplicity ignored). Retrieval field = title + text concatenated; only documents with score > 0 can be retrieved, so a query with fewer than 100 matching documents has fewer than 100 candidates (and a query with no match has none; such queries are kept in every metric with score 0 for every model).

### 3.2 Stage 2 - features (computed from raw text only; `src/features.py`)
`FEATURE_NAMES` = bm25_title, bm25_body, bm25_comb (field-specific BM25), tfidf_cos_title, tfidf_cos_body (sublinear tf x smoothed idf, cosine), lm_dirichlet (query likelihood with Dirichlet smoothing over title+text; unique query terms), query_len (tokens after stop-word removal), doc_len (body tokens), title_len, matched_terms (unique query terms present in title+text), coverage_ratio (matched / unique query terms), sum_idf_matched, max_idf_matched, bm25_rank (rank 1..n in the candidate list by bm25_comb).
No per-query normalisation was used. `StandardScaler` is fitted **on the training candidate rows only** (`results/scaler_stats.csv`); all weights below are on these standardised features.

### 3.3 Models (all re-rank the same candidate sets; ties are broken by BM25 rank)
1. BM25 baseline (score = bm25_comb).
2. Pointwise **linear regression** (OLS, target = graded label; in training all positives are 1).
3. Pointwise **logistic regression** (target rel >= 1; C tuned on dev from {C.C_GRID}).
4-6. **Ranking SVM**: `LinearSVC(fit_intercept=False, dual=False, loss="squared_hinge")` on pair differences x_i - x_j; pairs only inside a query and only when labels differ; a random **exactly-half** of the pairs is flipped (x_j - x_i, -1) so classes are balanced. Squared hinge is scikit-learn's default and is not the classical hinge Ranking SVM; the primal solver is used because #pairs >> #features.
   * *all pairs*: every valid pair of every training query. **Memory control:** {ps['n_valid_pairs_all']:,} valid pairs x {ps['n_features']} features; estimated training memory (float64 matrix + liblinear's internal copy) = {ps['est_train_memory_gb_all_pairs']:.2f} GB against a budget of {ps['memory_budget_gb']} GB, so **all-pairs fits** and is used. (Measured peak process memory: {mt.loc['ranksvm_all','peak_mb']:.0f} MB, section 5.)
   * *sampled pairs*: for each query, a uniform random sample **without replacement** of at most {ps['sample_cap_per_query']} valid pairs (queries with fewer keep all) -> {ps['n_pairs_ranksvm_sampled']:,} pairs ({pct(ps['n_pairs_ranksvm_sampled']/ps['n_valid_pairs_all'])} of all pairs). The cap is a design choice, not tuned.
   * *grade-gap-weighted*: all pairs, `sample_weight = |y_i - y_j|`.
   * *extra, not in the assignment*: **dNDCG-weighted** all pairs, `w = |2^y_i - 2^y_j| * |1/log2(1+rank_i) - 1/log2(1+rank_j)|` using BM25 ranks, normalised to mean 1. Added only because the gap weighting is degenerate on this training set.
* Hyper-parameter selection: **C in {C.C_GRID} chosen on the dev split by NDCG@10** (first maximum, i.e. smallest C on ties). The test set is used once per model with the chosen C. Models are trained on train only (no refit on train+dev).

### 3.4 Evaluation definitions (`src/metrics.py`, written from scratch)
* NDCG@k with **gain 2^rel - 1** and discount 1/log2(rank+1) (primary; linear-gain NDCG@10 is also stored in the CSVs as `NDCG@10_lin`). **IDCG is computed from the full qrels** of the query, not from the candidates, so missing relevant documents (candidate recall) lower the score.
* AP = sum_k P@k * rel_k / (number of relevant documents in the full qrels), on the ranked candidate list (<= 100); MAP = mean over all queries of the split (including queries with no candidates, which count as 0).
* Pairwise accuracy = fraction of same-query pairs with different labels where the higher-labelled document has the higher score (ties count 0.5; tie fraction is in the CSV). *micro* pools all pairs; *macro* averages over queries having >= 1 valid pair.
* Unit tests (`tests/test_metrics.py`) check DCG/NDCG/AP/pairwise accuracy on hand-computed examples. **Cross-check against `ranx` on the real test run** (rankings passed as strictly decreasing scores so ties cannot differ):

{cc_tab}

### 3.5 Significance
Per-query differences vs BM25 on the {int(sig['n_queries'].iloc[0])} test queries; paired t-test, Wilcoxon signed-rank (zero differences dropped), 95% percentile bootstrap CI of the mean ({C.N_BOOT:,} query resamples), effect size Cohen's d_z = mean(delta)/sd(delta); Holm correction across the {len(ltr)} models within each metric.

## 4. Candidate recall (Stage 1)

{rec_tab}

* Recall@100 is **low** ({pct(cr.loc['test','recall@100_macro'])} macro, {pct(cr.loc['test','recall@100_micro'])} micro on test) partly because {pct(cr.loc['test','frac_queries_with_gt100_rel'])} of test queries have more than 100 relevant documents (so Recall@100 cannot reach 1) and partly because NFCorpus relevance is indirect (query = video title, relevant = papers cited on the page; vocabulary mismatch).
* **{cr.loc['test','n_queries_zero_rel_in_cand']:.0f} of {cr.loc['test','n_queries']:.0f} test queries ({pct(cr.loc['test','frac_queries_zero_rel_in_cand'])}) have zero relevant documents among the candidates**; no re-ranker can change their NDCG/AP. {pct(cr.loc['test','frac_queries_lt100_cand'])} of test queries have fewer than 100 candidates because fewer than 100 documents contain any query term.
* Upper bound: re-ranking the same candidates with the true test labels would give MAP {f4(orc['MAP'])} / NDCG@10 {f4(orc['NDCG@10'])} (`results/oracle_upper_bound.csv`; uses test labels, for context only).

## 5. Results (test split, same candidate sets)

{metric_table(mt, best_name)}

All six assigned models and the extra dNDCG-weighted variant are listed. **Model selection is done on dev only**: the row marked *(dev-selected)* ({LABEL[best_name]}, chosen by dev NDCG@10 among the LTR models) is the system that would be reported as 'the' LTR result; the other rows are shown for completeness and must not be read as a test-set selection.

Dev-split values (used for C selection; `results/metrics_dev.csv`):

{metric_table(md_)}

Cost (train time of the final fit; predict time = median of 5 scorings of the {int(len(pd.read_csv(R/'test_scores.csv.gz'))):,} test candidate rows; memory = peak working set of the whole Python process sampled every 20 ms during `fit`, therefore it includes the already loaded data and is only an upper-level indicator, `results/metrics_test.csv`):

{cost}

Full C grid on dev (`results/tuning_dev.csv`, figure `results/figures/tuning_curves_dev.png`): for the Ranking SVMs and logistic regression the metrics are practically flat for C >= 0.1 (differences in the 4th decimal), so the *selected* C within the plateau (e.g. {mt.loc['ranksvm_all','C']:g} for all-pairs) is essentially arbitrary and should not be interpreted. No fit emitted a convergence warning ({'all fits converged' if not tun['no_converge'].any() else 'SOME fits did not converge'}).

<details><summary>Full tuning table</summary>

{tun_tab}

</details>

### 5.1 Significance vs BM25 (test, per-query paired differences)

{sig_tab}

*Statistical vs practical significance.* No difference is statistically significant (raw or Holm-adjusted), and every 95% CI includes 0. The largest mean change in NDCG@10 is {f4(sig[sig.metric=='ndcg10'].mean_delta.abs().max())} (absolute), well below the 0.01 that we (arbitrarily, by convention) regard as the smallest change a user could care about, and Cohen's d_z is at most {f3(sig.cohen_dz.abs().max())} (negligible). Note that {int(sg(best_name,'ndcg10','n_unchanged'))} of {int(sg(best_name,'ndcg10','n_queries'))} queries are unchanged for the best model, mostly the {cr.loc['test','n_queries_zero_rel_in_cand']:.0f} with no relevant candidate, so the paired tests have an effective sample of {int(sg(best_name,'ndcg10','n_nonzero'))} queries (this is what the Wilcoxon test uses). With ~{int(sg(best_name,'ndcg10','n_queries'))} queries the study could only detect effects of roughly |mean delta| >= {f3(mde)} (about the standard error x 2.8 for 80% power), so "no significant difference" is **not** evidence that the models are equal; it says the difference, if any, is small.

## 6. Pitfall checklist

| pitfall | how it is avoided / test |
|---|---|
| pairs across different queries | `pairs.build_pairs` forms pairs inside query groups only and asserts `qid[i]==qid[j]`; test `test_pairs_same_query_different_label_balanced`. Ablation B (section 7e) shows what cross-query pairs do. |
| equal-label pairs used as preferences | only `y_i > y_j` pairs are generated; assertion `y[I] > y[J]` and the same test. |
| unbalanced pair directions | exactly half of the pairs flipped (assert `|frac(+1)-0.5|<=1/n` in `run_models.py`; test checks <= 1 difference). A single-class pair set cannot even be fitted (ablation table). |
| C selected on test | C chosen on dev only (`run_models.py`, `tuning_dev.csv`); the test split is scored after the choice; the "best LTR model" for the per-query analysis is also chosen on dev. |
| ignoring candidate-generation recall | section 4 (`candidate_recall*.csv`); IDCG/AP use full qrels; oracle upper bound reported. |
| random row-level split | official **query-level** split; `assert_query_disjoint` + assertion in `run_models.py`; test `test_no_query_in_two_splits_detects_leak`. |
| scaler fitted on test | `StandardScaler.fit` on train rows only; assertion `scaler.mean_ == train mean`; tests `test_scaler_statistics_come_from_train_only` and `test_saved_scaler_matches_train_split_if_results_exist` (checks the saved `scaler_stats.csv` against the train rows). |
| row-level / non-query folds in CV | folds are a function of the query id (`run_cv.assign_folds`); the per-query scoring asserts that each query is scored out-of-fold exactly once; tests `test_folds_are_query_level_complete_and_balanced`, `test_row_level_split_would_be_detected_but_query_folds_never_share_queries`. |
| tuning on the CV test fold / scaler leakage in CV | C is chosen on an inner validation split of the *training-fold queries* (`inner_split`, test `test_inner_validation_split_is_query_level_disjoint_and_sized`); the scaler is fitted on training-fold rows only (`test_scaler_helper_uses_training_frame_only`). |
| label-derived features in v2 | `compute_v2_new` has no label argument (test `test_no_label_argument`); labels are never read by `features_v2.py`. |
| all pairs without memory control | `count_pairs` and `estimate_memory_gb` before building; budget {ps['memory_budget_gb']} GB; decision logged in `pair_stats.json`; capped sampled variant provided and compared. |
| evaluating only classification accuracy | MAP, NDCG@5/10 are reported next to pairwise accuracy; section 7c shows they can disagree. |
| duplicate docs across splits | corpus is shared by design (BEIR); duplicates are documented (section 2); features use no doc identity or labels (test `..._do_not_depend_on_labels...`). Query ids are disjoint across splits. |
| features leaking labels | `stage1_and_features` has no label argument (test inspects the signature); labels are attached after feature computation. |
| metric bugs | hand-computed unit tests + cross-check with ranx on the real run (section 3.4). |

## 7. Analysis

### 7a. Feature weights (standardised features; `results/weights_table.csv`, figure `results/figures/feature_weights.png`)

Weights are comparable *across features within one model* only because all features were standardised on the training set (unit variance). Magnitudes are **not** comparable *between* models (OLS predicts a 0/1 label, logistic regression works on log-odds, the SVM on margin units). Univariate columns: Spearman correlation of the feature with the label on the training candidates (pooled and after centring inside each query); VIF = collinearity diagnostic from the training correlation matrix; `expected_sign` was fixed *before* looking at weights (+1 = more is better, 0 = no expectation).

{wtab}

![weights](results/figures/feature_weights.png)

* Largest |weights| in the all-pairs SVM: {', '.join(f"{i} ({W.loc[i,'ranksvm_all']:+.2f})" for i in W['ranksvm_all'].abs().sort_values(ascending=False).index[:4])}; top positive: {W['ranksvm_all'].idxmax()} ({W['ranksvm_all'].max():+.3f}), top negative: {W['ranksvm_all'].idxmin()} ({W['ranksvm_all'].min():+.3f}).
* **Sign sanity check: it fails for lm_dirichlet and matched_terms** (negative weights in the SVM, linear and logistic models although each correlates *positively* with relevance, Spearman {f3(wt.loc['lm_dirichlet','spearman_with_label_train'])} and {f3(wt.loc['matched_terms','spearman_with_label_train'])}); bm25_body is also negative in all three. The reason is collinearity, not an IR principle: VIFs are {f3(wt.loc['bm25_comb','VIF'])} (bm25_comb), {f3(wt.loc['bm25_body','VIF'])} (bm25_body), {f3(wt.loc['sum_idf_matched','VIF'])} (sum_idf_matched), {f3(wt.loc['lm_dirichlet','VIF'])} (lm_dirichlet), so individual coefficients are not interpretable; the model uses them as corrections relative to bm25_comb (e.g. "BM25 score per number of matched terms"). Only the *combined* score is meaningful. Per-feature importance should therefore not be read from single weights.
* **query_len has weight exactly {W.loc['query_len','ranksvm_all']:.1f} in every Ranking SVM**: it is constant inside a query, so it cancels in every pair difference x_i - x_j. In the pointwise models it receives a small nonzero weight ({W.loc['query_len','linreg']:+.3f}, {W.loc['query_len','logreg']:+.3f}) which cannot change a within-query ranking either (a constant added to all candidates). Likewise bm25_rank is a deterministic monotone function of bm25_comb inside a query, so the two are redundant for ranking except for scale/non-linearity.
* The two pointwise models and the Ranking SVM agree on the sign for {int(((np.sign(W['linreg'])==np.sign(W['ranksvm_all'])) & (np.sign(W['logreg'])==np.sign(W['ranksvm_all']))).sum())} of {len(W)} features. Sign agreement for `{'`, `'.join([i for i in W.index if i != 'query_len' and np.sign(W.loc[i,'linreg'])!=np.sign(W.loc[i,'ranksvm_all'])])}` (OLS vs SVM) is lacking; weights of near-collinear features are unstable.
* Weights come from one training run per model (no bootstrap of the weights), so no uncertainty is attached to them.

### 7b. Queries improved / harmed by LTR

Best LTR model **chosen on dev** by NDCG@10: **{LABEL[best_name]}** (C = {best['chosen_C'][best_name]:g}). Per-query delta = NDCG@10(LTR) - NDCG@10(BM25) on test (`results/query_deltas_best_model.csv`, figure `results/figures/delta_ndcg10_hist.png`). {ex['n_queries_changed']} queries changed, {ex['n_queries_unchanged']} did not (of which {ex['unchanged_with_zero_recall']} have no relevant document in the candidates). Raw evidence for every query below (entering/leaving documents with labels and ranks, all relevant candidates, feature-contribution differences w*z) is in `results/query_evidence.md` and `results/query_evidence.csv`.

![delta](results/figures/delta_ndcg10_hist.png)

**Top 10 improved**

{qtable(imp)}

**Top 10 harmed**

{qtable(har)}

What the evidence shows (computed from `query_evidence.csv`):

* Improvements are mostly **a few relevant documents moving from BM25 ranks ~11-50 into the top-10**: in the improved list {imp_enter_rel} relevant documents enter the top-10 and {imp_left_rel} leave it; in the harmed list {harm_enter_rel} relevant documents enter and {harm_left_rel} leave. Harmed queries mainly lose **one or two relevant documents that BM25 already had in its top-10** (many such queries have only 1-3 relevant documents, so the NDCG swing is large).
* In the harmed queries the largest contribution differences are mostly from matched_terms ({harm_matched} of 10 have it in their top-3 contribution differences; {imp_matched} of 10 improved ones do as well) and lm_dirichlet - exactly the features that failed the sign check - i.e. the learned combination prefers documents with *fewer* matched terms / lower LM score at similar BM25, which is sometimes right and sometimes wrong. We therefore see no systematic "better matching" mechanism; both lists are dominated by single-document moves.
* Harm is often compounded by **unjudged-as-non-relevant**: the documents that replace the lost relevant ones often look on-topic judging by their titles (e.g. PLAIN-3085, PLAIN-2680); we did not read their full text.
* Groups by candidate recall (`query_groups_by_recall.csv`):

{grp_tab}

  Mean delta stays close to 0 in every group; queries with high BM25 NDCG@10 tend to lose slightly (Spearman between delta and BM25 NDCG@10 = {f3(ex['spearman_delta_vs_bm25_ndcg'])}, weak, consistent with regression to the mean plus noise).
* Top-10 lists are selected by extreme deltas out of {len(qd)} queries, so they are partly noise; they illustrate mechanisms and should not be read as per-query "rules".

### 7c. Why pairwise accuracy and NDCG can disagree (our results)

{pa_tab}

* Ranking SVMs have the **highest** pairwise accuracy (all-pairs {f4(sv['pair_acc_micro'])}, BM25 {f4(B['pair_acc_micro'])}) yet all-pairs NDCG@10 is **lower** ({f4(sv['NDCG@10'])} vs {f4(B['NDCG@10'])}). Pairwise accuracy counts every valid pair in the 100-document list equally, whereas NDCG@10 only looks at the top 10 positions with a log discount. Only {pct(ex['share_of_valid_pairs_with_a_bm25_top10_doc'])} of valid pairs involve a document in BM25's top-10; the SVM's pairwise gain over BM25 is concentrated in the other pairs (below-top-10 pairs: {f4(pa.loc['ranksvm_all','pair_acc_pairs_entirely_below_BM25_top10'])} vs {f4(pa.loc['bm25','pair_acc_pairs_entirely_below_BM25_top10'])}; pairs with a top-10 document: {f4(pa.loc['ranksvm_all','pair_acc_pairs_with_a_BM25_top10_doc'])} vs {f4(pa.loc['bm25','pair_acc_pairs_with_a_BM25_top10_doc'])}), which NDCG@10 ignores.
* Micro pairwise accuracy is dominated by queries with many relevant documents (many pairs); NDCG@10 averages queries equally. Per query, the change in pairwise accuracy and the change in NDCG@10 are positively but only moderately related (Spearman {f3(ex['spearman_delta_ndcg_vs_delta_pairacc'])}); NDCG falls while pairwise accuracy rises in {ex['n_queries_ndcg_down_pairacc_up']} queries and the opposite happens in {ex['n_queries_ndcg_up_pairacc_down']}.
* Pointwise models have the *lowest* pairwise accuracy (linear {f4(lr['pair_acc_micro'])}, logistic {f4(lg['pair_acc_micro'])}) because their loss does not compare documents of the same query; a global regression on raw scores partly learns query difficulty rather than within-query order. Their NDCG@10 is nevertheless close to BM25's, so a *low* pairwise accuracy does not imply a bad top-10.
* Hence pairwise accuracy alone is not a sufficient evaluation; the figure `results/figures/pairacc_vs_ndcg10.png` shows the (non-monotone) relation. The NDCG@10 axis spans only {f4(mt['NDCG@10'].min())}-{f4(mt['NDCG@10'].max())}, i.e. differences are tiny.

![pairacc](results/figures/pairacc_vs_ndcg10.png)

### 7d. Pointwise vs. pairwise; all-pairs vs. sampled vs. weighted

* **Pointwise vs pairwise.** Test NDCG@10: linear {f4(lr['NDCG@10'])}, logistic {f4(lg['NDCG@10'])}, SVM all-pairs {f4(sv['NDCG@10'])}, SVM sampled {f4(ss['NDCG@10'])}; MAP: {f4(lr['MAP'])}, {f4(lg['MAP'])}, {f4(sv['MAP'])}, {f4(ss['MAP'])}. The differences between these models are far smaller than the paired-test noise (section 5.1), so **no ranking of pointwise vs pairwise by NDCG/MAP is supported**. What *is* supported: the pairwise models have clearly higher pairwise accuracy than the pointwise ones (and than BM25), as expected from the objective they optimise. Pointwise models are 1-2 orders of magnitude cheaper to train ({lr['train_s']:.2f} s and {lg['train_s']:.2f} s vs {sv['train_s']:.1f} s for all-pairs).
* **All-pairs vs sampled.** Sampling to <= {ps['sample_cap_per_query']} pairs per query keeps {pct(ps['n_pairs_ranksvm_sampled']/ps['n_valid_pairs_all'])} of the pairs, trains {sv['train_s']/ss['train_s']:.1f}x faster and uses less memory ({ss['peak_mb']:.0f} MB vs {sv['peak_mb']:.0f} MB process peak), with no loss in MAP/NDCG (test NDCG@10 {f4(ss['NDCG@10'])} vs {f4(sv['NDCG@10'])}; dev {f4(md_.loc['ranksvm_sampled','NDCG@10'])} vs {f4(md_.loc['ranksvm_all','NDCG@10'])}) and slightly higher pairwise accuracy. Sampling also reduces the dominance of queries with many pairs (pairs per query grow ~ quadratically with the number of relevant candidates), which is a plausible reason; we did not test that hypothesis separately. These differences are within noise.
* **Weighted.** Gap-weighting is **identical** to all-pairs (all MAP/NDCG/pairwise values equal; the training pairs all have gap 1 because train qrels only contain grade 1), so this experiment cannot tell whether gap-weighting helps on graded data. The rank-aware extra variant gives NDCG@10 {f4(dn['NDCG@10'])} / MAP {f4(dn['MAP'])} (test), again within noise of the others; its dev NDCG@10 was {f4(md_.loc['ranksvm_dndcgweighted_extra','NDCG@10'])}.

### 7e. Course discussion questions that the results support

*Why must pairs come from the same query?* Relevance and feature scales are query-relative (e.g. a BM25 score of 12 can be excellent for one query and poor for another), so a cross-query pair asks the model to compare scores that are not comparable and its "preference" says nothing about ranking. Ablation (C fixed to the dev-chosen value for the sampled SVM, **no tuning**, same number of pairs; `results/ablation.csv`; the reference row differs a little from the main sampled SVM only because of a different sampling seed):

{ab_tab}

Training on cross-query pairs lowers test pairwise accuracy ({f4(ab.loc[1,'test_pair_acc'])} vs {f4(ab.loc[0,'test_pair_acc'])}); its NDCG@10 is {f4(ab.loc[1,'test_NDCG@10'])} vs {f4(ab.loc[0,'test_NDCG@10'])} on test and {f4(ab.loc[1,'dev_NDCG@10'])} vs {f4(ab.loc[0,'dev_NDCG@10'])} on dev, i.e. not distinguishable at this noise level and in the opposite directions on the two splits. So our data support the claim for the *pairwise* objective, not for NDCG@10; the conceptual argument above is the main reason.

*Why no intercept?* A ranking score only matters through differences between documents of one query; an intercept adds the same constant to every document and cancels in x_i - x_j. With balanced directions the fitted intercept is {ab.loc[2,'intercept']:.4f} and the metrics are unchanged (variant C vs A: test NDCG@10 {f4(ab.loc[2,'test_NDCG@10'])} vs {f4(ab.loc[0,'test_NDCG@10'])}).

*Why balance the pair directions?* If all pairs were stored as (better - worse, +1) the data contain a single class and the SVM cannot be fitted at all (`{ab.iloc[0,-1]}`); with an intercept it could learn a trivial solution (we did not test this). Random flipping gives a symmetric problem in which the weight vector alone must separate the pairs.

*How to sample pairs for large data?* Count valid pairs first (`count_pairs`) and estimate memory; if over budget, sample uniformly without replacement within each query with a per-query cap (our rule: <= {ps['sample_cap_per_query']} pairs/query), which bounds memory at ~ n_queries x cap x features and balances query contributions. Alternatives (not run here): sample documents instead of pairs, or sample pairs with probability proportional to grade gap or rank-based weights (our extra variant weights, not samples, by rank).

## 8. Limitations and uncertainties

* One dataset and one official train/dev/test split (section 9.2 repeats the evaluation on 5 query-level folds of the same data); one seed, no repeated runs, so no estimate of seed variance. All LTR-vs-BM25 differences are within noise; conclusions about *relative* model quality are not supported.
* Features are mostly lexical variants of BM25, nearly collinear, and the baseline already uses the best of them; there is little new signal for a linear model. No non-linear model, no stemming, no semantic features, no per-query normalisation.
* Candidate recall is the dominant bottleneck ({pct(cr.loc['test','frac_queries_zero_rel_in_cand'])} of test queries have no relevant candidate). BM25 parameters and LM mu were not tuned.
* Train qrels only have grade 1 while dev/test have grades 1-2: the graded-label aspects (linear regression on graded targets, gap weighting) are barely exercised.
* Unjudged documents are treated as non-relevant, penalising on-topic but unlinked documents.
* Squared-hinge LinearSVC is used (not hinge); pairwise ties count 0.5; Wilcoxon drops zero differences.
* C plateau: for C >= 0.1 SVM results hardly change, so the chosen C is arbitrary within the plateau.
* Peak memory is the whole-process working set (Windows API sampling), not isolated to the model; timings are single measurements on one desktop machine.
* The per-query explanations (section 7b) were written by the author after reading `query_evidence.md`; they are interpretations of that evidence (the evidence itself is generated), and several are explicitly inconclusive.
* Cross-check of the metrics used `ranx`; `pytrec_eval` could not be installed (no C compiler).
"""
from report_ext import ext_section, verdict
from report_stem import stem_section
ext_text, ext_tldr = ext_section(mt, md_, tun, best_name)
stem_text, stem_tldr = stem_section(mt, md_)
NL = chr(10)
report = report.replace(NL + "## 1. Setup and reproducibility", NL + ext_tldr + NL + stem_tldr + NL + NL + "## 1. Setup and reproducibility", 1)
report = report + NL + ext_text + NL + stem_text + NL
(Path(C.ROOT) / "report.md").write_text(report, encoding="utf8")

# ---- README "Main results": regenerated from result files (dev-selected model, never the test-best one)
bs = sig[sig.model == best_name].set_index("metric")
v_nd, v_ap = verdict(bs.loc["ndcg10"]), verdict(bs.loc["ap"])
readme_block = NL.join([
    f"* Official split, test ({int(bs.loc['ndcg10', 'n_queries'])} queries). BM25: NDCG@10 {f4(B['NDCG@10'])}, MAP {f4(B['MAP'])}. "
    f"The LTR model **selected on dev** ({LABEL[best_name]}, C = {best['chosen_C'][best_name]:g}): NDCG@10 {f4(mt.loc[best_name, 'NDCG@10'])}, MAP {f4(mt.loc[best_name, 'MAP'])}. "
    f"Paired tests vs BM25 (decision rule of report section 9): NDCG@10 -> {v_nd}, AP -> {v_ap}. All seven LTR models are listed in report section 5; none was chosen on test.",
    f"* Candidate recall is the bottleneck: {pct(cr.loc['test', 'frac_queries_zero_rel_in_cand'])} of test queries have no relevant document in BM25's Top-100 (`results/candidate_recall_summary.csv`).",
    "* The BEIR NFCorpus **train** qrels contain only grade 1, so grade-gap weighting is numerically identical to all-pairs (documented, plus one extra rank-aware weighting).",
    f"* All-pairs training fits in memory ({ps['n_valid_pairs_all']:,} pairs, ~{ps['est_train_memory_gb_all_pairs']:.2f} GB estimated), so the sampled variant is a comparison, not a necessity.",
    "* **Extensions and follow-up experiments are in report sections 9 and 10** (standard-hinge SVM, 5-fold CV, feature set v2 / reduced v1, stemmed Stage 1); summaries:",
    "  " + ext_tldr[2:],
    "  " + stem_tldr[2:],
])
rp = Path(C.ROOT) / "README.md"
rd = rp.read_text(encoding="utf8")
a_, b_ = "<!-- MAIN_RESULTS_START -->", "<!-- MAIN_RESULTS_END -->"
rp.write_text(rd[: rd.index(a_) + len(a_)] + NL + readme_block + NL + rd[rd.index(b_):], encoding="utf8")
print("report.md written,", len(report.split()), "words")
