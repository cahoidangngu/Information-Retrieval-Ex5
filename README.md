# Two-stage Learning-to-Rank on BEIR NFCorpus

BM25 Top-100 -> 14 query-document features -> pointwise (linear / logistic regression) and Ranking-SVM
(all pairs / sampled pairs / grade-gap-weighted) re-ranking of the **same** candidate sets, evaluated with MAP, NDCG@5/10
and pairwise accuracy, with paired significance tests, feature-weight analysis and improved/harmed-query analysis.

**Read `report.md` first** (setup, dataset, method, results, significance, analysis a–e, pitfall table, limitations).

**Hướng dẫn tiếng Việt cho người mới học (chưa học Chương 8):** bắt đầu từ [docs_vi/00_doc_truoc.md](docs_vi/00_doc_truoc.md). Bộ tài liệu gồm kiến thức Chương 8 từ số 0, bản đồ file, cách đọc kết quả, theo dõi một truy vấn thật, cách chạy/đọc code, 25 câu hỏi bảo vệ và bảng thuật ngữ. Mọi con số trong đó được đọc tự động từ `results/` và kiểm tra bằng `python -I docs_vi/tools/check_numbers.py`.

## Reproduce

```bash
conda env create -f environment.yml          # python 3.11 + numpy/pandas/scipy/scikit-learn/matplotlib/pytest (+ranx, optional)
conda activate ltr
bash run_all.sh                              # everything from raw data; ~30 min on the dev machine (see results/total_runtime_seconds.txt)
# or, without activating: PYTHON=/path/to/env/python bash run_all.sh
# quick smoke test on a random 60-queries-per-split subset (writes results_smoke/, no report):
bash run_all.sh --smoke
```

`run_all.sh` = download NFCorpus (BEIR mirror, sha256 logged) -> `src/run_features.py` (BM25 Top-100, features,
candidate recall) -> `pytest tests` -> `src/run_models.py` (training, C tuning on dev, test evaluation, significance)
-> `src/analysis.py` (weights, per-query, figures) -> `src/ablation.py` -> `src/crosscheck.py` (ranx)
-> **extensions** `src/run_hinge.py`, `src/run_featuresets.py`, `src/run_cv.py`, `src/run_stemmed_stage1.py` -> `src/make_report.py`.
Total wall-clock of the last full run: `results/total_runtime_seconds.txt` (including tests and the extensions). Commands, versions,
seeds and per-step timings: `results/run_log.txt`. Seed = 42 (`src/config.py`).

On Windows run the script from Git Bash. The scripts are run with `python -I` (isolated mode) so that nothing from the
working directory or downloaded data is importable; each entry script adds `src/` to `sys.path` explicitly.

## Layout

| path | content |
|---|---|
| `src/config.py` | paths, seed, K=100, BM25 k1/b, C grid, pair-memory budget and per-query pair cap |
| `src/data.py` | download / load / split guards / dataset statistics |
| `src/bm25.py`, `src/features.py` | own BM25, TF-IDF, LM-Dirichlet; feature table (no labels used) |
| `src/pairs.py` | within-query pair construction, balancing, sampling, memory estimate |
| `src/models.py` | linear / logistic regression, Ranking SVM (LinearSVC, no intercept), timing, memory sampling |
| `src/metrics.py`, `src/stats.py` | hand-written MAP/NDCG/pairwise accuracy; paired t, Wilcoxon, bootstrap CI, Holm |
| `src/run_features.py`, `run_models.py`, `analysis.py`, `ablation.py`, `crosscheck.py` | original pipeline steps (results unchanged by later additions) |
| `src/porter.py` | Porter stemmer written from scratch (verified on 64 examples from Porter's paper) |
| `src/features_v2.py`, `src/run_featuresets.py` | extension: feature set v2 (stemmed BM25, bigrams, term distances, BM25 k1=2/b=.5); v1 vs v1_reduced (VIF<10) vs v2 |
| `src/ltr_runner.py` | shared train / tune / evaluate helpers (pair variants incl. hinge, VIF pruning) for the extension scripts |
| `src/run_hinge.py` | extension: standard-hinge Ranking SVM (dual CD) next to squared hinge |
| `src/run_cv.py` | extension: query-level 5-fold cross-validation over all 3,237 queries |
| `src/stage1_eval.py`, `src/run_stemmed_stage1.py` | extension: Porter-stemmed Stage 1 (Top-100), recall/oracle comparison, retraining on the new candidates |
| `src/report_ext.py`, `src/make_report.py` | generate report.md (sections 9-10 from `ext_*` files) and the "Main results" block of this README |
| `results/` | CSVs of all metrics, per-query scores, weights, tuning grid, significance, figures, logs |
| `data/raw`, `data/processed` | downloaded data and cached feature table (not needed in the deliverable) |

## Main results (details and sources in `report.md`)

<!-- MAIN_RESULTS_START -->
* Official split, test (323 queries). BM25: NDCG@10 0.3134, MAP 0.1408. The LTR model **selected on dev** (RankSVM sampled, C = 0.1): NDCG@10 0.3117, MAP 0.1425. Paired tests vs BM25 (decision rule of report section 9): NDCG@10 -> no supported difference, AP -> no supported difference. All seven LTR models are listed in report section 5; none was chosen on test.
* Candidate recall is the bottleneck: 23.5% of test queries have no relevant document in BM25's Top-100 (`results/candidate_recall_summary.csv`).
* The BEIR NFCorpus **train** qrels contain only grade 1, so grade-gap weighting is numerically identical to all-pairs (documented, plus one extra rank-aware weighting).
* All-pairs training fits in memory (977,151 pairs, ~0.33 GB estimated), so the sampled variant is a comparison, not a necessity.
* **Extensions and follow-up experiments are in report sections 9 and 10** (standard-hinge SVM, 5-fold CV, feature set v2 / reduced v1, stemmed Stage 1); summaries:
  **Extensions (section 9; decision rule = paired t AND Wilcoxon p<0.05 AND CI excluding 0):** (i) standard-hinge Ranking SVM: test NDCG@10 0.3086 (all pairs) / 0.3109 (sampled) vs squared hinge 0.3093 / 0.3117; 10 of 12 hinge fits hit the iteration limit. (ii) 5-fold query-level CV over 3,237 queries: 4 of 6 models improve and 0 of 6 degrade NDCG@10 vs BM25 (largest mean difference 0.0035, below the 0.01 practical threshold). (iii) Feature set v2: supported improvement over v1 for 4 of 4 models, over BM25 for 2 of 4, and over a stemmed-BM25-only rerank for 0 of 4 (a non-zero count there would mean a supported difference in either direction). (iv) Reduced v1 (all VIF < 10): weights are still NOT sign-interpretable.
  **Stemmed Stage 1 (section 10):** Recall@100 (micro, test) 0.1534 -> 0.1729, queries without a relevant candidate 0.235 -> 0.211; stemmed BM25 alone vs unstemmed BM25: NDCG@10 improvement supported, AP improvement supported. Learned models on the stemmed candidates beat unstemmed BM25 for 3 of 3 (NDCG@10), but differ from stemmed BM25 alone in a supported way for 0 of 3 (NDCG@10) and 1 of 3 (AP).
<!-- MAIN_RESULTS_END -->
