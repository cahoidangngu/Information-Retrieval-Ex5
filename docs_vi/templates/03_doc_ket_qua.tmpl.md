# Cách đọc kết quả

> **Bạn cần nhớ:** Kết quả trung thực của dự án là **"không có cải thiện nào của mô hình học máy được kiểm định ủng hộ"**. Con số tăng nhỏ thấy được đến từ **stemming ở tầng 1**, không phải từ phần học máy.
> Mọi con số dưới đây được đọc từ `results/` (tên file ghi bên cạnh).

---

## 3.1 Bảng ánh xạ: yêu cầu đề → việc đã làm → bằng chứng

> **Bạn cần nhớ:** Mỗi yêu cầu của đề có một việc đã làm và một file làm bằng chứng.

Đề bài (bài tập lập trình 5 trong `Programming_Exercises.pdf`, trang 5) có 9 yêu cầu:

| # | Yêu cầu của đề | Dự án đã làm gì | File / mục chứng minh |
|---|---|---|---|
| 1 | Dùng BM25 lấy Top-100 | BM25 tự cài đặt, lấy 100 ứng viên/truy vấn; báo cáo recall tầng 1 | `bm25.py`, `candidate_recall_summary.csv`, `report.md` mục 3.1 và 4 |
| 2 | Xây đặc trưng truy vấn-tài liệu | [[count|weights_table.csv||d]] đặc trưng tính từ văn bản (không dùng nhãn) | `features.py`, `weights_table.csv`, `report.md` mục 3.2 |
| 3 | Huấn luyện hồi quy pointwise | hồi quy tuyến tính và hồi quy logistic | `models.py`, hàng `linreg`, `logreg` trong `metrics_test.csv` |
| 4 | Huấn luyện Ranking SVM | Ranking SVM: tất cả cặp, lấy mẫu cặp, gán trọng số theo chênh lệch nhãn (+ một biến thể phụ) | `pairs.py`, `models.py`, `tuning_dev.csv`, `metrics_test.csv` |
| 5 | Xếp hạng lại cùng tập ứng viên | mọi mô hình xếp hạng lại **đúng** các ứng viên của BM25 | `run_models.py`, `test_scores.csv.gz` |
| 6 | Đánh giá MAP và NDCG@10 | tự viết độ đo, kiểm tra bằng ví dụ tay và đối chiếu thư viện `ranx` | `metrics.py`, `metrics_test.csv`, `crosscheck_metrics.csv` |
| 7 | So sánh độ chính xác theo cặp | cột `pair_acc_micro`, phân tích vì sao lệch với NDCG | `metrics_test.csv`, `pairacc_vs_ndcg.csv` |
| 8 | Phân tích trọng số đặc trưng | bảng + biểu đồ, kiểm tra dấu | `weights_table.csv`, `figures/feature_weights.png` |
| 9 | Phân tích truy vấn tốt lên / xấu đi | 10 truy vấn tốt lên nhất, 10 xấu đi nhất, kèm bằng chứng | `top10_improved.csv`, `top10_harmed.csv`, `query_evidence.md` |

---

## 3.2 Kết quả chính trong 10 dòng

> **Bạn cần nhớ:** Kết quả trung thực là không có cải thiện nào của mô hình học máy được kiểm định ủng hộ.

1. Kho có [[json|dataset_stats.json|n_docs|,d]] tài liệu; chia train/dev/test theo truy vấn: [[json|dataset_stats.json|train.n_queries|,d]] / [[json|dataset_stats.json|dev.n_queries|d]] / [[json|dataset_stats.json|test.n_queries|d]] (`dataset_stats.json`).
2. Tầng 1 chỉ bắt được [[csv|candidate_recall_summary.csv|split=test|recall@100_micro|.1%]] tài liệu liên quan trên tập test; [[csv|candidate_recall_summary.csv|split=test|frac_queries_zero_rel_in_cand|.1%]] truy vấn không có tài liệu liên quan nào trong 100 ứng viên (`candidate_recall_summary.csv`).
3. Trên test, BM25 có NDCG@10 = [[csv|metrics_test.csv|model=bm25|NDCG@10|.4f]] và MAP = [[csv|metrics_test.csv|model=bm25|MAP|.4f]] (`metrics_test.csv`).
4. Mô hình được **chọn trên dev** là `ranksvm_sampled` (`best_model.json`); trên test nó đạt NDCG@10 = [[csv|metrics_test.csv|model=ranksvm_sampled|NDCG@10|.4f]] và MAP = [[csv|metrics_test.csv|model=ranksvm_sampled|MAP|.4f]].
5. Số mô hình học máy có p (t-test) < 0.05 so với BM25 trên bộ test chính thức, theo NDCG@10: [[count|significance.csv|metric=ndcg10&p_ttest<0.05|d]] trên [[count|significance.csv|metric=ndcg10|d]] (`significance.csv`).
6. Ranking SVM có độ chính xác theo cặp cao hơn BM25 ([[csv|metrics_test.csv|model=ranksvm_all|pair_acc_micro|.4f]] so với [[csv|metrics_test.csv|model=bm25|pair_acc_micro|.4f]]) nhưng NDCG@10 không cao hơn (`metrics_test.csv`).
7. Gán trọng số theo chênh lệch nhãn cho kết quả **giống hệt** Ranking SVM thường vì nhãn train chỉ có mức 1 (so hai hàng `ranksvm_all` và `ranksvm_gapweighted` trong `metrics_test.csv`).
8. Kiểm định chéo 5 phần trên mọi truy vấn: [[count|ext_cv_significance.csv|metric=ndcg10&p_ttest<0.05&p_wilcoxon<0.05&ci95_lo>0|d]] trên [[count|ext_cv_significance.csv|metric=ndcg10|d]] mô hình thỏa quy tắc "cải thiện" theo NDCG@10, nhưng số mô hình có mức tăng trung bình từ 0.01 trở lên (ngưỡng thực tế) là [[count|ext_cv_significance.csv|metric=ndcg10&mean_delta>=0.01|d]]; ví dụ Ranking SVM lấy mẫu tăng [[csv|ext_cv_significance.csv|model=ranksvm_sampled&metric=ndcg10|mean_delta|.4f]] (`ext_cv_significance.csv`).
9. Chỉ riêng việc **stemming ở tầng 1** (BM25 có stemming) đã cải thiện NDCG@10 so với BM25 gốc thêm [[csv|ext_stem_significance.csv|comparison=(a) vs unstemmed BM25&model=stemmed_bm25&metric=ndcg10|mean_delta|.4f]], và được kiểm định ủng hộ (`ext_stem_significance.csv`).
10. Khi so với **BM25 có stemming đứng một mình**, các mô hình học máy không khác biệt được ủng hộ về NDCG@10 (số dòng thỏa quy tắc: [[count|ext_stem_significance.csv|comparison=(b) vs stemmed BM25 alone&metric=ndcg10&p_ttest<0.05&p_wilcoxon<0.05&ci95_lo>0|d]]) (`ext_stem_significance.csv`).

---

## 3.3 Đọc từng bảng chính của `report.md`

> **Bạn cần nhớ:** Mỗi bảng có một con số chính để nhìn trước; các cột còn lại giải thích con số đó.

### Bảng A: recall của tầng 1 (`candidate_recall_summary.csv`)

> **Bạn cần nhớ:** Recall thấp của tầng 1 là trần của mọi thứ ở tầng 2.

[[table|candidate_recall_summary.csv|split|train;dev;test|n_queries;recall@100_micro;recall@100_macro;frac_queries_zero_rel_in_cand;frac_queries_lt100_cand|.4f]]

| Cột | Cách đọc |
|---|---|
| `split` | nhóm dữ liệu (train / dev / test) |
| `n_queries` | số truy vấn trong nhóm |
| `recall@100_micro` | gộp mọi truy vấn: (tổng tài liệu liên quan nằm trong ứng viên) / (tổng tài liệu liên quan) |
| `recall@100_macro` | tính recall từng truy vấn rồi lấy trung bình |
| `frac_queries_zero_rel_in_cand` | tỷ lệ truy vấn **không** có tài liệu liên quan nào trong 100 ứng viên |
| `frac_queries_lt100_cand` | tỷ lệ truy vấn có **ít hơn** 100 ứng viên (vì ít hơn 100 tài liệu chứa từ nào đó của truy vấn) |

*Con số đầu bảng cần nhớ:* recall rất thấp; tầng 2 bị giới hạn bởi đây.

### Bảng B: kết quả test (`metrics_test.csv`)

> **Bạn cần nhớ:** Nhìn hàng BM25 làm mốc, rồi nhìn hàng được chọn trên dev.

[[table|metrics_test.csv|model|bm25;linreg;logreg;ranksvm_all;ranksvm_sampled;ranksvm_gapweighted;ranksvm_dndcgweighted_extra|MAP;NDCG@5;NDCG@10;pair_acc_micro;C|.4f|bm25=BM25;linreg=Hồi quy tuyến tính;logreg=Hồi quy logistic;ranksvm_all=Ranking SVM tất cả cặp;ranksvm_sampled=Ranking SVM lấy mẫu (chọn trên dev);ranksvm_gapweighted=Ranking SVM trọng số chênh nhãn;ranksvm_dndcgweighted_extra=Ranking SVM trọng số dNDCG (phụ)]]

| Cột | Cách đọc |
|---|---|
| `MAP` | trung bình AP theo truy vấn (nhãn ≥ 1 là liên quan); cao hơn = tốt hơn |
| `NDCG@5`, `NDCG@10` | chất lượng 5 hoặc 10 vị trí đầu, có dùng nhãn 0/1/2; cao hơn = tốt hơn |
| `pair_acc_micro` | tỷ lệ cặp (cùng truy vấn, khác nhãn) được xếp đúng thứ tự |
| `C` | tham số C đã chọn trên dev (BM25 và hồi quy tuyến tính không có C) |

*Số cần nhớ:* hàng BM25 là mốc so sánh. Hàng "chọn trên dev" là mô hình học máy duy nhất được phép coi là kết quả chính; các hàng khác chỉ để tham khảo, **không** được chọn "hàng đẹp nhất trên test" vì như vậy là chọn trên test.

### Bảng C: chi phí (`metrics_test.csv`)

> **Bạn cần nhớ:** Chi phí chỉ để tham khảo; điểm xếp hạng mới là kết quả chính.

[[table|metrics_test.csv|model|linreg;logreg;ranksvm_all;ranksvm_sampled|train_s;predict_s;peak_mb|.4f|]]

`train_s`: thời gian huấn luyện (giây); `predict_s`: thời gian chấm điểm tập test (giây); `peak_mb`: bộ nhớ đỉnh của cả tiến trình Python lúc huấn luyện (MB), nên gồm cả dữ liệu đã nạp, chỉ mang tính tham khảo.

### Bảng D: kiểm định so với BM25, NDCG@10, tập test (`significance.csv`)

> **Bạn cần nhớ:** Khoảng tin cậy chứa 0 nghĩa là chưa chứng minh được khác biệt.

[[table|significance.csv|model|linreg;logreg;ranksvm_all;ranksvm_sampled;ranksvm_gapweighted;ranksvm_dndcgweighted_extra|mean_delta;ci95_lo;ci95_hi;p_ttest;p_wilcoxon;cohen_dz|.4f||metric=ndcg10]]

| Cột | Cách đọc |
|---|---|
| `mean_delta` | trung bình (NDCG@10 của mô hình trừ NDCG@10 của BM25) trên các truy vấn; dương = tốt hơn BM25 |
| `ci95_lo`, `ci95_hi` | khoảng tin cậy bootstrap 95% của `mean_delta` |
| `p_ttest`, `p_wilcoxon` | giá trị p của hai kiểm định |
| `cohen_dz` | cỡ hiệu ứng |

*Cách kết luận:* khoảng `[ci95_lo, ci95_hi]` chứa 0 ở mọi dòng và mọi p lớn hơn 0.05 → **không có cải thiện (hay suy giảm) được ủng hộ**.

### Bảng E: trọng số (`weights_table.csv`)

> **Bạn cần nhớ:** Trọng số chỉ so sánh được trong cùng một mô hình, và không nên đọc từng trọng số như một quy luật.

[[table|weights_table.csv|feature|bm25_comb;lm_dirichlet;matched_terms;coverage_ratio;bm25_rank|ranksvm_all;linreg;logreg;expected_sign;VIF|.3f|]]

`ranksvm_all`, `linreg`, `logreg`: trọng số trên đặc trưng đã chuẩn hóa (so sánh được **trong** một mô hình, không so sánh giữa các mô hình). `expected_sign`: dấu kỳ vọng được đặt **trước** khi xem trọng số (1 = nhiều hơn là tốt, −1 = ít hơn là tốt, 0 = không kỳ vọng). `VIF`: chỉ số đa cộng tuyến; càng cao càng bị trùng thông tin với các đặc trưng khác. VIF cao của `bm25_comb` cho thấy các đặc trưng trùng thông tin nhau; đó là một nguyên nhân **có thể** của việc vài trọng số có dấu trái trực giác, nhưng chưa được chứng minh (xem bảng I).

### Bảng F: mở rộng 9.1, hinge chuẩn (`ext_hinge_metrics_test.csv`)

> **Bạn cần nhớ:** Hinge chuẩn cho kết quả gần như bằng hinge bình phương nhưng khó hội tụ.

[[table|ext_hinge_metrics_test.csv|model|ranksvm_hinge_all;ranksvm_hinge_sampled|MAP;NDCG@10;pair_acc_micro;converged|.4f||]]

Cột `converged` = False nghĩa là bộ giải tối ưu dừng vì hết số vòng lặp tối đa ([[csv|ext_hinge_metrics_test.csv|model=ranksvm_hinge_all|n_iter|d]] vòng) chứ chưa đạt dung sai. Bạn cần nhớ: hinge chuẩn cho kết quả gần như bằng hinge bình phương (`ranksvm_all`: NDCG@10 = [[csv|metrics_test.csv|model=ranksvm_all|NDCG@10|.4f]]) nhưng huấn luyện bản được chọn mất [[csv|ext_hinge_metrics_test.csv|model=ranksvm_hinge_all|train_s|.1f]] giây (so với [[csv|metrics_test.csv|model=ranksvm_all|train_s|.1f]] giây của bản hinge bình phương được chọn; hai bản được chọn với giá trị C khác nhau) và khó hội tụ; với C = 100 kết quả trên dev hỏng hẳn (NDCG@10 dev = [[csv|ext_hinge_tuning_dev.csv|model=ranksvm_hinge_all&C=100.0|NDCG@10|.4f]], `ext_hinge_tuning_dev.csv`).

### Bảng G: mở rộng 9.2, kiểm định chéo 5 phần (`ext_cv_summary.csv`)

> **Bạn cần nhớ:** Mẫu lớn hơn cho khoảng tin cậy chặt hơn, nhưng mức tăng vẫn rất nhỏ.

[[table|ext_cv_summary.csv|model|bm25;linreg;logreg;ranksvm_all;ranksvm_sampled;ranksvm_gapweighted;ranksvm_dndcgweighted_extra|NDCG@10_mean;NDCG@10_std;MAP_mean;MAP_std|.4f||]]

Cách đọc: `NDCG@10_mean` là trung bình qua 5 phần; `NDCG@10_std` là độ lệch chuẩn giữa 5 phần (chủ yếu do độ khó khác nhau của từng phần, ảnh hưởng đều lên mọi mô hình kể cả BM25). Kiểm định ghép cặp trên [[csv|ext_cv_significance.csv|model=ranksvm_sampled&metric=ndcg10|n_queries|,d]] truy vấn gộp (`ext_cv_significance.csv`):

[[table|ext_cv_significance.csv|model|linreg;logreg;ranksvm_all;ranksvm_sampled;ranksvm_gapweighted;ranksvm_dndcgweighted_extra|mean_delta;ci95_lo;ci95_hi;p_ttest;p_wilcoxon;p_ttest_holm|.4f||metric=ndcg10]]

Đọc: nhiều mô hình có khoảng tin cậy nằm hẳn trên 0, tức mức tăng nhỏ nhưng đo được nhờ mẫu lớn hơn. Nhưng không mô hình nào có `mean_delta` đạt 0.01 (ngưỡng thực tế), nên **có ý nghĩa thống kê, không có ý nghĩa thực tế**.

### Bảng H: mở rộng 9.3, bộ đặc trưng v2 (`ext_featuresets_metrics_test.csv`)

> **Bạn cần nhớ:** Đặc trưng v2 tốt hơn v1 nhưng không tốt hơn việc chỉ dùng BM25 có stemming.

| NDCG@10 trên test | v1 (gốc) | v2 (thêm 5 đặc trưng) |
|---|---|---|
| Hồi quy tuyến tính | [[csv|ext_featuresets_metrics_test.csv|feature_set=v1&model=linreg|NDCG@10|.4f]] | [[csv|ext_featuresets_metrics_test.csv|feature_set=v2&model=linreg|NDCG@10|.4f]] |
| Hồi quy logistic | [[csv|ext_featuresets_metrics_test.csv|feature_set=v1&model=logreg|NDCG@10|.4f]] | [[csv|ext_featuresets_metrics_test.csv|feature_set=v2&model=logreg|NDCG@10|.4f]] |
| Ranking SVM tất cả cặp | [[csv|ext_featuresets_metrics_test.csv|feature_set=v1&model=ranksvm_all|NDCG@10|.4f]] | [[csv|ext_featuresets_metrics_test.csv|feature_set=v2&model=ranksvm_all|NDCG@10|.4f]] |
| Ranking SVM lấy mẫu | [[csv|ext_featuresets_metrics_test.csv|feature_set=v1&model=ranksvm_sampled|NDCG@10|.4f]] | [[csv|ext_featuresets_metrics_test.csv|feature_set=v2&model=ranksvm_sampled|NDCG@10|.4f]] |

Điểm mấu chốt (`ext_single_feature_reranks.csv`): chỉ cần xếp hạng lại các ứng viên **gốc** bằng đúng một đặc trưng `bm25_stem_comb` (BM25 có stemming, không học gì) đã được NDCG@10 = [[csv|ext_single_feature_reranks.csv|split=test&model=rerank:bm25_stem_comb|NDCG@10|.4f]]. So với đó (`ext_featuresets_significance.csv`), mô hình v2 hồi quy logistic hơn [[csv|ext_featuresets_significance.csv|comparison=v2:logreg vs rerank:bm25_stem_comb&metric=ndcg10|mean_delta|.4f]] với p (t) = [[csv|ext_featuresets_significance.csv|comparison=v2:logreg vs rerank:bm25_stem_comb&metric=ndcg10|p_ttest|.4f]]: **không có khác biệt được ủng hộ**. Nghĩa là phần lợi của v2 so với v1 có thể giải thích bằng đặc trưng stemming, không cần các đặc trưng còn lại.

### Bảng I: mở rộng 9.4, bộ v1 rút gọn (`ext_featuresets_sign_check.csv`)

> **Bạn cần nhớ:** Rút gọn VIF không làm dấu trọng số diễn giải được.

Quy trình bỏ lần lượt đặc trưng có VIF lớn nhất cho đến khi mọi VIF < 10: đã bỏ `bm25_comb` (VIF [[csv|ext_vif_pruning_v1.csv|feature=bm25_comb|vif_at_drop|.1f]]), rồi các đặc trưng khác (`ext_vif_pruning_v1.csv`). Kiểm tra dấu: các đặc trưng `lm_dirichlet` và `matched_terms` vẫn có dấu trái kỳ vọng ở cả ba mô hình: ví dụ trọng số `lm_dirichlet` của Ranking SVM sau khi rút gọn = [[csv|ext_featuresets_sign_check.csv|feature_set=v1_reduced&feature=lm_dirichlet|w_ranksvm_all|.3f]], cột `ok_ranksvm_all` = [[csv|ext_featuresets_sign_check.csv|feature_set=v1_reduced&feature=lm_dirichlet|ok_ranksvm_all|s]]. Kết luận trung thực: **rút gọn VIF không làm dấu trọng số trở nên diễn giải được.**

### Bảng J: mở rộng mục 10, tầng 1 có stemming (`ext_stem_stage1_summary.csv`, tập test)

> **Bạn cần nhớ:** Stemming ở tầng 1 là thứ thật sự giúp; phần học máy thêm vào gần như không đáng kể.

| Chỉ số (tập test) | Tầng 1 gốc | Tầng 1 có stemming |
|---|---|---|
| recall@100 gộp | [[csv|ext_stem_stage1_summary.csv|stage1=unstemmed (main)&split=test|recall@100_micro|.4f]] | [[csv|ext_stem_stage1_summary.csv|stage1=Porter-stemmed&split=test|recall@100_micro|.4f]] |
| tỷ lệ truy vấn không có ứng viên liên quan | [[csv|ext_stem_stage1_summary.csv|stage1=unstemmed (main)&split=test|frac_queries_no_relevant_candidate|.4f]] | [[csv|ext_stem_stage1_summary.csv|stage1=Porter-stemmed&split=test|frac_queries_no_relevant_candidate|.4f]] |
| NDCG@10 nếu xếp hạng lại hoàn hảo (giới hạn trên, dùng nhãn) | [[csv|ext_stem_stage1_summary.csv|stage1=unstemmed (main)&split=test|oracle_NDCG@10|.4f]] | [[csv|ext_stem_stage1_summary.csv|stage1=Porter-stemmed&split=test|oracle_NDCG@10|.4f]] |
| NDCG@10 của chính thứ tự BM25 | [[csv|ext_stem_stage1_summary.csv|stage1=unstemmed (main)&split=test|stage1_NDCG@10|.4f]] | [[csv|ext_stem_stage1_summary.csv|stage1=Porter-stemmed&split=test|stage1_NDCG@10|.4f]] |

Các mô hình học máy huấn luyện trên ứng viên có stemming (`ext_stem_metrics_test.csv`, `ext_stem_significance.csv`):

[[table|ext_stem_significance.csv|model|stemmed_bm25;linreg;logreg;ranksvm_sampled|mean_delta;ci95_lo;ci95_hi;p_ttest;p_wilcoxon|.4f|stemmed_bm25=BM25 có stemming (đứng một mình);linreg=Hồi quy tuyến tính;logreg=Hồi quy logistic;ranksvm_sampled=Ranking SVM lấy mẫu|comparison=(a) vs unstemmed BM25&metric=ndcg10]]

*(so với BM25 gốc; NDCG@10 trên test)*

[[table|ext_stem_significance.csv|model|linreg;logreg;ranksvm_sampled|mean_delta;ci95_lo;ci95_hi;p_ttest;p_wilcoxon|.4f|linreg=Hồi quy tuyến tính;logreg=Hồi quy logistic;ranksvm_sampled=Ranking SVM lấy mẫu|comparison=(b) vs stemmed BM25 alone&metric=ndcg10]]

*(so với BM25 có stemming đứng một mình; NDCG@10 trên test)*

Đọc bảng trên: so với BM25 **gốc**, tất cả đều tốt hơn một cách được ủng hộ; so với BM25 **có stemming**, không mô hình học máy nào khác biệt được ủng hộ. Vậy lợi ích đến từ stemming ở tầng 1.

*Lưu ý trung thực (forking paths):* stemming được thử **sau khi** thấy ở mục 9.3 rằng đặc trưng stemming có vẻ giúp trên chính các truy vấn test này. Nên đây là kiểm tra nối tiếp trên cùng dữ liệu test, **không** phải một lần lặp lại độc lập.

---

## 3.4 Vì sao recall của tầng 1 giới hạn mọi thứ

> **Bạn cần nhớ:** Tầng 2 chỉ xếp lại các ứng viên nên recall của tầng 1 chặn trần kết quả.

Tầng 2 chỉ xếp hạng lại 100 ứng viên. Nếu truy vấn không có tài liệu liên quan nào trong đó, NDCG và AP của truy vấn đó bằng 0 với **mọi** mô hình. Tỷ lệ truy vấn như vậy ở tập test là [[csv|candidate_recall_summary.csv|split=test|frac_queries_zero_rel_in_cand|.1%]] (`candidate_recall_summary.csv`). Ngay cả xếp hạng hoàn hảo cũng chỉ đạt NDCG@10 = [[csv|oracle_upper_bound.csv||NDCG@10|.4f]] (`oracle_upper_bound.csv`). Vì thế các cải tiến ở tầng 2 có "trần" thấp, còn stemming tăng recall ở tầng 1 mới đem lại lợi ích rõ hơn.

---

## 3.5 Điều kết quả KHÔNG chứng minh

> **Bạn cần nhớ:** Phần này liệt kê những điều **không** được phép khẳng định từ kết quả.

- **Không** chứng minh mô hình học máy tốt hơn BM25 trên bộ test chính thức (khoảng tin cậy chứa 0, p lớn).
- **Không** chứng minh Ranking SVM tốt hơn hay kém hơn hồi quy: khác biệt giữa các mô hình nhỏ hơn nhiều so với độ dao động giữa các truy vấn.
- **Không** chứng minh gán trọng số theo chênh lệch nhãn vô dụng: ở đây nó trùng với "tất cả cặp" vì nhãn train chỉ có mức 1; dữ liệu này không thể kiểm tra giả thuyết đó.
- **Không** chứng minh các đặc trưng thêm của v2 vô ích: chỉ cho thấy không phát hiện được lợi ích ngoài stemming với số truy vấn test hiện có. Không có bằng chứng cũng chưa phải bằng chứng của sự bằng nhau.
- **Không** chứng minh rằng các trọng số âm của `lm_dirichlet`, `matched_terms` phản ánh một quy luật thật về xếp hạng; nguyên nhân có thể là đa cộng tuyến nhưng chưa được chứng minh (xem bảng I).
- Kết quả stemming là **khám phá nối tiếp (exploratory)**, không phải xác nhận độc lập.
- Mức tăng ở kiểm định chéo là **có ý nghĩa thống kê nhưng không có ý nghĩa thực tế** (dưới ngưỡng 0.01).
- Mọi kết quả chỉ trên **một** bộ dữ liệu (NFCorpus), một hạt giống ngẫu nhiên (seed); không ước lượng được dao động do seed.

---

## Tự kiểm tra

**Câu 1.** Mô hình nào được coi là "kết quả chính" của phần học máy, và vì sao không chọn mô hình có NDCG@10 test cao nhất?
*Đáp án:* Mô hình chọn theo NDCG@10 trên dev (Ranking SVM lấy mẫu, `best_model.json`). Chọn theo test là dùng test để chọn mô hình, làm điểm test thiên lệch lạc quan.

**Câu 2.** Kiểm định chéo cho thấy khoảng tin cậy nằm trên 0. Vì sao vẫn chưa nên nói "mô hình học máy cải thiện rõ rệt"?
*Đáp án:* Vì `mean_delta` nhỏ hơn 0.01, tức không có ý nghĩa thực tế; mẫu lớn khiến sai khác rất nhỏ cũng đo được.

**Câu 3.** Vì sao nói "lợi ích đến từ stemming chứ không phải từ phần học máy"?
*Đáp án:* BM25 có stemming đứng một mình đã tốt hơn BM25 gốc được ủng hộ, còn mô hình học máy trên cùng ứng viên không khác biệt được ủng hộ so với BM25 có stemming đứng một mình (về NDCG@10).

---

## Nếu chỉ có 10 phút

1. Recall của tầng 1 thấp là giới hạn lớn nhất.
2. Trên test chính thức: không mô hình học máy nào có cải thiện được ủng hộ so với BM25.
3. Ranking SVM thắng ở độ chính xác theo cặp nhưng không thắng ở NDCG@10.
4. Kiểm định chéo cho mức tăng nhỏ, đo được nhưng dưới ngưỡng thực tế.
5. Stemming ở tầng 1 là thứ thực sự giúp; kết quả này là khám phá nối tiếp, không phải xác nhận độc lập.
