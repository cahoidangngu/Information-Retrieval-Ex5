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
| 2 | Xây đặc trưng truy vấn-tài liệu | 14 đặc trưng tính từ văn bản (không dùng nhãn) | `features.py`, `weights_table.csv`, `report.md` mục 3.2 |
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

1. Kho có 3,633 tài liệu; chia train/dev/test theo truy vấn: 2,590 / 324 / 323 (`dataset_stats.json`).
2. Tầng 1 chỉ bắt được 15.3% tài liệu liên quan trên tập test; 23.5% truy vấn không có tài liệu liên quan nào trong 100 ứng viên (`candidate_recall_summary.csv`).
3. Trên test, BM25 có NDCG@10 = 0.3134 và MAP = 0.1408 (`metrics_test.csv`).
4. Mô hình được **chọn trên dev** là `ranksvm_sampled` (`best_model.json`); trên test nó đạt NDCG@10 = 0.3117 và MAP = 0.1425.
5. Số mô hình học máy có p (t-test) < 0.05 so với BM25 trên bộ test chính thức, theo NDCG@10: 0 trên 6 (`significance.csv`).
6. Ranking SVM có độ chính xác theo cặp cao hơn BM25 (0.7426 so với 0.7358) nhưng NDCG@10 không cao hơn (`metrics_test.csv`).
7. Gán trọng số theo chênh lệch nhãn cho kết quả **giống hệt** Ranking SVM thường vì nhãn train chỉ có mức 1 (so hai hàng `ranksvm_all` và `ranksvm_gapweighted` trong `metrics_test.csv`).
8. Kiểm định chéo 5 phần trên mọi truy vấn: 4 trên 6 mô hình thỏa quy tắc "cải thiện" theo NDCG@10, nhưng số mô hình có mức tăng trung bình từ 0.01 trở lên (ngưỡng thực tế) là 0; ví dụ Ranking SVM lấy mẫu tăng 0.0035 (`ext_cv_significance.csv`).
9. Chỉ riêng việc **stemming ở tầng 1** (BM25 có stemming) đã cải thiện NDCG@10 so với BM25 gốc thêm 0.0167, và được kiểm định ủng hộ (`ext_stem_significance.csv`).
10. Khi so với **BM25 có stemming đứng một mình**, các mô hình học máy không khác biệt được ủng hộ về NDCG@10 (số dòng thỏa quy tắc: 0) (`ext_stem_significance.csv`).

---

## 3.3 Đọc từng bảng chính của `report.md`

> **Bạn cần nhớ:** Mỗi bảng có một con số chính để nhìn trước; các cột còn lại giải thích con số đó.

### Bảng A: recall của tầng 1 (`candidate_recall_summary.csv`)

> **Bạn cần nhớ:** Recall thấp của tầng 1 là trần của mọi thứ ở tầng 2.

| split | n_queries | recall@100_micro | recall@100_macro | frac_queries_zero_rel_in_cand | frac_queries_lt100_cand |
|---|---|---|---|---|---|
| train | 2590.0000 | 0.1448 | 0.2323 | 0.2255 | 0.4502 |
| dev | 324.0000 | 0.1572 | 0.2049 | 0.2747 | 0.4599 |
| test | 323.0000 | 0.1534 | 0.2331 | 0.2353 | 0.4613 |

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

| model | MAP | NDCG@5 | NDCG@10 | pair_acc_micro | C |
|---|---|---|---|---|---|
| BM25 | 0.1408 | 0.3433 | 0.3134 | 0.7358 | - |
| Hồi quy tuyến tính | 0.1408 | 0.3426 | 0.3097 | 0.7175 | - |
| Hồi quy logistic | 0.1419 | 0.3449 | 0.3125 | 0.7234 | 0.0100 |
| Ranking SVM tất cả cặp | 0.1418 | 0.3437 | 0.3093 | 0.7426 | 10.0000 |
| Ranking SVM lấy mẫu (chọn trên dev) | 0.1425 | 0.3465 | 0.3117 | 0.7437 | 0.1000 |
| Ranking SVM trọng số chênh nhãn | 0.1418 | 0.3437 | 0.3093 | 0.7426 | 10.0000 |
| Ranking SVM trọng số dNDCG (phụ) | 0.1421 | 0.3469 | 0.3127 | 0.7434 | 0.0100 |

| Cột | Cách đọc |
|---|---|
| `MAP` | trung bình AP theo truy vấn (nhãn ≥ 1 là liên quan); cao hơn = tốt hơn |
| `NDCG@5`, `NDCG@10` | chất lượng 5 hoặc 10 vị trí đầu, có dùng nhãn 0/1/2; cao hơn = tốt hơn |
| `pair_acc_micro` | tỷ lệ cặp (cùng truy vấn, khác nhãn) được xếp đúng thứ tự |
| `C` | tham số C đã chọn trên dev (BM25 và hồi quy tuyến tính không có C) |

*Số cần nhớ:* hàng BM25 là mốc so sánh. Hàng "chọn trên dev" là mô hình học máy duy nhất được phép coi là kết quả chính; các hàng khác chỉ để tham khảo, **không** được chọn "hàng đẹp nhất trên test" vì như vậy là chọn trên test.

### Bảng C: chi phí (`metrics_test.csv`)

> **Bạn cần nhớ:** Chi phí chỉ để tham khảo; điểm xếp hạng mới là kết quả chính.

| model | train_s | predict_s | peak_mb |
|---|---|---|---|
| linreg | 0.0359 | 0.0001 | 274.8398 |
| logreg | 0.1969 | 0.0001 | 279.3438 |
| ranksvm_all | 6.5558 | 0.0000 | 668.6562 |
| ranksvm_sampled | 1.9761 | 0.0000 | 387.4453 |

`train_s`: thời gian huấn luyện (giây); `predict_s`: thời gian chấm điểm tập test (giây); `peak_mb`: bộ nhớ đỉnh của cả tiến trình Python lúc huấn luyện (MB), nên gồm cả dữ liệu đã nạp, chỉ mang tính tham khảo.

### Bảng D: kiểm định so với BM25, NDCG@10, tập test (`significance.csv`)

> **Bạn cần nhớ:** Khoảng tin cậy chứa 0 nghĩa là chưa chứng minh được khác biệt.

| model | mean_delta | ci95_lo | ci95_hi | p_ttest | p_wilcoxon | cohen_dz |
|---|---|---|---|---|---|---|
| linreg | -0.0037 | -0.0097 | 0.0021 | 0.2148 | 0.3257 | -0.0692 |
| logreg | -0.0009 | -0.0066 | 0.0043 | 0.7412 | 0.7370 | -0.0184 |
| ranksvm_all | -0.0041 | -0.0122 | 0.0036 | 0.3133 | 0.4490 | -0.0562 |
| ranksvm_sampled | -0.0017 | -0.0098 | 0.0057 | 0.6550 | 0.6894 | -0.0249 |
| ranksvm_gapweighted | -0.0041 | -0.0122 | 0.0038 | 0.3133 | 0.4490 | -0.0562 |
| ranksvm_dndcgweighted_extra | -0.0007 | -0.0073 | 0.0058 | 0.8411 | 0.9297 | -0.0112 |

| Cột | Cách đọc |
|---|---|
| `mean_delta` | trung bình (NDCG@10 của mô hình trừ NDCG@10 của BM25) trên các truy vấn; dương = tốt hơn BM25 |
| `ci95_lo`, `ci95_hi` | khoảng tin cậy bootstrap 95% của `mean_delta` |
| `p_ttest`, `p_wilcoxon` | giá trị p của hai kiểm định |
| `cohen_dz` | cỡ hiệu ứng |

*Cách kết luận:* khoảng `[ci95_lo, ci95_hi]` chứa 0 ở mọi dòng và mọi p lớn hơn 0.05 → **không có cải thiện (hay suy giảm) được ủng hộ**.

### Bảng E: trọng số (`weights_table.csv`)

> **Bạn cần nhớ:** Trọng số chỉ so sánh được trong cùng một mô hình, và không nên đọc từng trọng số như một quy luật.

| feature | ranksvm_all | linreg | logreg | expected_sign | VIF |
|---|---|---|---|---|---|
| bm25_comb | 0.438 | 0.106 | 0.746 | 1.000 | 39.008 |
| lm_dirichlet | -3.135 | -0.049 | -0.297 | 1.000 | 9.776 |
| matched_terms | -0.213 | -0.081 | -0.438 | 1.000 | 9.464 |
| coverage_ratio | 0.083 | 0.117 | 0.904 | 1.000 | 4.730 |
| bm25_rank | -0.049 | 0.024 | 0.166 | -1.000 | 1.875 |

`ranksvm_all`, `linreg`, `logreg`: trọng số trên đặc trưng đã chuẩn hóa (so sánh được **trong** một mô hình, không so sánh giữa các mô hình). `expected_sign`: dấu kỳ vọng được đặt **trước** khi xem trọng số (1 = nhiều hơn là tốt, −1 = ít hơn là tốt, 0 = không kỳ vọng). `VIF`: chỉ số đa cộng tuyến; càng cao càng bị trùng thông tin với các đặc trưng khác. VIF cao của `bm25_comb` cho thấy các đặc trưng trùng thông tin nhau; đó là một nguyên nhân **có thể** của việc vài trọng số có dấu trái trực giác, nhưng chưa được chứng minh (xem bảng I).

### Bảng F: mở rộng 9.1, hinge chuẩn (`ext_hinge_metrics_test.csv`)

> **Bạn cần nhớ:** Hinge chuẩn cho kết quả gần như bằng hinge bình phương nhưng khó hội tụ.

| model | MAP | NDCG@10 | pair_acc_micro | converged |
|---|---|---|---|---|
| ranksvm_hinge_all | 0.1413 | 0.3086 | 0.7444 | False |
| ranksvm_hinge_sampled | 0.1421 | 0.3109 | 0.7448 | False |

Cột `converged` = False nghĩa là bộ giải tối ưu dừng vì hết số vòng lặp tối đa (1000 vòng) chứ chưa đạt dung sai. Bạn cần nhớ: hinge chuẩn cho kết quả gần như bằng hinge bình phương (`ranksvm_all`: NDCG@10 = 0.3093) nhưng huấn luyện bản được chọn mất 49.3 giây (so với 6.6 giây của bản hinge bình phương được chọn; hai bản được chọn với giá trị C khác nhau) và khó hội tụ; với C = 100 kết quả trên dev hỏng hẳn (NDCG@10 dev = 0.1991, `ext_hinge_tuning_dev.csv`).

### Bảng G: mở rộng 9.2, kiểm định chéo 5 phần (`ext_cv_summary.csv`)

> **Bạn cần nhớ:** Mẫu lớn hơn cho khoảng tin cậy chặt hơn, nhưng mức tăng vẫn rất nhỏ.

| model | NDCG@10_mean | NDCG@10_std | MAP_mean | MAP_std |
|---|---|---|---|---|
| bm25 | 0.3037 | 0.0130 | 0.1257 | 0.0079 |
| linreg | 0.3063 | 0.0131 | 0.1268 | 0.0080 |
| logreg | 0.3068 | 0.0124 | 0.1272 | 0.0080 |
| ranksvm_all | 0.3060 | 0.0123 | 0.1265 | 0.0073 |
| ranksvm_sampled | 0.3072 | 0.0114 | 0.1272 | 0.0074 |
| ranksvm_gapweighted | 0.3059 | 0.0123 | 0.1265 | 0.0073 |
| ranksvm_dndcgweighted_extra | 0.3071 | 0.0119 | 0.1271 | 0.0074 |

Cách đọc: `NDCG@10_mean` là trung bình qua 5 phần; `NDCG@10_std` là độ lệch chuẩn giữa 5 phần (chủ yếu do độ khó khác nhau của từng phần, ảnh hưởng đều lên mọi mô hình kể cả BM25). Kiểm định ghép cặp trên 3,237 truy vấn gộp (`ext_cv_significance.csv`):

| model | mean_delta | ci95_lo | ci95_hi | p_ttest | p_wilcoxon | p_ttest_holm |
|---|---|---|---|---|---|---|
| linreg | 0.0026 | 0.0005 | 0.0048 | 0.0164 | 0.0042 | 0.0492 |
| logreg | 0.0031 | 0.0011 | 0.0052 | 0.0022 | 0.0003 | 0.0091 |
| ranksvm_all | 0.0023 | -0.0001 | 0.0048 | 0.0612 | 0.0967 | 0.1224 |
| ranksvm_sampled | 0.0035 | 0.0013 | 0.0058 | 0.0018 | 0.0044 | 0.0091 |
| ranksvm_gapweighted | 0.0022 | -0.0002 | 0.0046 | 0.0713 | 0.0993 | 0.1224 |
| ranksvm_dndcgweighted_extra | 0.0035 | 0.0014 | 0.0056 | 0.0010 | 0.0006 | 0.0063 |

Đọc: nhiều mô hình có khoảng tin cậy nằm hẳn trên 0, tức mức tăng nhỏ nhưng đo được nhờ mẫu lớn hơn. Nhưng không mô hình nào có `mean_delta` đạt 0.01 (ngưỡng thực tế), nên **có ý nghĩa thống kê, không có ý nghĩa thực tế**.

### Bảng H: mở rộng 9.3, bộ đặc trưng v2 (`ext_featuresets_metrics_test.csv`)

> **Bạn cần nhớ:** Đặc trưng v2 tốt hơn v1 nhưng không tốt hơn việc chỉ dùng BM25 có stemming.

| NDCG@10 trên test | v1 (gốc) | v2 (thêm 5 đặc trưng) |
|---|---|---|
| Hồi quy tuyến tính | 0.3097 | 0.3202 |
| Hồi quy logistic | 0.3125 | 0.3246 |
| Ranking SVM tất cả cặp | 0.3093 | 0.3233 |
| Ranking SVM lấy mẫu | 0.3117 | 0.3234 |

Điểm mấu chốt (`ext_single_feature_reranks.csv`): chỉ cần xếp hạng lại các ứng viên **gốc** bằng đúng một đặc trưng `bm25_stem_comb` (BM25 có stemming, không học gì) đã được NDCG@10 = 0.3204. So với đó (`ext_featuresets_significance.csv`), mô hình v2 hồi quy logistic hơn 0.0042 với p (t) = 0.2293: **không có khác biệt được ủng hộ**. Nghĩa là phần lợi của v2 so với v1 có thể giải thích bằng đặc trưng stemming, không cần các đặc trưng còn lại.

### Bảng I: mở rộng 9.4, bộ v1 rút gọn (`ext_featuresets_sign_check.csv`)

> **Bạn cần nhớ:** Rút gọn VIF không làm dấu trọng số diễn giải được.

Quy trình bỏ lần lượt đặc trưng có VIF lớn nhất cho đến khi mọi VIF < 10: đã bỏ `bm25_comb` (VIF 39.0), rồi các đặc trưng khác (`ext_vif_pruning_v1.csv`). Kiểm tra dấu: các đặc trưng `lm_dirichlet` và `matched_terms` vẫn có dấu trái kỳ vọng ở cả ba mô hình: ví dụ trọng số `lm_dirichlet` của Ranking SVM sau khi rút gọn = -1.509, cột `ok_ranksvm_all` = NO. Kết luận trung thực: **rút gọn VIF không làm dấu trọng số trở nên diễn giải được.**

### Bảng J: mở rộng mục 10, tầng 1 có stemming (`ext_stem_stage1_summary.csv`, tập test)

> **Bạn cần nhớ:** Stemming ở tầng 1 là thứ thật sự giúp; phần học máy thêm vào gần như không đáng kể.

| Chỉ số (tập test) | Tầng 1 gốc | Tầng 1 có stemming |
|---|---|---|
| recall@100 gộp | 0.1534 | 0.1729 |
| tỷ lệ truy vấn không có ứng viên liên quan | 0.2353 | 0.2105 |
| NDCG@10 nếu xếp hạng lại hoàn hảo (giới hạn trên, dùng nhãn) | 0.5235 | 0.5472 |
| NDCG@10 của chính thứ tự BM25 | 0.3134 | 0.3302 |

Các mô hình học máy huấn luyện trên ứng viên có stemming (`ext_stem_metrics_test.csv`, `ext_stem_significance.csv`):

| model | mean_delta | ci95_lo | ci95_hi | p_ttest | p_wilcoxon |
|---|---|---|---|---|---|
| BM25 có stemming (đứng một mình) | 0.0167 | 0.0065 | 0.0271 | 0.0017 | 0.0048 |
| Hồi quy tuyến tính | 0.0149 | 0.0037 | 0.0260 | 0.0093 | 0.0164 |
| Hồi quy logistic | 0.0179 | 0.0070 | 0.0290 | 0.0014 | 0.0043 |
| Ranking SVM lấy mẫu | 0.0185 | 0.0061 | 0.0310 | 0.0034 | 0.0199 |

*(so với BM25 gốc; NDCG@10 trên test)*

| model | mean_delta | ci95_lo | ci95_hi | p_ttest | p_wilcoxon |
|---|---|---|---|---|---|
| Hồi quy tuyến tính | -0.0019 | -0.0086 | 0.0047 | 0.5756 | 0.3951 |
| Hồi quy logistic | 0.0012 | -0.0046 | 0.0071 | 0.6954 | 0.6499 |
| Ranking SVM lấy mẫu | 0.0018 | -0.0051 | 0.0090 | 0.6143 | 0.8977 |

*(so với BM25 có stemming đứng một mình; NDCG@10 trên test)*

Đọc bảng trên: so với BM25 **gốc**, tất cả đều tốt hơn một cách được ủng hộ; so với BM25 **có stemming**, không mô hình học máy nào khác biệt được ủng hộ. Vậy lợi ích đến từ stemming ở tầng 1.

*Lưu ý trung thực (forking paths):* stemming được thử **sau khi** thấy ở mục 9.3 rằng đặc trưng stemming có vẻ giúp trên chính các truy vấn test này. Nên đây là kiểm tra nối tiếp trên cùng dữ liệu test, **không** phải một lần lặp lại độc lập.

---

## 3.4 Vì sao recall của tầng 1 giới hạn mọi thứ

> **Bạn cần nhớ:** Tầng 2 chỉ xếp lại các ứng viên nên recall của tầng 1 chặn trần kết quả.

Tầng 2 chỉ xếp hạng lại 100 ứng viên. Nếu truy vấn không có tài liệu liên quan nào trong đó, NDCG và AP của truy vấn đó bằng 0 với **mọi** mô hình. Tỷ lệ truy vấn như vậy ở tập test là 23.5% (`candidate_recall_summary.csv`). Ngay cả xếp hạng hoàn hảo cũng chỉ đạt NDCG@10 = 0.5235 (`oracle_upper_bound.csv`). Vì thế các cải tiến ở tầng 2 có "trần" thấp, còn stemming tăng recall ở tầng 1 mới đem lại lợi ích rõ hơn.

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
