# Hỏi đáp bảo vệ: 25 câu hỏi thường gặp và 5 câu hỏi "bẫy"

> **Bạn cần nhớ:** Giáo viên nghiêm khắc thích hỏi vào chỗ **kết quả không chứng minh được gì**. Câu trả lời tốt luôn có: (1) kết luận, (2) con số kèm tên file, (3) giới hạn của kết luận.
> Mọi con số đọc từ `results/` (tên file ghi bên cạnh).

---

## Phần 1. 25 câu hỏi

> **Bạn cần nhớ:** Câu trả lời tốt luôn có kết luận, con số kèm tên file, và giới hạn của kết luận.

### Câu 1. Hệ thống xếp hạng hai tầng là gì, vì sao không dùng một tầng?
**Đáp.** Tầng 1 (BM25) nhanh, chọn [[cfg|TOP_K|d]] ứng viên từ kho [[json|dataset_stats.json|n_docs|,d]] tài liệu. Tầng 2 (mô hình học máy) chậm hơn nhưng dùng nhiều đặc trưng, chỉ xếp hạng lại các ứng viên đó. Hai tầng vì chạy mô hình phức tạp trên cả kho thì tốn kém. Cái giá: tài liệu liên quan nằm ngoài ứng viên thì mất hẳn (xem câu 8).

### Câu 2. Vì sao cặp phải lấy trong cùng một truy vấn?
**Đáp.** Mức liên quan và thang đặc trưng phụ thuộc truy vấn: điểm BM25 cao với truy vấn này có thể thấp với truy vấn khác. So hai tài liệu của hai truy vấn khác nhau là so hai thứ không cùng thước đo. Đối chứng (`ablation.csv`): huấn luyện trên cặp khác truy vấn làm độ chính xác theo cặp trên test giảm từ [[csv|ablation.csv|variant=A within-query pairs, no intercept (reference)|test_pair_acc|.4f]] xuống [[csv|ablation.csv|variant=B cross-query pairs, no intercept (pitfall)|test_pair_acc|.4f]]. *Giới hạn:* NDCG@10 test không phân biệt được ([[csv|ablation.csv|variant=A within-query pairs, no intercept (reference)|test_NDCG@10|.4f]] so với [[csv|ablation.csv|variant=B cross-query pairs, no intercept (pitfall)|test_NDCG@10|.4f]]), nên bằng chứng thực nghiệm chỉ ủng hộ phần độ chính xác theo cặp; lý do chính là lập luận khái niệm.

### Câu 3. Vì sao Ranking SVM không có hệ số chặn (intercept)?
**Đáp.** Điểm của tài liệu là w·x + b; hiệu hai điểm là w·(x_i − x_j), hệ số b triệt tiêu (slide 93). Kiểm chứng (`ablation.csv`): thêm hệ số chặn cho kết quả hệ số chặn học được = [[csv|ablation.csv|variant=C within-query pairs, WITH intercept|intercept|.4f]] (gần 0) và NDCG@10 test gần như không đổi ([[csv|ablation.csv|variant=C within-query pairs, WITH intercept|test_NDCG@10|.4f]] so với [[csv|ablation.csv|variant=A within-query pairs, no intercept (reference)|test_NDCG@10|.4f]]).

### Câu 4. Vì sao phải cân bằng hướng của cặp?
**Đáp.** Nếu mọi cặp đều là "tốt − xấu" với nhãn +1 thì bộ phân loại chỉ thấy một lớp, không học được (slide 100); thử trong `ablation.csv` cũng báo lỗi một lớp. Dự án lật ngẫu nhiên đúng một nửa số cặp thành "xấu − tốt" với nhãn −1.

### Câu 5. Vì sao độ chính xác theo cặp tăng mà NDCG@10 không tăng?
**Đáp.** Hai độ đo nhìn khác nhau: độ chính xác theo cặp đếm mọi cặp trong 100 tài liệu như nhau, NDCG@10 chỉ nhìn 10 vị trí đầu. Số liệu (`pairacc_vs_ndcg.csv`, tập test): Ranking SVM (tất cả cặp) có độ chính xác theo cặp [[csv|pairacc_vs_ndcg.csv|model=ranksvm_all|pair_acc_all_pairs|.4f]] so với BM25 [[csv|pairacc_vs_ndcg.csv|model=bm25|pair_acc_all_pairs|.4f]], nhưng NDCG@10 là [[csv|pairacc_vs_ndcg.csv|model=ranksvm_all|NDCG@10|.4f]] so với [[csv|pairacc_vs_ndcg.csv|model=bm25|NDCG@10|.4f]]. Mức tăng nằm ở các cặp hoàn toàn dưới top 10 của BM25 ([[csv|pairacc_vs_ndcg.csv|model=ranksvm_all|pair_acc_pairs_entirely_below_BM25_top10|.4f]] so với [[csv|pairacc_vs_ndcg.csv|model=bm25|pair_acc_pairs_entirely_below_BM25_top10|.4f]]), mà NDCG@10 không nhìn tới. Chỉ [[json|analysis_extra.json|share_of_valid_pairs_with_a_bm25_top10_doc|.1%]] số cặp có dính top 10.

### Câu 6. Vì sao Ranking SVM "trọng số theo chênh lệch nhãn" giống hệt "tất cả cặp"?
**Đáp.** Nhãn ở tập train **chỉ có mức 1** ([[json|dataset_stats.json|train.levels.1|,d]] cặp nhãn, không có nhãn 2; `dataset_stats.json`), nên mọi cặp train có chênh lệch nhãn bằng 1 và trọng số đều bằng 1: mô hình y hệt. Bằng chứng: MAP của hai mô hình là [[csv|metrics_test.csv|model=ranksvm_all|MAP|.6f]] và [[csv|metrics_test.csv|model=ranksvm_gapweighted|MAP|.6f]] (`metrics_test.csv`). Vì vậy thí nghiệm này **không** thể cho biết gán trọng số theo chênh lệch nhãn có ích hay không. Dự án thêm một biến thể phụ (trọng số dNDCG theo vị trí) vì lý do này.

### Câu 7. Vì sao dùng hinge bình phương làm kết quả chính?
**Đáp.** Slide 92 viết mục tiêu bằng hinge chuẩn, còn đoạn code mẫu `LinearSVC(C=1.0, fit_intercept=False)` (slide 109) mặc định là hinge bình phương. Dự án dùng hinge bình phương cho kết quả chính (bộ giải nguyên thủy hội tụ nhanh). Hinge chuẩn được làm riêng (mục 9.1 `report.md`): [[count|ext_hinge_tuning_dev.csv|no_converge=True|d]] trên [[count|ext_hinge_tuning_dev.csv||d]] lần huấn luyện **không hội tụ** (hết số vòng lặp), và với C = 100 kết quả trên dev hỏng (NDCG@10 dev = [[csv|ext_hinge_tuning_dev.csv|model=ranksvm_hinge_all&C=100.0|NDCG@10|.4f]], `ext_hinge_tuning_dev.csv`). Kết quả test của hinge chuẩn gần bằng hinge bình phương (NDCG@10 [[csv|ext_hinge_metrics_test.csv|model=ranksvm_hinge_all|NDCG@10|.4f]] so với [[csv|metrics_test.csv|model=ranksvm_all|NDCG@10|.4f]]), nên lựa chọn này không đổi kết luận.

### Câu 8. Recall của tầng 1 là gì và của bạn là bao nhiêu?
**Đáp.** Là tỷ lệ tài liệu liên quan nằm trong 100 ứng viên. Trên tập test: gộp (micro) [[csv|candidate_recall_summary.csv|split=test|recall@100_micro|.4f]], trung bình theo truy vấn (macro) [[csv|candidate_recall_summary.csv|split=test|recall@100_macro|.4f]]; [[csv|candidate_recall_summary.csv|split=test|frac_queries_zero_rel_in_cand|.1%]] truy vấn không có tài liệu liên quan nào trong ứng viên (`candidate_recall_summary.csv`). Xếp hạng lại hoàn hảo cũng chỉ đạt NDCG@10 = [[csv|oracle_upper_bound.csv||NDCG@10|.4f]] (`oracle_upper_bound.csv`). Recall thấp một phần vì [[csv|candidate_recall_summary.csv|split=test|frac_queries_with_gt100_rel|.1%]] truy vấn có hơn 100 tài liệu liên quan (recall@100 không thể đạt 1) và một phần vì nhãn NFCorpus dựa trên trích dẫn (khác từ vựng giữa truy vấn và bài báo).

### Câu 9. Cải thiện của mô hình học máy có ý nghĩa thống kê không?
**Đáp.** Không, trên bộ test chính thức. Mô hình chọn trên dev (Ranking SVM lấy mẫu): hiệu NDCG@10 so với BM25 = [[csv|significance.csv|model=ranksvm_sampled&metric=ndcg10|mean_delta|.4f]], khoảng tin cậy 95% từ [[csv|significance.csv|model=ranksvm_sampled&metric=ndcg10|ci95_lo|.4f]] đến [[csv|significance.csv|model=ranksvm_sampled&metric=ndcg10|ci95_hi|.4f]] (chứa 0), p (t) = [[csv|significance.csv|model=ranksvm_sampled&metric=ndcg10|p_ttest|.4f]], p (Wilcoxon) = [[csv|significance.csv|model=ranksvm_sampled&metric=ndcg10|p_wilcoxon|.4f]] (`significance.csv`). Số mô hình có p (t) < 0.05: [[count|significance.csv|metric=ndcg10&p_ttest<0.05|d]] trên [[count|significance.csv|metric=ndcg10|d]]. Không được kết luận "không khác biệt" thành "bằng nhau": mẫu chỉ [[json|dataset_stats.json|test.n_queries|d]] truy vấn.

### Câu 10. "Forking paths" (đường rẽ) là gì, và nó ảnh hưởng đến kết quả nào của bạn?
**Đáp.** Là việc thử nhiều hướng phân tích, chọn hướng sau khi đã nhìn kết quả, làm p-value đẹp hơn thực tế. Ở dự án: stemming ở tầng 1 được thử **sau khi** mục 9.3 cho thấy đặc trưng stemming có vẻ giúp trên chính các truy vấn test. Kết quả của mục 10 (BM25 có stemming hơn BM25 gốc [[csv|ext_stem_significance.csv|comparison=(a) vs unstemmed BM25&model=stemmed_bm25&metric=ndcg10|mean_delta|.4f]] NDCG@10, p (t) = [[csv|ext_stem_significance.csv|comparison=(a) vs unstemmed BM25&model=stemmed_bm25&metric=ndcg10|p_ttest|.4f]], `ext_stem_significance.csv`) vì vậy là **khám phá nối tiếp (exploratory)**, không phải xác nhận độc lập. Ngoài ra bảng so sánh ở mục 9.3 có nhiều phép so sánh nên p đã hiệu chỉnh Holm quan trọng hơn p thô.

### Câu 11. Vì sao chia theo truy vấn, không chia theo dòng?
**Đáp.** Mọi tài liệu của một truy vấn phải ở cùng một tập, nếu không thông tin của truy vấn test lọt vào train (rò rỉ). Dự án dùng cách chia có sẵn: train [[json|dataset_stats.json|train.n_queries|,d]], dev [[json|dataset_stats.json|dev.n_queries|d]], test [[json|dataset_stats.json|test.n_queries|d]] truy vấn, không giao nhau (`assert_query_disjoint()`, có kiểm thử `test_no_query_in_two_splits_detects_leak`). Kiểm định chéo cũng chia 5 phần theo mã truy vấn.

### Câu 12. Vì sao chọn C trên dev, và C nào được chọn?
**Đáp.** C điều khiển độ "cứng" của mô hình; chọn trên train sẽ luôn thiên về mô hình bám sát train, chọn trên test là gian lận. Dự án chọn C theo NDCG@10 trên dev; Ranking SVM tất cả cặp chọn C = [[csv|metrics_test.csv|model=ranksvm_all|C|g]], bản lấy mẫu C = [[csv|metrics_test.csv|model=ranksvm_sampled|C|g]] (`metrics_test.csv`). Đường NDCG@10 trên dev gần như phẳng khi C tăng: ở C = [[csv|tuning_dev.csv|model=ranksvm_all&C=0.1|C|g]] là [[csv|tuning_dev.csv|model=ranksvm_all&C=0.1|NDCG@10|.4f]] và ở C = [[csv|tuning_dev.csv|model=ranksvm_all&C=100.0|C|g]] là [[csv|tuning_dev.csv|model=ranksvm_all&C=100.0|NDCG@10|.4f]] (`tuning_dev.csv`), nên giá trị C được chọn trong vùng phẳng gần như tùy ý.

### Câu 13. Vì sao chuẩn hóa chỉ được fit trên train?
**Đáp.** Nếu fit trên test thì trung bình/độ lệch chuẩn của test lọt vào quá trình xử lý (rò rỉ). Dự án lưu thống kê chuẩn hóa vào `scaler_stats.csv` và có kiểm thử so khớp với dữ liệu train (`test_saved_scaler_matches_train_split_if_results_exist`).

### Câu 14. Pointwise hay pairwise tốt hơn trong dự án này?
**Đáp.** Dữ liệu không cho kết luận. Test NDCG@10: hồi quy tuyến tính [[csv|metrics_test.csv|model=linreg|NDCG@10|.4f]], hồi quy logistic [[csv|metrics_test.csv|model=logreg|NDCG@10|.4f]], Ranking SVM tất cả cặp [[csv|metrics_test.csv|model=ranksvm_all|NDCG@10|.4f]], lấy mẫu [[csv|metrics_test.csv|model=ranksvm_sampled|NDCG@10|.4f]]; chênh lệch giữa chúng nhỏ hơn nhiều so với độ dao động giữa các truy vấn. Điều thấy được: Ranking SVM có độ chính xác theo cặp cao hơn (mô hình học đúng thứ được huấn luyện), và các mô hình pointwise huấn luyện nhanh hơn rõ rệt ([[csv|metrics_test.csv|model=linreg|train_s|.2f]] giây so với [[csv|metrics_test.csv|model=ranksvm_all|train_s|.1f]] giây).

### Câu 15. Tất cả cặp, lấy mẫu cặp, trọng số: đánh đổi thế nào?
**Đáp.** Tất cả cặp: [[json|pair_stats.json|n_valid_pairs_all|,d]] cặp, bộ nhớ ước lượng [[json|pair_stats.json|est_train_memory_gb_all_pairs|.2f]] GB, vẫn nằm trong ngân sách [[json|pair_stats.json|memory_budget_gb|.1f]] GB nên dùng được (`pair_stats.json`). Lấy mẫu tối đa [[json|pair_stats.json|sample_cap_per_query|d]] cặp/truy vấn còn [[json|pair_stats.json|n_pairs_ranksvm_sampled|,d]] cặp, huấn luyện [[csv|metrics_test.csv|model=ranksvm_sampled|train_s|.1f]] giây so với [[csv|metrics_test.csv|model=ranksvm_all|train_s|.1f]] giây, mà NDCG@10 không kém ([[csv|metrics_test.csv|model=ranksvm_sampled|NDCG@10|.4f]] so với [[csv|metrics_test.csv|model=ranksvm_all|NDCG@10|.4f]]). Khi dữ liệu lớn hơn, lấy mẫu có kiểm soát bộ nhớ là cách làm; lấy mẫu còn giảm việc các truy vấn có nhiều cặp áp đảo (giả thuyết này chưa được kiểm tra riêng).

### Câu 16. Vì sao IDCG tính từ toàn bộ nhãn chứ không chỉ 100 ứng viên?
**Đáp.** Để hệ thống bị phạt vì không lấy được tài liệu liên quan ở tầng 1. Nếu IDCG chỉ tính trên ứng viên thì recall thấp sẽ bị che đi (đó là lỗi "bỏ qua recall của tầng 1" trong slide 119).

### Câu 17. Tài liệu không có nhãn xử lý thế nào?
**Đáp.** Tính là **không liên quan** (nhãn 0) ở mọi nơi: huấn luyện, MAP, NDCG, độ chính xác theo cặp. Điều này phạt oan các tài liệu đúng chủ đề nhưng chưa được gán nhãn (nhãn NFCorpus dựa trên liên kết trích dẫn). Ví dụ trong `04_theo_doi_mot_truy_van.md`, truy vấn xấu đi có thể một phần là do hiện tượng này.

### Câu 18. Trọng số âm của `lm_dirichlet` nghĩa là gì?
**Đáp.** Không nên hiểu là "khớp truy vấn thì bị phạt". Đặc trưng này tương quan dương với nhãn (hệ số Spearman [[csv|weights_table.csv|feature=lm_dirichlet|spearman_with_label_train|.3f]]) nhưng có trọng số Ranking SVM [[csv|weights_table.csv|feature=lm_dirichlet|ranksvm_all|.3f]]. Một nguyên nhân có thể là đa cộng tuyến (chỉ là giả thuyết): VIF của `bm25_comb` là [[csv|weights_table.csv|feature=bm25_comb|VIF|.1f]] (`weights_table.csv`); nếu đúng vậy thì các đặc trưng trùng nhau có thể bù trừ lẫn nhau. Dù lý do là gì, từng trọng số riêng lẻ không nên được đọc như một quy luật (xem câu 19: rút gọn VIF cũng không làm dấu trở nên hợp lý).

### Câu 19. Rút gọn VIF có làm trọng số dễ hiểu hơn không?
**Đáp.** Không. Sau khi bỏ lần lượt các đặc trưng đến khi mọi VIF < 10, `lm_dirichlet` vẫn có trọng số Ranking SVM [[csv|ext_featuresets_sign_check.csv|feature_set=v1_reduced&feature=lm_dirichlet|w_ranksvm_all|.3f]] (dấu trái kỳ vọng, cột `ok_ranksvm_all` = [[csv|ext_featuresets_sign_check.csv|feature_set=v1_reduced&feature=lm_dirichlet|ok_ranksvm_all|s]]; `ext_featuresets_sign_check.csv`). Lý do chưa được kiểm tra; một giả thuyết là VIF thấp chỉ loại bỏ trùng lặp tuyến tính.

### Câu 20. Kiểm định t và Wilcoxon khác nhau thế nào, vì sao dùng cả hai?
**Đáp.** t kiểm tra **trung bình** hiệu có khác 0; Wilcoxon bỏ các hiệu bằng 0 và kiểm tra thứ hạng của các hiệu còn lại, ít nhạy với ngoại lệ. Chúng có thể cho p khác nhau, nhất là với AP có nhiều hiệu bằng 0. Dùng cả hai cùng khoảng tin cậy bootstrap làm quy tắc chặt: chỉ gọi "cải thiện" khi cả hai p < 0.05 **và** khoảng tin cậy không chứa 0.

### Câu 21. Ý nghĩa thống kê khác ý nghĩa thực tế thế nào, lấy ví dụ từ kết quả?
**Đáp.** Trong kiểm định chéo ([[csv|ext_cv_significance.csv|model=ranksvm_sampled&metric=ndcg10|n_queries|,d]] truy vấn), Ranking SVM lấy mẫu có hiệu NDCG@10 = [[csv|ext_cv_significance.csv|model=ranksvm_sampled&metric=ndcg10|mean_delta|.4f]], khoảng tin cậy từ [[csv|ext_cv_significance.csv|model=ranksvm_sampled&metric=ndcg10|ci95_lo|.4f]] đến [[csv|ext_cv_significance.csv|model=ranksvm_sampled&metric=ndcg10|ci95_hi|.4f]], p (t) = [[csv|ext_cv_significance.csv|model=ranksvm_sampled&metric=ndcg10|p_ttest|.4f]]: có ý nghĩa thống kê nhưng nhỏ hơn nhiều so với ngưỡng thực tế 0.01 mà dự án đặt ra. Mẫu lớn khiến sai khác rất nhỏ cũng "đo được" nhưng người dùng sẽ không nhận ra.

### Câu 22. Vì sao làm kiểm định chéo, và nó cho thấy gì?
**Đáp.** Bộ test chính thức chỉ có [[json|dataset_stats.json|test.n_queries|d]] truy vấn nên kiểm định yếu. Kiểm định chéo 5 phần theo truy vấn trên mọi truy vấn cho mẫu lớn hơn. Kết quả: [[count|ext_cv_significance.csv|metric=ndcg10&p_ttest<0.05&p_wilcoxon<0.05&ci95_lo>0|d]] trên [[count|ext_cv_significance.csv|metric=ndcg10|d]] mô hình thỏa quy tắc "cải thiện" nhưng mức tăng dưới ngưỡng thực tế. Bộ test chính thức vẫn là bảng kết quả chính.

### Câu 23. Làm sao đảm bảo đặc trưng không dùng nhãn (không rò rỉ)?
**Đáp.** `stage1_and_features()` và `compute_v2_new()` không có tham số nhãn; nhãn chỉ được ghép **sau** khi đặc trưng đã tính xong (trong `run_features.py`). Có kiểm thử đọc chữ ký hàm (`test_features_do_not_depend_on_labels_and_have_no_label_argument`, `test_no_label_argument`). Thống kê kho (độ hiếm IDF, độ dài trung bình) dùng văn bản của mọi tài liệu nhưng không dùng nhãn.

### Câu 24. Độ đo tự viết có đúng không?
**Đáp.** Có ba lớp kiểm tra: (1) ví dụ tính tay trong `test_metrics.py`; (2) đối chiếu với thư viện `ranx` trên chạy thật: chênh lệch lớn nhất cho BM25 là [[csv|crosscheck_metrics.csv|model=bm25|max_abs_diff|.1e]] (`crosscheck_metrics.csv`), tức trùng đến sai số làm tròn; (3) `pytrec_eval` không cài được vì cần trình biên dịch C, nên chỉ đối chiếu với `ranx`.

### Câu 25. Nếu làm tiếp, em sẽ làm gì?
**Đáp.** (Đây là hướng đề xuất, **chưa được thử** nên chưa có kết quả.) (1) Tăng recall của tầng 1 (đó là nút thắt chính, xem câu 8), ví dụ mở rộng truy vấn hoặc truy hồi dày. (2) Làm lại các thí nghiệm với nhiều hạt giống và nhiều bộ dữ liệu để ước lượng độ dao động. (3) Thử mô hình phi tuyến để nắm tương tác giữa đặc trưng. (4) Nếu có dữ liệu nhãn nhiều mức ở tập train, kiểm tra lại gán trọng số theo chênh lệch nhãn (hiện không kiểm tra được). (5) Lặp lại kết quả stemming trên dữ liệu độc lập để có xác nhận.

---

## Phần 2. 5 câu hỏi "bẫy" (đáp án trung thực: "dữ liệu không ủng hộ điều đó")

> **Bạn cần nhớ:** Với câu hỏi bẫy, câu trả lời đúng là "dữ liệu không ủng hộ điều đó" kèm lý do.

### Bẫy 1. "Ranking SVM tốt hơn hồi quy rõ rệt, đúng không?"
**Đáp.** **Dữ liệu không ủng hộ.** Chênh lệch NDCG@10 giữa Ranking SVM ([[csv|metrics_test.csv|model=ranksvm_sampled|NDCG@10|.4f]]) và hồi quy tuyến tính ([[csv|metrics_test.csv|model=linreg|NDCG@10|.4f]]) nhỏ và không được kiểm định riêng cho thấy khác biệt; cả hai đều không khác BM25 một cách được ủng hộ (`significance.csv`). Cái thấy được chỉ là độ chính xác theo cặp cao hơn.

### Bẫy 2. "Stemming đã được chứng minh là cải thiện một cách độc lập, đúng không?"
**Đáp.** **Dữ liệu không ủng hộ phát biểu "độc lập".** Kết quả được kiểm định ủng hộ trên tập test này (`ext_stem_significance.csv`), nhưng stemming được thử **sau khi** thấy gợi ý trên chính các truy vấn test (đường rẽ). Cần xác nhận trên dữ liệu độc lập.

### Bẫy 3. "Trọng số lớn nhất nghĩa là đặc trưng quan trọng nhất, đúng không?"
**Đáp.** **Không.** Trọng số lớn nhất của Ranking SVM là của `lm_dirichlet` ([[csv|weights_table.csv|feature=lm_dirichlet|ranksvm_all|.3f]]), âm và trái trực giác; các đặc trưng bị đa cộng tuyến (VIF của `bm25_comb` là [[csv|weights_table.csv|feature=bm25_comb|VIF|.1f]]) nên độ lớn từng trọng số không phản ánh độ quan trọng. Ngoài ra trọng số chỉ so sánh được trong cùng một mô hình trên đặc trưng đã chuẩn hóa.

### Bẫy 4. "Gán trọng số theo chênh lệch nhãn không giúp ích gì, đúng không?"
**Đáp.** **Dữ liệu không cho phép kết luận.** Trong dự án nó trùng với "tất cả cặp" vì nhãn train chỉ có mức 1 (xem câu 6); thí nghiệm này không kiểm tra được giả thuyết đó.

### Bẫy 5. "Các đặc trưng mới của v2 (bigram, khoảng cách từ) giúp xếp hạng, đúng không?"
**Đáp.** **Dữ liệu không ủng hộ.** v2 tốt hơn v1 (NDCG@10 test hồi quy logistic: [[csv|ext_featuresets_metrics_test.csv|feature_set=v2&model=logreg|NDCG@10|.4f]] so với [[csv|ext_featuresets_metrics_test.csv|feature_set=v1&model=logreg|NDCG@10|.4f]]), nhưng chỉ cần xếp hạng lại bằng đặc trưng stemming đơn lẻ đã được [[csv|ext_single_feature_reranks.csv|split=test&model=rerank:bm25_stem_comb|NDCG@10|.4f]]; so với đó, v2 không khác biệt được ủng hộ (hiệu [[csv|ext_featuresets_significance.csv|comparison=v2:logreg vs rerank:bm25_stem_comb&metric=ndcg10|mean_delta|.4f]], p (t) = [[csv|ext_featuresets_significance.csv|comparison=v2:logreg vs rerank:bm25_stem_comb&metric=ndcg10|p_ttest|.4f]]). Không thấy bằng chứng cho lợi ích của bigram và khoảng cách từ ngoài stemming.

---

## Tự kiểm tra

**Câu 1.** Khi giáo viên hỏi "mô hình của em có tốt hơn BM25 không?", ba ý nào nên có trong câu trả lời?
*Đáp án:* Kết luận (không có cải thiện được kiểm định ủng hộ), con số kèm file (`significance.csv`), và giới hạn (mẫu nhỏ, khoảng tin cậy chứa 0; không phải bằng chứng của sự bằng nhau).

**Câu 2.** Vì sao không được nói "stemming độc lập cải thiện"?
*Đáp án:* Vì nó được thử sau khi thấy gợi ý trên chính các truy vấn test (forking paths); chỉ là khám phá nối tiếp.

**Câu 3.** Vì sao "gán trọng số theo chênh lệch nhãn" trùng với "tất cả cặp"?
*Đáp án:* Nhãn train chỉ có mức 1 nên mọi cặp chênh nhau 1 và có cùng trọng số.

---

## Nếu chỉ có 10 phút

1. Thuộc lòng 5 con số (đọc từ file): recall@100, NDCG@10 của BM25 và của mô hình chọn trên dev, khoảng tin cậy của hiệu, số cặp.
2. Thuộc 3 lý do kỹ thuật: cặp cùng truy vấn, không hệ số chặn, cân bằng hướng.
3. Giải thích được vì sao độ chính xác theo cặp tăng mà NDCG@10 không.
4. Luôn nói giới hạn: không có cải thiện được ủng hộ; stemming là khám phá nối tiếp.
5. Với câu "bẫy": trả lời "dữ liệu không ủng hộ" và nêu lý do.
