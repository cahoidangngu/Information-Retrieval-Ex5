# Theo dõi hai truy vấn thật từ đầu đến cuối

> **Bạn cần nhớ:** Hai truy vấn dưới đây (một cái tốt lên, một cái xấu đi) là **ví dụ để hiểu cơ chế**. Chúng được chọn vì nằm ở hai đầu cực trị trong [[json|dataset_stats.json|test.n_queries|d]] truy vấn test, nên **không đại diện** và không rút ra được quy luật chung từ chúng.
> Dữ liệu lấy từ `test_scores.csv.gz`, `per_query_test.csv`, `query_evidence.csv` và hai file trích `docs_vi/data_vi/*.csv` (xem cách trích ở cuối file). Mô hình "sau" là mô hình được chọn trên dev: Ranking SVM lấy mẫu cặp (`best_model.json`).

**Cách đọc:** "Nhãn" 0 = không liên quan, 1 = liên quan, 2 = liên quan nhiều. `bm25_rank` = vị trí trong danh sách ứng viên của BM25; `ltr_rank` = vị trí sau khi mô hình xếp hạng lại.

---

## Phần A. Truy vấn TỐT LÊN: "[[csv|top10_improved.csv|qid=PLAIN-1537|query|s]]"

> **Bạn cần nhớ:** Truy vấn tốt lên nhờ vài tài liệu liên quan được kéo vào top 10.

### A1. Tình huống

- Có [[csv|per_query_test.csv|qid=PLAIN-1537&model=bm25|n_rel_total|d]] tài liệu liên quan thật trong toàn bộ nhãn.
- BM25 lấy [[csv|per_query_test.csv|qid=PLAIN-1537&model=bm25|n_cand|d]] ứng viên, trong đó chỉ [[csv|per_query_test.csv|qid=PLAIN-1537&model=bm25|n_rel_in_cand|d]] tài liệu liên quan. Tức recall của truy vấn này là [[csv|top10_improved.csv|qid=PLAIN-1537|recall_at_100|.3f]] (`top10_improved.csv`).
- Số cặp (cùng truy vấn, nhãn khác nhau) có thể tạo ở truy vấn này: [[csv|per_query_test.csv|qid=PLAIN-1537&model=bm25|pair_total|,d]] (`per_query_test.csv`, cột `pair_total`). Truy vấn test không dùng để huấn luyện; con số chỉ cho thấy cặp được đếm thế nào.

### A2. Top 10 của BM25 (trước)

[[table|docs_vi/data_vi/q1537_bm25_top10.csv|bm25_rank|1;2;3;4;5;6;7;8;9;10|doc_id:s;label:d;bm25_comb:.2f;matched_terms:.0f;coverage_ratio:.2f;ltr_rank:d;title:s|.2f]]

Cột `bm25_comb`, `matched_terms`, `coverage_ratio` là ba trong [[count|weights_table.csv||d]] đặc trưng của mỗi tài liệu (cột `ltr_rank` cho biết tài liệu này sẽ ở đâu sau khi xếp hạng lại).

Đọc bảng: trong top 10 của BM25 **không có tài liệu nào liên quan** (cột `label` toàn 0). Vì vậy NDCG@10 của BM25 ở truy vấn này = [[csv|per_query_test.csv|qid=PLAIN-1537&model=bm25|ndcg10|.3f]] và AP = [[csv|per_query_test.csv|qid=PLAIN-1537&model=bm25|ap|.3f]] (`per_query_test.csv`).

### A3. Top 10 sau khi mô hình xếp hạng lại (sau)

[[table|docs_vi/data_vi/q1537_ltr_top10.csv|ltr_rank|1;2;3;4;5;6;7;8;9;10|doc_id:s;label:d;bm25_rank:d;ltr_score:.3f;title:s|.2f]]

Đọc bảng:

- Hai tài liệu **liên quan** (cột `label` bằng 1) được kéo từ vị trí BM25 [[csv|docs_vi/data_vi/q1537_ltr_top10.csv|ltr_rank=3|bm25_rank|d]] và [[csv|docs_vi/data_vi/q1537_ltr_top10.csv|ltr_rank=4|bm25_rank|d]] lên vị trí [[csv|docs_vi/data_vi/q1537_ltr_top10.csv|ltr_rank=3|ltr_rank|d]] và [[csv|docs_vi/data_vi/q1537_ltr_top10.csv|ltr_rank=4|ltr_rank|d]].
- Tài liệu đứng đầu vẫn là của BM25 (không liên quan).
- NDCG@10 của truy vấn này đổi từ [[csv|per_query_test.csv|qid=PLAIN-1537&model=bm25|ndcg10|.3f]] lên [[csv|per_query_test.csv|qid=PLAIN-1537&model=ranksvm_sampled|ndcg10|.3f]], mức đổi [[csv|query_deltas_best_model.csv|qid=PLAIN-1537|delta_ndcg10|+.3f]] (`query_deltas_best_model.csv`); AP từ [[csv|per_query_test.csv|qid=PLAIN-1537&model=bm25|ap|.3f]] lên [[csv|per_query_test.csv|qid=PLAIN-1537&model=ranksvm_sampled|ap|.3f]].

### A4. Giải thích bằng lời

Điểm của mô hình là **tổng có trọng số của các đặc trưng**. File bằng chứng `query_evidence.csv` cho biết các đặc trưng có chênh lệch đóng góp lớn nhất giữa nhóm tài liệu mới vào top 10 và nhóm bị đẩy ra: `[[csv|query_evidence.csv|qid=PLAIN-1537|top_feature_1|s]]`, `[[csv|query_evidence.csv|qid=PLAIN-1537|top_feature_2|s]]`, `[[csv|query_evidence.csv|qid=PLAIN-1537|top_feature_3|s]]`. Tiêu đề của hai tài liệu liên quan chứa chính các từ của truy vấn ("Low-Carbohydrate Diets ..."), phù hợp với việc `bm25_title` có vai trò.

**Nhưng đừng đọc quá mức:** trong [[csv|query_evidence.csv|qid=PLAIN-1537|n_entered|d]] tài liệu mới vào top 10 thì chỉ [[csv|query_evidence.csv|qid=PLAIN-1537|n_entered_relevant|d]] là liên quan (còn lại không liên quan). Cải thiện của truy vấn này nằm ở **hai tài liệu**. Vì BM25 không có tài liệu liên quan nào trong top 10 (NDCG@10 = [[csv|per_query_test.csv|qid=PLAIN-1537&model=bm25|ndcg10|.3f]]), chỉ cần kéo được một vài tài liệu liên quan lên là mức thay đổi đã "trông lớn".

Vị trí của các tài liệu liên quan trong ứng viên (BM25 → sau xếp hạng lại; `query_evidence.csv`): `[[csv|query_evidence.csv|qid=PLAIN-1537|relevant_candidate_rank_moves|s]]`. Ta thấy một số tài liệu liên quan khác đi lên, một số đi xuống (xem chuỗi trên); mô hình không "hiểu" chủ đề, nó chỉ cộng các điểm.

---

## Phần B. Truy vấn XẤU ĐI: "[[csv|top10_harmed.csv|qid=PLAIN-3085|query|s]]"

> **Bạn cần nhớ:** Truy vấn xấu đi vì một hai tài liệu liên quan bị đẩy ra khỏi top 10.

### B1. Tình huống

- Có [[csv|per_query_test.csv|qid=PLAIN-3085&model=bm25|n_rel_total|d]] tài liệu liên quan thật; cả [[csv|per_query_test.csv|qid=PLAIN-3085&model=bm25|n_rel_in_cand|d]] đều nằm trong ứng viên (recall truy vấn = [[csv|top10_harmed.csv|qid=PLAIN-3085|recall_at_100|.3f]], `top10_harmed.csv`).

### B2. Top 10 của BM25 (trước)

[[table|docs_vi/data_vi/q3085_bm25_top10.csv|bm25_rank|1;2;3;4;5;6;7;8;9;10|doc_id:s;label:d;bm25_comb:.2f;matched_terms:.0f;coverage_ratio:.2f;ltr_rank:d;title:s|.2f]]

BM25 đặt hai tài liệu liên quan (nhãn 2) ở vị trí [[csv|docs_vi/data_vi/q3085_bm25_top10.csv|bm25_rank=3|bm25_rank|d]] và [[csv|docs_vi/data_vi/q3085_bm25_top10.csv|bm25_rank=10|bm25_rank|d]]. NDCG@10 của BM25 = [[csv|per_query_test.csv|qid=PLAIN-3085&model=bm25|ndcg10|.3f]].

### B3. Top 10 sau khi xếp hạng lại (sau)

[[table|docs_vi/data_vi/q3085_ltr_top10.csv|ltr_rank|1;2;3;4;5;6;7;8;9;10|doc_id:s;label:d;bm25_rank:d;ltr_score:.3f;title:s|.2f]]

Đọc bảng: sau khi xếp lại, **không còn tài liệu liên quan nào trong top 10**; NDCG@10 rơi từ [[csv|per_query_test.csv|qid=PLAIN-3085&model=bm25|ndcg10|.3f]] xuống [[csv|per_query_test.csv|qid=PLAIN-3085&model=ranksvm_sampled|ndcg10|.3f]], mức đổi [[csv|query_deltas_best_model.csv|qid=PLAIN-3085|delta_ndcg10|+.3f]].

### B4. Giải thích bằng lời

- Trong [[csv|query_evidence.csv|qid=PLAIN-3085|n_entered|d]] tài liệu mới vào top 10, [[csv|query_evidence.csv|qid=PLAIN-3085|n_entered_relevant|d]] tài liệu liên quan; trong [[csv|query_evidence.csv|qid=PLAIN-3085|n_left|d]] tài liệu bị đẩy ra có [[csv|query_evidence.csv|qid=PLAIN-3085|n_left_relevant|d]] tài liệu liên quan.
- Đặc trưng có chênh lệch đóng góp lớn nhất: `[[csv|query_evidence.csv|qid=PLAIN-3085|top_feature_1|s]]`, `[[csv|query_evidence.csv|qid=PLAIN-3085|top_feature_2|s]]`, `[[csv|query_evidence.csv|qid=PLAIN-3085|top_feature_3|s]]`. Các tài liệu vào top 10 là các bài về vitamin D có tiêu đề khớp từ truy vấn.
- **Hai khả năng, dữ liệu không cho phép phân biệt:** (1) mô hình xếp sai; (2) các bài mới vào top 10 thực ra có liên quan về chủ đề nhưng **không được gán nhãn** trong NFCorpus (nhãn dựa trên liên kết trích dẫn; tài liệu chưa gán nhãn bị tính là không liên quan). Ta chỉ đọc tiêu đề, chưa đọc nội dung.
- Đặc trưng nổi bật ở đây (`lm_dirichlet`) là một trong những đặc trưng có **trọng số âm** trái trực giác (xem `01_kien_thuc_chuong8.md`, mục 1e), nên việc đọc "mô hình ưu tiên điều gì" từ một truy vấn riêng lẻ là thiếu căn cứ.

---

## Phần C. Hai ví dụ này nói lên điều gì (và không nói lên điều gì)

> **Bạn cần nhớ:** Hai ví dụ chỉ giúp hiểu cơ chế, không đại diện cho toàn bộ.

- Với mô hình được chọn trên dev, trên [[csv|significance.csv|model=ranksvm_sampled&metric=ndcg10|n_queries|d]] truy vấn test: [[csv|significance.csv|model=ranksvm_sampled&metric=ndcg10|n_improved|d]] truy vấn tốt lên, [[csv|significance.csv|model=ranksvm_sampled&metric=ndcg10|n_harmed|d]] xấu đi, [[csv|significance.csv|model=ranksvm_sampled&metric=ndcg10|n_unchanged|d]] không đổi (`significance.csv`). Gần như cân bằng, đúng với kết luận "không có khác biệt được ủng hộ".
- Có [[json|analysis_extra.json|n_queries_unchanged|d]] truy vấn không đổi điểm (`analysis_extra.json`); trong số đó [[json|analysis_extra.json|unchanged_with_zero_recall|d]] truy vấn không có tài liệu liên quan nào trong ứng viên, nên xếp lại thế nào điểm vẫn bằng 0.
- Hai truy vấn này nằm trong danh sách "10 tốt lên nhất" và "10 xấu đi nhất" (`top10_improved.csv`, `top10_harmed.csv`), chọn theo mức đổi cực trị, nên có phần là may rủi.
- Cơ chế chung thấy được: cải thiện thường là **một vài tài liệu liên quan** được kéo vào top 10; xấu đi thường là **một hai tài liệu liên quan** bị đẩy ra.

---

## Phụ lục: hai file trích được tạo thế nào

> **Bạn cần nhớ:** Hai bảng top 10 được trích tự động và được kiểm tra lại bằng một cách dựng độc lập.

`docs_vi/tools/build_docs.py` đọc `results/test_scores.csv.gz` (điểm từng mô hình cho từng tài liệu test), ghép với cache đặc trưng `data/processed/full/features.pkl` và tiêu đề trong dữ liệu gốc, rồi ghi `docs_vi/data_vi/*.csv`. `docs_vi/tools/check_numbers.py` dựng lại các bảng này bằng đường khác (sắp xếp độc lập) và so khớp. Thứ tự xếp hạng lại dùng cùng luật hòa điểm với dự án: điểm cao hơn trước, nếu bằng thì ai ở trên trong BM25 đứng trước.

---

## Tự kiểm tra

**Câu 1.** Vì sao BM25 có NDCG@10 bằng 0 ở truy vấn A dù có tài liệu liên quan trong 100 ứng viên?
*Đáp án:* Vì không có tài liệu liên quan nào nằm trong **10** vị trí đầu; NDCG@10 chỉ nhìn 10 vị trí đầu.

**Câu 2.** Có thể kết luận "mô hình hiểu chủ đề low-carb" từ truy vấn A không?
*Đáp án:* Không. Chỉ thấy hai tài liệu liên quan được kéo lên; có tài liệu vào top 10 không liên quan, nhiều tài liệu liên quan khác bị đẩy xuống; mô hình chỉ cộng điểm đặc trưng.

**Câu 3.** Truy vấn B xấu đi: lỗi của mô hình hay lỗi của nhãn?
*Đáp án:* Dữ liệu không phân biệt được; các bài vào top 10 có thể có liên quan về chủ đề nhưng không được gán nhãn (tính là không liên quan).

---

## Nếu chỉ có 10 phút

1. Truy vấn A: BM25 không có tài liệu liên quan trong top 10; mô hình kéo hai tài liệu liên quan lên hạng 3 và 4.
2. Truy vấn B: BM25 có hai tài liệu liên quan trong top 10; sau xếp lại không còn cái nào.
3. Cả hai là cực trị được chọn ra, không đại diện.
4. Nhiều thay đổi phụ thuộc vào nhãn thiếu: tài liệu chưa gán nhãn bị tính là không liên quan.
5. Tổng thể: số truy vấn tốt lên và xấu đi gần cân bằng, đúng với "không có cải thiện được ủng hộ".
