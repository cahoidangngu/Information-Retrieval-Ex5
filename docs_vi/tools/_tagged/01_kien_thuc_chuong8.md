# Kiến thức Chương 8 (Học xếp hạng) từ số 0

> **Cách đọc các con số:** số lấy từ file kết quả viết giống file gốc: dấu chấm là dấu thập phân (ví dụ 0.5), dấu phẩy ngăn cách hàng nghìn (ví dụ 1,000). Tên file đặt trong dấu ``.
> Mỗi phần mở đầu bằng "Bạn cần nhớ".

---

## 1a. Xếp hạng là gì, và vì sao cần hai tầng?

**Bạn cần nhớ:** Xếp hạng = sắp xếp tài liệu theo mức độ liên quan với truy vấn. Làm hai tầng vì tầng 1 nhanh nhưng thô, tầng 2 chậm nhưng tinh.

- **Truy vấn (query):** câu người dùng hỏi, ví dụ "low-carb diets".
- **Tài liệu (document):** một bài báo trong kho.
- **Xếp hạng (ranking):** trả về danh sách tài liệu, cái liên quan nhất ở đầu.
- **Nhãn liên quan (relevance label):** con người ghi cho mỗi cặp (truy vấn, tài liệu): 0 = không liên quan, 1 = có liên quan, 2 = liên quan nhiều.

**Ví dụ thủ thư.** Kho có 3,633 bài (`dataset_stats.json`). Thủ thư (BM25) đếm từ khóa và rút nhanh 100 cuốn "có vẻ đúng" cho mỗi câu hỏi. Sau đó một người giúp việc thông minh hơn (mô hình học máy) xếp lại 100 cuốn đó cho khéo hơn. Người giúp việc **không** được tìm thêm sách ngoài 100 cuốn.

- **BM25:** công thức chấm điểm dựa trên số lần từ truy vấn xuất hiện trong tài liệu, có điều chỉnh theo độ dài tài liệu.
- **Ứng viên (candidate):** 100 tài liệu mà BM25 chọn ở tầng 1.
- **Xếp hạng lại (re-ranking):** sắp xếp lại đúng 100 ứng viên đó ở tầng 2.

**Hệ quả quan trọng:** nếu một tài liệu liên quan **không** nằm trong 100 ứng viên, tầng 2 không cứu được. Trong dự án này, trên tập test, chỉ 15.3% tài liệu liên quan nằm trong 100 ứng viên, và 23.5% truy vấn **không có tài liệu liên quan nào** trong ứng viên (`candidate_recall_summary.csv`). Đây là giới hạn lớn nhất của cả dự án (xem mục 1f).

---

## 1b. "Đặc trưng" (feature) là gì?

**Bạn cần nhớ:** Đặc trưng = một con số mô tả mức "khớp" giữa một truy vấn và một tài liệu. Mô hình học máy chỉ nhìn thấy các con số này, không đọc chữ.

Dự án dùng 14 đặc trưng (tên trong `weights_table.csv`, cột `feature`):

| Đặc trưng | Ý nghĩa đơn giản |
|---|---|
| `bm25_title`, `bm25_body`, `bm25_comb` | điểm BM25 tính trên tiêu đề / nội dung / gộp cả hai |
| `tfidf_cos_title`, `tfidf_cos_body` | độ giống nhau (cosin) giữa vector từ của truy vấn và tài liệu |
| `lm_dirichlet` | xác suất tài liệu "sinh ra" truy vấn (mô hình ngôn ngữ, làm mượt Dirichlet) |
| `query_len` | số từ của truy vấn (sau khi bỏ từ dừng) |
| `doc_len`, `title_len` | số từ của nội dung / tiêu đề |
| `matched_terms` | số từ khác nhau của truy vấn tìm thấy trong tài liệu |
| `coverage_ratio` | `matched_terms` chia cho số từ khác nhau của truy vấn |
| `sum_idf_matched`, `max_idf_matched` | tổng / lớn nhất của độ "hiếm" (IDF) các từ khớp |
| `bm25_rank` | vị trí của tài liệu trong danh sách 100 ứng viên (1 = đầu) |

**Ví dụ thật.** Truy vấn `low-carb diets` (mã PLAIN-1537). Hai tài liệu trong 100 ứng viên (file `docs_vi/data_vi/example_pair_features.csv`, trích từ cache đặc trưng `data/processed/full/features.pkl`):

| | Tài liệu A | Tài liệu B |
|---|---|---|
| Tiêu đề | Effects of very-low-carbohydrate (horsemeat- or beef-based)  | Low-Carbohydrate Diets and All-Cause Mortality: A Systematic |
| Nhãn (label) | 0 (không liên quan) | 1 (liên quan) |
| `bm25_comb` | 21.20 | 7.01 |
| `matched_terms` | 3 | 2 |
| `coverage_ratio` | 1.000 | 0.667 |

Đọc bảng: tài liệu A khớp nhiều từ hơn và có điểm BM25 cao hơn, nên BM25 xếp A đầu bảng; nhưng theo nhãn thì B mới liên quan. Đây chính là chỗ tầng 2 hy vọng sửa được. (Ví dụ minh họa; nó **không** nói mô hình có sửa tốt được hay không, xem `04_theo_doi_mot_truy_van.md`.)

**Chuẩn hóa (standardization):** các đặc trưng có thang rất khác nhau. Ví dụ trên tập train (`scaler_stats.csv`): trung bình của `bm25_comb` là 6.06 (độ lệch chuẩn 2.71), còn trung bình của `doc_len` là 152.7 (độ lệch chuẩn 61.9). Dự án trừ trung bình và chia độ lệch chuẩn, **chỉ tính trên tập train** (`scaler_stats.csv`). Nhờ vậy các trọng số mới so sánh được với nhau trong cùng một mô hình.

---

## 1c. Học từng điểm (pointwise) và học theo cặp (pairwise)

**Bạn cần nhớ:** Pointwise: dự đoán nhãn của *từng* tài liệu rồi sắp xếp theo dự đoán. Pairwise: học "tài liệu nào nên đứng trên tài liệu nào".

**Ví dụ nhỏ.** Một truy vấn có 3 tài liệu A, B, C với nhãn thật 2, 0, 1.

*Pointwise (hồi quy):* mô hình dự đoán điểm cho từng tài liệu, mục tiêu là làm điểm gần nhãn. Giả sử dự đoán 1.5, 0.5, 0.8 (số bịa để minh họa). Sai số bình phương: A: 0.25, B: 0.25, C: 0.04. Mô hình chỉ cố làm các sai số này nhỏ; nó không quan tâm thứ tự có đúng không.

*Pairwise:* từ nhãn, ta suy ra các "ưu tiên": A hơn B, A hơn C, C hơn B. Mô hình được thưởng khi điểm(A) > điểm(C) > điểm(B), bất kể điểm tuyệt đối là bao nhiêu.

| | Pointwise | Pairwise |
|---|---|---|
| Đơn vị học | một (truy vấn, tài liệu) | một cặp tài liệu cùng truy vấn |
| Mục tiêu | nhãn | tài liệu nào hơn |
| Trong dự án | hồi quy tuyến tính, hồi quy logistic | Ranking SVM |

- **Hồi quy tuyến tính (linear regression):** dự đoán nhãn 0/1/2 như một số thực.
- **Hồi quy logistic (logistic regression):** dự đoán xác suất "liên quan" (nhãn 1 trở lên là liên quan).
- **Ranking SVM:** mô hình pairwise, xem mục 1e.

---

## 1d. Cặp ưu tiên (preference pair) được tạo thế nào?

**Bạn cần nhớ:** Chỉ tạo cặp trong **cùng một truy vấn** và **chỉ khi nhãn khác nhau**.

**Ví dụ của khóa học (slide 116).** Một truy vấn có 4 tài liệu: d1 nhãn 2, d2 nhãn 2, d3 nhãn 1, d4 nhãn 0.

- Cặp hợp lệ: d1 ≻ d3, d1 ≻ d4, d2 ≻ d3, d2 ≻ d4, d3 ≻ d4. Tổng cộng **5** cặp.
- **Không** tạo cặp (d1, d2) vì nhãn bằng nhau: không có ưu tiên nào để học.
- Cặp chênh nhãn lớn nhất: (d1, d4) và (d2, d4) (chênh 2).

**Vì sao không tạo cặp giữa hai truy vấn khác nhau?** Điểm BM25 12 có thể rất tốt cho truy vấn này nhưng kém cho truy vấn khác. So sánh chéo truy vấn là so sánh những thứ không cùng thước đo.

**Số cặp tăng rất nhanh (slide 98).** Với 1,000 tài liệu có tối đa 499,500 cặp. Vì thế cần kiểm soát bộ nhớ.

**Số cặp thật của dự án (`pair_stats.json`):**

- Tập train có 2,440 truy vấn (có ứng viên), 167,902 dòng (truy vấn, tài liệu).
- Số cặp hợp lệ nếu dùng **tất cả**: 977,151.
- Ước lượng bộ nhớ khi huấn luyện: 0.33 GB, so với ngân sách 2.0 GB, nên **dùng được tất cả cặp**.
- Biến thể "lấy mẫu": tối đa 200 cặp/truy vấn, còn 283,577 cặp.

**Cân bằng hướng cặp (slide 100).** Nếu mọi cặp đều ghi dạng "tốt − xấu" với nhãn +1, bộ phân loại chỉ thấy một lớp và không học được. Dự án lật ngẫu nhiên đúng một nửa số cặp thành "xấu − tốt" với nhãn −1.

---

## 1e. Ranking SVM = điểm tuyến tính trên hiệu của đặc trưng

**Bạn cần nhớ:** Điểm của tài liệu = tổng (trọng số × đặc trưng). Ranking SVM chọn trọng số sao cho tài liệu tốt hơn có điểm cao hơn **ít nhất một đoạn "lề" (margin)**.

**Ví dụ của khóa học (slide 115).** Đặc trưng theo thứ tự: BM25, khớp tiêu đề, độ mới. Trọng số w = (0.4, 1.5, 0.5).

- Tài liệu A: đặc trưng (3, 1, 0.8), nhãn 2. Điểm: 0.4×3 + 1.5×1 + 0.5×0.8 = 3.1.
- Tài liệu B: đặc trưng (5, 0, 0.4), nhãn 0. Điểm: 0.4×5 + 1.5×0 + 0.5×0.4 = 2.2.
- Thứ tự dự đoán: A đứng trên B (đúng, vì nhãn A cao hơn).
- **Lề (margin)** = điểm A − điểm B = 0.9.
- **Mất mát hinge (hinge loss)** = max(0, 1 − lề) = 0.1 (lề nhỏ hơn 1 nên vẫn bị phạt một chút)

Tính thêm: vector hiệu z = x_A − x_B = (−2, 1, 0.4). Điểm của z = w·z = 0.9, đúng bằng lề. Đó là ý chính: **Ranking SVM chỉ cần nhìn hiệu đặc trưng**, nên hằng số cộng chung (hệ số chặn, intercept) bị triệt tiêu: s_i − s_j = w·(x_i − x_j). Vì vậy dự án đặt `fit_intercept=False`.

**Hinge chuẩn và hinge bình phương (squared hinge), 3 câu:**

1. Hinge chuẩn phạt max(0, 1 − lề); ở ví dụ trên là 0.1.
2. Hinge bình phương phạt max(0, 1 − lề)², ở ví dụ trên là 0.01: phạt nhẹ khi lệch ít, nặng khi lệch nhiều, và dễ tối ưu hơn.
3. Khóa học viết mục tiêu bằng hinge chuẩn (slide 92) nhưng đoạn code mẫu `LinearSVC(C=1.0, fit_intercept=False)` (slide 109) mặc định dùng hinge bình phương; dự án dùng hinge bình phương cho kết quả chính và có thí nghiệm hinge chuẩn riêng (mục 9.1 của `report.md`).

**Trọng số thật** (đã chuẩn hóa; `weights_table.csv`):

| feature | ranksvm_all | linreg | logreg |
|---|---|---|---|
| bm25_comb | 0.438 | 0.106 | 0.746 |
| lm_dirichlet | -3.135 | -0.049 | -0.297 |
| matched_terms | -0.213 | -0.081 | -0.438 |
| query_len | 0.000 | 0.005 | 0.005 |

Chú ý `query_len` có trọng số Ranking SVM đúng bằng 0.0: độ dài truy vấn giống nhau cho mọi tài liệu của cùng truy vấn nên triệt tiêu khi lấy hiệu. Hai đặc trưng `lm_dirichlet` và `matched_terms` có trọng số **âm** dù về trực giác "nhiều từ khớp hơn thì tốt hơn". Một nguyên nhân có thể là các đặc trưng **tương quan mạnh** với nhau (xem cột `VIF` trong `weights_table.csv`), nhưng đây mới là giả thuyết: khi đã bỏ bớt đặc trưng để mọi VIF nhỏ hơn 10 thì dấu âm vẫn còn (mục 9.4 của `report.md`). Dù lý do là gì, **không nên đọc từng trọng số như một quy luật**.

---

## 1f. Các độ đo, bắt đầu từ số 0

**Bạn cần nhớ:** MAP và NDCG@10 chấm **chất lượng vị trí** trong danh sách; độ chính xác theo cặp chấm **mọi cặp** như nhau; recall@100 chấm **tầng 1**.

### Precision và AP / MAP
- **Precision@k:** trong k tài liệu đầu, bao nhiêu phần là liên quan.
- Ví dụ: danh sách có nhãn liên quan theo thứ tự 1, 0, 1, 0. Precision@3 = 2/3 = 0.667.
- **Average Precision (AP):** lấy Precision tại **mỗi vị trí có tài liệu liên quan**, cộng lại, chia cho **tổng số tài liệu liên quan trong toàn bộ nhãn** (kể cả cái không nằm trong danh sách).
- Ví dụ: liên quan ở vị trí 1 (P = 1/1) và vị trí 3 (P = 2/3); tổng số tài liệu liên quan thật là 3. AP = (1/1 + 2/3)/3 = 0.5556.
- **MAP:** trung bình AP trên mọi truy vấn. Dự án nhị phân hóa nhãn: nhãn ≥ 1 là liên quan.

### NDCG@10
- **Gain (lợi ích):** 2^nhãn − 1. Nhãn 2 → 3, nhãn 1 → 1, nhãn 0 → 0.
- **DCG:** cộng gain, mỗi vị trí bị chia cho log2(vị trí + 1), vị trí càng sâu càng bị "giảm giá".
- **IDCG:** DCG của danh sách **lý tưởng**. Dự án tính IDCG từ **toàn bộ** nhãn của truy vấn (không chỉ 100 ứng viên).
- **NDCG = DCG / IDCG**, nằm giữa 0 và 1.
- Ví dụ: xếp hạng có nhãn 2, 0, 1; nhãn thật của truy vấn là {2, 1, 0}. DCG = 3/log2(2) + 0 + 1/log2(4) = 3.5000. Danh sách lý tưởng 2, 1, 0 có IDCG = 3/log2(2) + 1/log2(3) = 3.6309. NDCG = 0.9639.

### Độ chính xác theo cặp (pairwise accuracy)
- Lấy mọi cặp cùng truy vấn có nhãn khác nhau; đếm tỷ lệ cặp mà tài liệu nhãn cao hơn có điểm cao hơn.
- Ví dụ: nhãn (2, 1, 0), điểm mô hình (3, 1, 2). Cặp (nhãn 2, nhãn 1): 3 > 1 đúng. Cặp (2, 0): 3 > 2 đúng. Cặp (1, 0): 1 > 2 sai. Độ chính xác = 2/3 = 0.667.

### Recall@100 của tầng 1 (candidate recall)
- Trong các tài liệu liên quan thật của một truy vấn, bao nhiêu phần nằm trong 100 ứng viên.
- Ví dụ: 20 tài liệu liên quan, 5 nằm trong ứng viên → recall = 0.25.
- Thật (tập test, `candidate_recall_summary.csv`): recall@100 gộp (micro) = 0.1534; trung bình theo truy vấn (macro) = 0.2331.
- Nếu có ground truth hoàn hảo xếp lại đúng 100 ứng viên (giới hạn trên, dùng nhãn test, chỉ để tham khảo), NDCG@10 cao nhất cũng chỉ là 0.5235 (`oracle_upper_bound.csv`).

### Vì sao độ chính xác theo cặp tăng mà NDCG@10 không tăng?
Độ chính xác theo cặp đếm **mọi** cặp trong 100 tài liệu như nhau. NDCG@10 chỉ nhìn **10 vị trí đầu**. Một mô hình có thể sắp đúng nhiều cặp ở phía dưới nhưng không cải thiện top 10.

Số liệu thật (`pairacc_vs_ndcg.csv`, tập test):

| model | NDCG@10 | pair_acc_all_pairs | pair_acc_pairs_with_a_BM25_top10_doc | pair_acc_pairs_entirely_below_BM25_top10 |
|---|---|---|---|---|
| BM25 | 0.3134 | 0.7358 | 0.8468 | 0.6640 |
| Ranking SVM (all pairs) | 0.3093 | 0.7426 | 0.8473 | 0.6750 |

Đọc bảng: Ranking SVM có độ chính xác theo cặp cao hơn BM25, nhưng NDCG@10 **thấp hơn**. Mức tăng chủ yếu nằm ở các cặp hoàn toàn nằm dưới top 10 của BM25 (cột cuối), còn cột "có tài liệu trong top 10 của BM25" gần như không đổi. Chỉ 39.3% số cặp hợp lệ có dính top 10 (`analysis_extra.json`).

---

## 1g. Train / dev / test, chọn C, rò rỉ dữ liệu và các lỗi hay gặp

**Bạn cần nhớ:** Huấn luyện trên train, chọn tham số trên dev, chạm test **một lần duy nhất**; chia theo **truy vấn**, không chia theo dòng.

- **Train:** dữ liệu để mô hình học. **Dev (validation):** dữ liệu để chọn tham số. **Test:** dữ liệu chấm điểm cuối.
- Số truy vấn (`dataset_stats.json`): train 2,590, dev 324, test 323. Ba nhóm **không có truy vấn chung** (có kiểm tra tự động).
- **Tham số C** của SVM: C lớn = phạt lỗi nặng, mô hình bám dữ liệu train chặt hơn; C nhỏ = mô hình "mềm" hơn. Lưới của dự án: [0.001, 0.01, 0.1, 1, 10, 100]; chọn C theo NDCG@10 trên dev. Với Ranking SVM (all pairs), NDCG@10 trên dev theo C (`tuning_dev.csv`):

| C | NDCG@10 |
|---|---|
| 0.001 | 0.2798 |
| 0.01 | 0.2804 |
| 0.1 | 0.2805 |
| 1.0 | 0.2805 |
| 10.0 | 0.2805 |
| 100.0 | 0.2805 |

  Đường này gần như phẳng từ giá trị C thứ ba của lưới trở lên, nên C được chọn trong vùng phẳng gần như tùy ý.
- **Chạm test một lần:** nếu chọn C hay chọn mô hình theo test thì điểm test không còn trung thực.
- **Rò rỉ dữ liệu (leakage):** thông tin của test "lọt" vào quá trình huấn luyện, làm kết quả đẹp giả. Ví dụ: chuẩn hóa dùng trung bình của cả test; chia ngẫu nhiên các dòng khiến cùng một truy vấn nằm ở cả train lẫn test.
- **Nhãn train đặc biệt:** nhãn ở tập train **chỉ có** mức 1 (110,575 cặp có nhãn; không có nhãn 2), còn dev có 521 nhãn 2 (`dataset_stats.json`). Hậu quả: Ranking SVM "gán trọng số theo chênh lệch nhãn" luôn chênh 1, nên **giống hệt** Ranking SVM thường (kiểm chứng: hai dòng trong `metrics_test.csv` bằng nhau).

**Các lỗi hay gặp (slide 119) và cách dự án tránh:**

| Lỗi trong slide | Cách dự án tránh |
|---|---|
| Tạo cặp giữa các truy vấn khác nhau | `build_pairs()` chỉ tạo cặp trong nhóm cùng truy vấn và có `assert` (test: `test_pairs_same_query_different_label_balanced`) |
| Chia ngẫu nhiên theo dòng | Chia theo truy vấn, có kiểm tra `assert_query_disjoint()` |
| Dùng cặp nhãn bằng nhau | Chỉ lấy cặp có nhãn khác nhau (có `assert`) |
| Chuẩn hóa fit trên test | `StandardScaler` chỉ fit trên train (test kiểm tra) |
| Quên cân bằng hướng cặp | Lật đúng một nửa số cặp |
| Tạo mọi cặp không kiểm soát bộ nhớ | `count_pairs()` + `estimate_memory_gb()` trước khi dựng cặp |
| Chọn C bằng test | Chọn C trên dev; mô hình tốt nhất cũng chọn trên dev |
| Chỉ đánh giá bằng độ chính xác phân loại | Báo cáo MAP, NDCG@5/10 cạnh độ chính xác theo cặp |
| Bỏ qua recall của tầng 1 | Có bảng recall@100 và giới hạn trên (mục 1f) |
| Dùng nhãn click không hiệu chỉnh thiên lệch | Không áp dụng: dự án dùng nhãn của con người |

---

## 1h. Kiểm định thống kê bằng lời đơn giản

**Bạn cần nhớ:** Kiểm định trả lời "chênh lệch này có thể chỉ là may rủi không?". "Có ý nghĩa thống kê" **khác** "đáng kể trong thực tế".

Dự án so từng truy vấn của mô hình với BM25: **hiệu NDCG@10 theo từng truy vấn** (mô hình trừ BM25).

- **Kiểm định t ghép cặp (paired t-test):** kiểm tra trung bình của các hiệu có khác 0 không.
- **Kiểm định Wilcoxon (signed-rank):** bỏ các hiệu bằng 0, xếp hạng độ lớn các hiệu còn lại, kiểm tra xem hiệu dương/âm có lệch hẳn về một phía không. Không giả định phân phối chuẩn.
- **Khoảng tin cậy bootstrap 95% (bootstrap CI):** lấy mẫu lại các truy vấn có hoàn lại nhiều lần (10,000 lần), mỗi lần tính trung bình hiệu; lấy khoảng 95% ở giữa. Nếu khoảng này chứa 0 thì chưa chắc có khác biệt.
- **Giá trị p (p-value):** xác suất thấy chênh lệch lớn như vậy (hoặc lớn hơn) **nếu thực ra không có khác biệt**. p nhỏ → khó là may rủi. p **không** phải xác suất "mô hình tốt hơn".
- **Cỡ hiệu ứng (Cohen's d_z):** trung bình hiệu chia độ lệch chuẩn của hiệu; cho biết khác biệt lớn đến đâu so với độ dao động giữa các truy vấn.
- **Ý nghĩa thống kê vs thực tế:** có thể có p rất nhỏ nhưng chênh lệch chỉ vài phần nghìn điểm NDCG, người dùng không nhận ra. Dự án coi chênh lệch dưới 0.01 NDCG@10 là không đáng kể (một quy ước do dự án chọn, ghi trong `report.md`, mục 5.1).
- **Quy tắc quyết định của dự án:** chỉ gọi là "cải thiện" khi **cả** kiểm định t **và** Wilcoxon cho p thô < 0.05 **và** khoảng tin cậy bootstrap không chứa 0. Giá trị p đã hiệu chỉnh Holm (Holm-adjusted, hiệu chỉnh khi làm nhiều so sánh) luôn được in kèm.

**Ví dụ 1 (`significance.csv`): mô hình được chọn trên dev (Ranking SVM lấy mẫu) so với BM25, NDCG@10 trên test.**
Hiệu trung bình = -0.0017; khoảng tin cậy 95% = từ -0.0098 đến 0.0057; p (t) = 0.6550; p (Wilcoxon) = 0.6894. Khoảng chứa 0 và p lớn → **không có khác biệt được ủng hộ**. Số truy vấn cải thiện / xấu đi / không đổi: 67 / 78 / 178.

**Ví dụ 2 (`ext_stem_significance.csv`): BM25 có stemming so với BM25 gốc, NDCG@10 trên test.**
Hiệu trung bình = 0.0167; khoảng tin cậy 95% = từ 0.0065 đến 0.0271; p (t) = 0.0017; p (Wilcoxon) = 0.0048. Cả hai p < 0.05 và khoảng không chứa 0 → theo quy tắc, **cải thiện được ủng hộ** (nhưng chênh lệch vẫn dưới ngưỡng thực tế 0.01 hay không thì phải so với con số trên).

---

## Tự kiểm tra

**Câu 1.** Với 4 tài liệu có nhãn (2, 2, 1, 0) trong cùng một truy vấn có bao nhiêu cặp ưu tiên hợp lệ, và vì sao cặp hai tài liệu nhãn 2 bị loại?
*Đáp án:* 5 cặp. Hai tài liệu nhãn 2 có nhãn bằng nhau nên không có ưu tiên để học.

**Câu 2.** NDCG@10 và độ chính xác theo cặp khác nhau ở điểm nào, và vì sao cái này tăng cái kia có thể không tăng?
*Đáp án:* NDCG@10 chỉ nhìn 10 vị trí đầu với trọng số giảm dần; độ chính xác theo cặp đếm mọi cặp trong 100 tài liệu như nhau. Mô hình có thể sửa nhiều cặp ở phía dưới mà không đổi top 10 (xem bảng ở mục 1f).

**Câu 3.** Một mô hình có p (t) = 0.01 và p (Wilcoxon) = 0.5. Theo quy tắc của dự án có gọi là cải thiện không?
*Đáp án:* Không. Quy tắc đòi **cả hai** p < 0.05 và khoảng tin cậy không chứa 0.

---

## Nếu chỉ có 10 phút

1. Hai tầng: BM25 lấy 100 ứng viên, mô hình học máy xếp hạng lại; tài liệu liên quan nằm ngoài 100 ứng viên thì không cứu được.
2. Đặc trưng là con số mô tả mức khớp truy vấn-tài liệu; Ranking SVM học trên **hiệu** đặc trưng của các cặp cùng truy vấn, nhãn khác nhau.
3. Ví dụ slide: điểm A = 3.1, B = 2.2, lề 0.9, hinge 0.1.
4. MAP/NDCG@10 chấm vị trí; độ chính xác theo cặp chấm mọi cặp; chúng có thể lệch nhau.
5. Chọn C trên dev, chạm test một lần; "cải thiện" chỉ khi cả hai kiểm định p < 0.05 **và** khoảng tin cậy không chứa 0.
