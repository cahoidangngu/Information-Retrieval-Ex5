# Theo dõi hai truy vấn thật từ đầu đến cuối

> **Bạn cần nhớ:** Hai truy vấn dưới đây (một cái tốt lên, một cái xấu đi) là **ví dụ để hiểu cơ chế**. Chúng được chọn vì nằm ở hai đầu cực trị trong 323 truy vấn test, nên **không đại diện** và không rút ra được quy luật chung từ chúng.
> Dữ liệu lấy từ `test_scores.csv.gz`, `per_query_test.csv`, `query_evidence.csv` và hai file trích `docs_vi/data_vi/*.csv` (xem cách trích ở cuối file). Mô hình "sau" là mô hình được chọn trên dev: Ranking SVM lấy mẫu cặp (`best_model.json`).

**Cách đọc:** "Nhãn" 0 = không liên quan, 1 = liên quan, 2 = liên quan nhiều. `bm25_rank` = vị trí trong danh sách ứng viên của BM25; `ltr_rank` = vị trí sau khi mô hình xếp hạng lại.

---

## Phần A. Truy vấn TỐT LÊN: "low-carb diets"

> **Bạn cần nhớ:** Truy vấn tốt lên nhờ vài tài liệu liên quan được kéo vào top 10.

### A1. Tình huống

- Có 45 tài liệu liên quan thật trong toàn bộ nhãn.
- BM25 lấy 100 ứng viên, trong đó chỉ 8 tài liệu liên quan. Tức recall của truy vấn này là 0.178 (`top10_improved.csv`).
- Số cặp (cùng truy vấn, nhãn khác nhau) có thể tạo ở truy vấn này: 736 (`per_query_test.csv`, cột `pair_total`). Truy vấn test không dùng để huấn luyện; con số chỉ cho thấy cặp được đếm thế nào.

### A2. Top 10 của BM25 (trước)

| bm25_rank | doc_id | label | bm25_comb | matched_terms | coverage_ratio | ltr_rank | title |
|---|---|---|---|---|---|---|---|
| 1 | MED-4880 | 0 | 21.20 | 3 | 1.00 | 1 | Effects of very-low-carbohydrate (horsemeat- or beef-based)  |
| 2 | MED-3149 | 0 | 7.76 | 2 | 0.67 | 10 | Four therapeutic diets: adherence and acceptability. |
| 3 | MED-3273 | 0 | 7.66 | 2 | 0.67 | 5 | The low-methionine content of vegan diets may make methionin |
| 4 | MED-4353 | 0 | 7.36 | 2 | 0.67 | 33 | Effects of dietary proteins on plasma lipoprotein levels in  |
| 5 | MED-1257 | 0 | 7.33 | 2 | 0.67 | 25 | Protein and coronary heart disease: the role of different pr |
| 6 | MED-5267 | 0 | 7.14 | 2 | 0.67 | 2 | Mediterranean and low-fat diets improve endothelial function |
| 7 | MED-4831 | 0 | 7.07 | 2 | 0.67 | 8 | Effects of plant-based diets on plasma lipids. |
| 8 | MED-3874 | 0 | 7.06 | 2 | 0.67 | 48 | Flaxseed Supplementation (not Dietary Fat Restriction) Reduc |
| 9 | MED-4885 | 0 | 7.06 | 2 | 0.67 | 50 | Flaxseed Supplementation (not Dietary Fat Restriction) Reduc |
| 10 | MED-5272 | 0 | 7.03 | 2 | 0.67 | 13 | Chronic effects of a high-fat diet enriched with virgin oliv |

Cột `bm25_comb`, `matched_terms`, `coverage_ratio` là ba trong 14 đặc trưng của mỗi tài liệu (cột `ltr_rank` cho biết tài liệu này sẽ ở đâu sau khi xếp hạng lại).

Đọc bảng: trong top 10 của BM25 **không có tài liệu nào liên quan** (cột `label` toàn 0). Vì vậy NDCG@10 của BM25 ở truy vấn này = 0.000 và AP = 0.018 (`per_query_test.csv`).

### A3. Top 10 sau khi mô hình xếp hạng lại (sau)

| ltr_rank | doc_id | label | bm25_rank | ltr_score | title |
|---|---|---|---|---|---|
| 1 | MED-4880 | 0 | 1 | 1.465 | Effects of very-low-carbohydrate (horsemeat- or beef-based)  |
| 2 | MED-5267 | 0 | 6 | -0.419 | Mediterranean and low-fat diets improve endothelial function |
| 3 | MED-2588 | 1 | 11 | -0.441 | Low-Carbohydrate Diets and All-Cause Mortality: A Systematic |
| 4 | MED-4514 | 1 | 21 | -0.457 | Low-carbohydrate diets and all-cause and cause-specific mort |
| 5 | MED-3273 | 0 | 3 | -0.477 | The low-methionine content of vegan diets may make methionin |
| 6 | MED-1134 | 0 | 25 | -0.479 | Effects of low animal protein or high-fiber diets on urine c |
| 7 | MED-1669 | 0 | 20 | -0.493 | The effect of two energy-restricted diets, a low-fructose di |
| 8 | MED-4831 | 0 | 7 | -0.501 | Effects of plant-based diets on plasma lipids. |
| 9 | MED-3557 | 0 | 59 | -0.503 | VEGETARIAN DIETS AND THE INCIDENCE OF CANCER IN A LOW-RISK P |
| 10 | MED-3149 | 0 | 2 | -0.529 | Four therapeutic diets: adherence and acceptability. |

Đọc bảng:

- Hai tài liệu **liên quan** (cột `label` bằng 1) được kéo từ vị trí BM25 11 và 21 lên vị trí 3 và 4.
- Tài liệu đứng đầu vẫn là của BM25 (không liên quan).
- NDCG@10 của truy vấn này đổi từ 0.000 lên 0.205, mức đổi +0.205 (`query_deltas_best_model.csv`); AP từ 0.018 lên 0.036.

### A4. Giải thích bằng lời

Điểm của mô hình là **tổng có trọng số của các đặc trưng**. File bằng chứng `query_evidence.csv` cho biết các đặc trưng có chênh lệch đóng góp lớn nhất giữa nhóm tài liệu mới vào top 10 và nhóm bị đẩy ra: `bm25_title`, `bm25_comb`, `lm_dirichlet`. Tiêu đề của hai tài liệu liên quan chứa chính các từ của truy vấn ("Low-Carbohydrate Diets ..."), phù hợp với việc `bm25_title` có vai trò.

**Nhưng đừng đọc quá mức:** trong 5 tài liệu mới vào top 10 thì chỉ 2 là liên quan (còn lại không liên quan). Cải thiện của truy vấn này nằm ở **hai tài liệu**. Vì BM25 không có tài liệu liên quan nào trong top 10 (NDCG@10 = 0.000), chỉ cần kéo được một vài tài liệu liên quan lên là mức thay đổi đã "trông lớn".

Vị trí của các tài liệu liên quan trong ứng viên (BM25 → sau xếp hạng lại; `query_evidence.csv`): `rel=1 BM25#11->LTR#3, rel=1 BM25#18->LTR#29, rel=1 BM25#21->LTR#4, rel=1 BM25#52->LTR#36, rel=1 BM25#53->LTR#15, rel=1 BM25#54->LTR#74, rel=1 BM25#85->LTR#72, rel=1 BM25#92->LTR#60`. Ta thấy một số tài liệu liên quan khác đi lên, một số đi xuống (xem chuỗi trên); mô hình không "hiểu" chủ đề, nó chỉ cộng các điểm.

---

## Phần B. Truy vấn XẤU ĐI: "The Difficulty of Arriving at a Vitamin D Recommendation"

> **Bạn cần nhớ:** Truy vấn xấu đi vì một hai tài liệu liên quan bị đẩy ra khỏi top 10.

### B1. Tình huống

- Có 2 tài liệu liên quan thật; cả 2 đều nằm trong ứng viên (recall truy vấn = 1.000, `top10_harmed.csv`).

### B2. Top 10 của BM25 (trước)

| bm25_rank | doc_id | label | bm25_comb | matched_terms | coverage_ratio | ltr_rank | title |
|---|---|---|---|---|---|---|---|
| 1 | MED-4872 | 0 | 13.15 | 3 | 0.60 | 3 | Safety considerations and potential interactions of vitamins |
| 2 | MED-4470 | 0 | 11.28 | 2 | 0.40 | 9 | Effects of vitamins C and E on N-nitroso compound formation, |
| 3 | MED-4573 | 2 | 11.26 | 2 | 0.40 | 22 | Low Vitamin D Status: Definition, Prevalence, Consequences a |
| 4 | MED-716 | 0 | 11.04 | 2 | 0.40 | 5 | VITAMIN D: A D-LIGHTFUL SOLUTION FOR HEALTH |
| 5 | MED-920 | 0 | 11.01 | 2 | 0.40 | 2 | Vitamin D in the treatment of pulmonary tuberculosis. |
| 6 | MED-3990 | 0 | 10.98 | 2 | 0.40 | 10 | Vitamin D supplementation for prevention of mortality in adu |
| 7 | MED-4574 | 0 | 10.94 | 2 | 0.40 | 8 | Prevalence and correlates of vitamin D deficiency in US adul |
| 8 | MED-3986 | 0 | 10.94 | 2 | 0.40 | 20 | Bioavailability of vitamin D₂ from UV-B-irradiated button mu |
| 9 | MED-3988 | 0 | 10.85 | 2 | 0.40 | 18 | Vitamin D2 Is as Effective as Vitamin D3 in Maintaining Circ |
| 10 | MED-4570 | 2 | 10.81 | 2 | 0.40 | 12 | Vitamin D supplement doses and serum 25-hydroxyvitamin D in  |

BM25 đặt hai tài liệu liên quan (nhãn 2) ở vị trí 3 và 10. NDCG@10 của BM25 = 0.484.

### B3. Top 10 sau khi xếp hạng lại (sau)

| ltr_rank | doc_id | label | bm25_rank | ltr_score | title |
|---|---|---|---|---|---|
| 1 | MED-4575 | 0 | 14 | 2.110 | Vitamin D: extraskeletal health. |
| 2 | MED-920 | 0 | 5 | 2.103 | Vitamin D in the treatment of pulmonary tuberculosis. |
| 3 | MED-4872 | 0 | 1 | 2.032 | Safety considerations and potential interactions of vitamins |
| 4 | MED-961 | 0 | 21 | 2.024 | Vitamin D(3) is more potent than vitamin D(2) in humans. |
| 5 | MED-716 | 0 | 4 | 2.016 | VITAMIN D: A D-LIGHTFUL SOLUTION FOR HEALTH |
| 6 | MED-3985 | 0 | 18 | 2.010 | Vitamin D deficiency treated by consuming UVB-irradiated mus |
| 7 | MED-714 | 0 | 11 | 1.982 | Towards prevention of vitamin D deficiency and beyond: knowl |
| 8 | MED-4574 | 0 | 7 | 1.959 | Prevalence and correlates of vitamin D deficiency in US adul |
| 9 | MED-4470 | 0 | 2 | 1.931 | Effects of vitamins C and E on N-nitroso compound formation, |
| 10 | MED-3990 | 0 | 6 | 1.930 | Vitamin D supplementation for prevention of mortality in adu |

Đọc bảng: sau khi xếp lại, **không còn tài liệu liên quan nào trong top 10**; NDCG@10 rơi từ 0.484 xuống 0.000, mức đổi -0.484.

### B4. Giải thích bằng lời

- Trong 4 tài liệu mới vào top 10, 0 tài liệu liên quan; trong 4 tài liệu bị đẩy ra có 2 tài liệu liên quan.
- Đặc trưng có chênh lệch đóng góp lớn nhất: `lm_dirichlet`, `bm25_comb`, `bm25_title`. Các tài liệu vào top 10 là các bài về vitamin D có tiêu đề khớp từ truy vấn.
- **Hai khả năng, dữ liệu không cho phép phân biệt:** (1) mô hình xếp sai; (2) các bài mới vào top 10 thực ra có liên quan về chủ đề nhưng **không được gán nhãn** trong NFCorpus (nhãn dựa trên liên kết trích dẫn; tài liệu chưa gán nhãn bị tính là không liên quan). Ta chỉ đọc tiêu đề, chưa đọc nội dung.
- Đặc trưng nổi bật ở đây (`lm_dirichlet`) là một trong những đặc trưng có **trọng số âm** trái trực giác (xem `01_kien_thuc_chuong8.md`, mục 1e), nên việc đọc "mô hình ưu tiên điều gì" từ một truy vấn riêng lẻ là thiếu căn cứ.

---

## Phần C. Hai ví dụ này nói lên điều gì (và không nói lên điều gì)

> **Bạn cần nhớ:** Hai ví dụ chỉ giúp hiểu cơ chế, không đại diện cho toàn bộ.

- Với mô hình được chọn trên dev, trên 323 truy vấn test: 67 truy vấn tốt lên, 78 xấu đi, 178 không đổi (`significance.csv`). Gần như cân bằng, đúng với kết luận "không có khác biệt được ủng hộ".
- Có 178 truy vấn không đổi điểm (`analysis_extra.json`); trong số đó 76 truy vấn không có tài liệu liên quan nào trong ứng viên, nên xếp lại thế nào điểm vẫn bằng 0.
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
