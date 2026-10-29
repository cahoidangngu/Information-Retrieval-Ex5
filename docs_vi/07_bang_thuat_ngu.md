# Bảng thuật ngữ

> **Bạn cần nhớ:** Bảng này để **tra cứu**. Mỗi dòng: thuật ngữ tiếng Việt | thuật ngữ tiếng Anh | nghĩa trong một dòng | nơi xuất hiện trong dự án.

## Dữ liệu và bài toán

> **Bạn cần nhớ:** Hiểu truy vấn, tài liệu, nhãn, ứng viên và hai tầng là hiểu đề bài.

| Tiếng Việt | English | Nghĩa một dòng | Xuất hiện ở |
|---|---|---|---|
| truy vấn | query | câu người dùng hỏi (ở đây là tiêu đề video dinh dưỡng) | `dataset_stats.json`, `per_query_test.csv` |
| tài liệu | document | một bài báo trong kho | `data.py` |
| kho văn bản | corpus | toàn bộ tài liệu dùng để tìm kiếm | `data.py` |
| nhãn liên quan | relevance label | điểm con người gán cho cặp (truy vấn, tài liệu): 0, 1, 2 | `dataset_stats.json` |
| không có nhãn = không liên quan | unjudged = non-relevant | tài liệu chưa được gán nhãn bị tính như không liên quan | `report.md` mục 2 |
| xếp hạng | ranking | sắp tài liệu theo mức liên quan giảm dần | toàn dự án |
| ứng viên | candidate | tài liệu mà tầng 1 chọn ra | `candidate_recall.csv` |
| tầng 1 / tầng 2 | stage 1 / stage 2 | tầng 1: BM25 chọn ứng viên; tầng 2: mô hình học máy xếp hạng lại | `features.py`, `run_models.py` |
| xếp hạng lại | re-ranking | sắp xếp lại đúng tập ứng viên của tầng 1 | `run_models.py` |
| học xếp hạng | learning to rank (LTR) | dùng học máy để học hàm chấm điểm cho xếp hạng | Chương 8 |

## Tầng 1 và xử lý văn bản

> **Bạn cần nhớ:** Tầng 1 chỉ chọn ứng viên; recall của nó là trần của tầng 2.

| Tiếng Việt | English | Nghĩa một dòng | Xuất hiện ở |
|---|---|---|---|
| BM25 | BM25 | công thức chấm điểm dựa trên tần suất từ và độ dài tài liệu | `bm25.py` |
| TF-IDF | TF-IDF | trọng số từ: tần suất trong tài liệu nhân độ hiếm của từ | `bm25.py` |
| IDF | inverse document frequency | độ hiếm của một từ trong kho | `bm25.py`, đặc trưng `sum_idf_matched` |
| từ dừng | stop word | từ quá phổ biến (the, of, ...) bị bỏ đi | `tokenize()` |
| stemming | stemming | đưa từ về gốc (diets, diet → cùng một từ gốc) | `porter.py`, mục 10 `report.md` |
| bộ stemmer Porter | Porter stemmer | thuật toán stemming kinh điển, tự cài đặt | `porter.py` |
| recall của tầng 1 | candidate recall (Recall@100) | tỷ lệ tài liệu liên quan nằm trong 100 ứng viên | `candidate_recall_summary.csv` |
| giới hạn trên (oracle) | oracle upper bound | điểm cao nhất có thể nếu xếp hạng lại hoàn hảo bằng nhãn thật | `oracle_upper_bound.csv` |

## Đặc trưng và mô hình

> **Bạn cần nhớ:** Mô hình chỉ thấy các con số đặc trưng; Ranking SVM học trên hiệu đặc trưng của cặp.

| Tiếng Việt | English | Nghĩa một dòng | Xuất hiện ở |
|---|---|---|---|
| đặc trưng | feature | con số mô tả mức khớp truy vấn-tài liệu | `features.py` |
| chuẩn hóa | standardization | trừ trung bình, chia độ lệch chuẩn (chỉ tính trên train) | `scaler_stats.csv` |
| hồi quy tuyến tính | linear regression | dự đoán nhãn như một số thực | `models.py` |
| hồi quy logistic | logistic regression | dự đoán xác suất liên quan | `models.py` |
| pointwise | pointwise | học từng tài liệu độc lập | `models.py` |
| pairwise | pairwise | học theo cặp tài liệu | `pairs.py` |
| Ranking SVM | Ranking SVM | SVM tuyến tính trên hiệu đặc trưng của các cặp | `models.py` |
| cặp ưu tiên | preference pair | cặp (tài liệu tốt hơn, tài liệu kém hơn) cùng truy vấn, khác nhãn | `pairs.py` |
| vector hiệu | difference vector | x_i trừ x_j | `pairs.py` |
| hệ số chặn | intercept (bias) | hằng số cộng vào điểm; triệt tiêu khi lấy hiệu | `ablation.csv` |
| lề | margin | chênh điểm giữa tài liệu tốt và kém của một cặp | Chương 8 (slide 115) |
| mất mát hinge | hinge loss | max(0, 1 − lề) | `models.py` |
| hinge bình phương | squared hinge | bình phương của hinge; mặc định của `LinearSVC` | `fit_ranksvm()` |
| hinge chuẩn | standard (L1) hinge | hinge không bình phương; slide 92 | `fit_ranksvm_hinge()` |
| tham số C | regularization parameter C | mức phạt lỗi; lớn = bám dữ liệu train chặt hơn | `tuning_dev.csv` |
| siêu tham số | hyperparameter | tham số chọn bằng dev, không học từ train | `config.py` |
| lấy mẫu cặp | pair sampling | chỉ dùng một phần cặp (tối đa một số cặp/truy vấn) | `pair_stats.json` |
| trọng số chênh lệch nhãn | grade-gap weighting | cặp có chênh nhãn lớn được trọng số lớn hơn | `metrics_test.csv` (`ranksvm_gapweighted`) |
| trọng số dNDCG | dNDCG weighting | trọng số theo vị trí (biến thể phụ của dự án) | `ranksvm_dndcgweighted_extra` |
| đa cộng tuyến | collinearity | các đặc trưng cho cùng một thông tin | `weights_table.csv` |
| VIF | variance inflation factor | chỉ số đo mức đa cộng tuyến | `ext_vif_by_set.csv` |
| hội tụ | convergence | bộ giải tối ưu đạt dung sai trước khi hết số vòng lặp | `ext_hinge_tuning_dev.csv` |

## Dữ liệu huấn luyện và chống lỗi

> **Bạn cần nhớ:** Chia theo truy vấn và fit chuẩn hóa trên train là hai chốt chặn rò rỉ quan trọng nhất.

| Tiếng Việt | English | Nghĩa một dòng | Xuất hiện ở |
|---|---|---|---|
| tập train / dev / test | train / validation (dev) / test | học / chọn tham số / chấm cuối | `data.py` |
| rò rỉ dữ liệu | data leakage | thông tin tập test lọt vào quá trình huấn luyện | `tests/test_pairs_and_leakage.py` |
| chia theo truy vấn | query-level split | mọi tài liệu của một truy vấn nằm cùng một tập | `assert_query_disjoint()` |
| kiểm định chéo | cross-validation | chia dữ liệu thành các phần, lần lượt giữ một phần làm test | `run_cv.py` |
| tập validation trong | inner validation split | tập con của phần train dùng để chọn C trong kiểm định chéo | `inner_split()` |
| hạt giống | random seed | số cố định để mọi lần chạy cho cùng kết quả | `config.py` |
| chạy thử nhanh | smoke test | chạy trên tập con nhỏ để kiểm tra mọi bước chạy được | `run_all.sh --smoke` |
| tái lập | reproducibility | chạy lại cho đúng cùng kết quả | `run_all.sh` |
| đối chứng | ablation | thí nghiệm đổi một yếu tố để xem ảnh hưởng | `ablation.csv` |

## Độ đo

> **Bạn cần nhớ:** MAP và NDCG chấm vị trí; độ chính xác theo cặp chấm mọi cặp như nhau.

| Tiếng Việt | English | Nghĩa một dòng | Xuất hiện ở |
|---|---|---|---|
| Precision@k | Precision@k | tỷ lệ tài liệu liên quan trong k tài liệu đầu | Chương 8 |
| AP / MAP | average precision / mean AP | trung bình Precision tại các vị trí liên quan / trung bình theo truy vấn | `metrics.py` |
| gain | gain | lợi ích của một tài liệu: 2^nhãn − 1 | `metrics.py` |
| DCG / IDCG | discounted cumulative gain / ideal DCG | tổng gain bị giảm giá theo vị trí / DCG của thứ tự lý tưởng | `metrics.py` |
| NDCG@k | NDCG@k | DCG chia IDCG, trong khoảng 0 đến 1, nhìn k vị trí đầu | `metrics.py` |
| độ chính xác theo cặp | pairwise accuracy | tỷ lệ cặp được xếp đúng thứ tự | `pairwise_counts()` |
| micro / macro | micro / macro averaging | gộp mọi cặp rồi tính / tính từng truy vấn rồi lấy trung bình | `metrics_test.csv` |

## Thống kê

> **Bạn cần nhớ:** Kiểm định cho biết chênh lệch có đáng tin không; ý nghĩa thực tế là chuyện khác.

| Tiếng Việt | English | Nghĩa một dòng | Xuất hiện ở |
|---|---|---|---|
| kiểm định t ghép cặp | paired t-test | kiểm tra trung bình các hiệu có khác 0 | `stats.py` |
| kiểm định Wilcoxon | Wilcoxon signed-rank test | kiểm tra bằng thứ hạng các hiệu (bỏ các hiệu bằng 0) | `stats.py` |
| bootstrap | bootstrap | lấy mẫu lại có hoàn lại để ước lượng độ bất định | `stats.py` |
| khoảng tin cậy | confidence interval (CI) | khoảng có khả năng chứa giá trị thật | `significance.csv` |
| giá trị p | p-value | xác suất thấy chênh lệch này nếu thực ra không có khác biệt | `significance.csv` |
| hiệu chỉnh Holm | Holm correction | điều chỉnh p khi làm nhiều so sánh | cột `p_ttest_holm` |
| cỡ hiệu ứng | effect size (Cohen's d_z) | độ lớn khác biệt so với độ dao động giữa các truy vấn | cột `cohen_dz` |
| ý nghĩa thống kê | statistical significance | khác biệt khó là do may rủi | `report.md` mục 5.1 |
| ý nghĩa thực tế | practical significance | khác biệt đủ lớn để người dùng nhận ra | `report.md` mục 5.1 |
| quy tắc quyết định | decision rule | cả hai p < 0.05 và khoảng tin cậy không chứa 0 | `report.md` mục 9 |
| đường rẽ | forking paths | chọn hướng phân tích sau khi đã nhìn kết quả | `report.md` mục 10 |
| khám phá nối tiếp | exploratory follow-up | kiểm tra tiếp trên cùng dữ liệu, chưa phải xác nhận độc lập | `report.md` mục 10 |
| xác nhận độc lập | independent confirmation | kiểm tra lại trên dữ liệu mới | (chưa làm) |

---

## Tự kiểm tra

**Câu 1.** "Tầng 1" và "tầng 2" khác nhau thế nào?
*Đáp án:* Tầng 1 (BM25) chọn nhanh 100 ứng viên từ kho; tầng 2 (mô hình học máy) xếp hạng lại các ứng viên đó.

**Câu 2.** Hinge chuẩn và hinge bình phương khác nhau chỗ nào?
*Đáp án:* Hinge chuẩn là max(0, 1 − lề); hinge bình phương là bình phương của giá trị đó.

**Câu 3.** "Có ý nghĩa thống kê" khác "có ý nghĩa thực tế" thế nào?
*Đáp án:* Thống kê: khó là may rủi. Thực tế: đủ lớn để người dùng nhận ra.

---

## Nếu chỉ có 10 phút

1. Học các cặp: tầng 1/tầng 2, pointwise/pairwise, hinge chuẩn/hinge bình phương, thống kê/thực tế.
2. Nhớ ba độ đo: MAP, NDCG@k, độ chính xác theo cặp.
3. Nhớ recall của tầng 1 là giới hạn của tầng 2.
4. Nhớ "rò rỉ" nghĩa là test lọt vào huấn luyện.
5. Nhớ "đường rẽ" nghĩa là chọn hướng sau khi nhìn kết quả.
