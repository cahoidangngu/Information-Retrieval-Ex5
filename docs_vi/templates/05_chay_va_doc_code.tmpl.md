# Chạy dự án và đọc code

> **Bạn cần nhớ:** Luôn chạy **thử nhanh (`--smoke`) trước**, rồi mới chạy đầy đủ. Chạy đầy đủ mất khoảng [[txt|total_runtime_seconds.txt|x/60|.0f]] phút trên máy đã dùng (đọc từ `total_runtime_seconds.txt`: [[txt|total_runtime_seconds.txt|x|.0f]] giây, gồm cả kiểm thử).

---

## 5.1 Chuẩn bị môi trường

> **Bạn cần nhớ:** Cài môi trường từ `environment.yml` rồi mới chạy.

Cần: Python và các thư viện ghi trong `environment.yml` (môi trường conda tên `ltr`).

```bash
conda env create -f environment.yml
conda activate ltr
```

Trên Windows hãy chạy script bằng **Git Bash**. Nếu không muốn kích hoạt môi trường, chỉ đường dẫn Python bằng biến `PYTHON`:

```bash
PYTHON=/duong/dan/den/env/python bash run_all.sh
```

---

## 5.2 Chạy thử nhanh trước (smoke test)

> **Bạn cần nhớ:** Chạy thử nhanh trước để biết mọi bước chạy được.

```bash
bash run_all.sh --smoke
```

- Chạy trên một **tập con nhỏ** (mỗi nhóm train/dev/test lấy ngẫu nhiên vài chục truy vấn), ghi kết quả vào `results_smoke/` (không đụng vào `results/`).
- Mục đích: kiểm tra mọi bước chạy được, không cần chờ lâu. Thời gian của lần chạy thử: không có trong kết quả (không có file nào ghi lại).
- Con số trong `results_smoke/` **không** dùng làm kết quả; chúng dao động nhiều vì quá ít truy vấn.

---

## 5.3 Chạy đầy đủ

> **Bạn cần nhớ:** Một lệnh `bash run_all.sh` dựng lại mọi thứ từ dữ liệu gốc.

```bash
bash run_all.sh
```

Các bước (thứ tự trong `run_all.sh`):

1. `src/run_features.py`: tải dữ liệu, tầng 1, đặc trưng, recall tầng 1.
2. `pytest tests`: kiểm thử tự động.
3. `src/run_models.py`: huấn luyện, chọn C trên dev, chấm test, kiểm định.
4. `src/analysis.py`, `src/ablation.py`, `src/crosscheck.py`: phân tích, đối chứng, kiểm tra độ đo.
5. `src/run_hinge.py`, `src/run_featuresets.py`, `src/run_cv.py`, `src/run_stemmed_stage1.py`: các thí nghiệm mở rộng.
6. `src/make_report.py`: dựng `report.md`.

Chạy từng bước riêng (ví dụ khi sửa một bước):

```bash
python -I src/run_features.py
python -I src/run_models.py
python -I src/analysis.py
```

Cờ `-I` (isolated) để Python không tự nạp mã từ thư mục dữ liệu; mỗi script tự thêm `src/` vào đường dẫn.

**Thời gian từng bước:** không có file kết quả riêng cho từng bước; `run_log.txt` có thời điểm bắt đầu/kết thúc của các dòng log, đọc từ đó nếu cần.

---

## 5.4 Kết quả mong đợi

> **Bạn cần nhớ:** Chạy đúng thì bạn thấy lại chính các con số chính trong bảng dưới.

Sau khi chạy:

- `results/` đầy đủ file (xem `02_ban_do_file.md`), `report.md` được sinh lại, khối "Kết quả chính" trong `README.md` được cập nhật.
- Kiểm tra nhanh các con số chính (nếu chạy đúng bạn sẽ thấy chính các số này):

| Chỉ số (tập test) | Giá trị | File |
|---|---|---|
| NDCG@10 của BM25 | [[csv|metrics_test.csv|model=bm25|NDCG@10|.4f]] | `metrics_test.csv` |
| MAP của BM25 | [[csv|metrics_test.csv|model=bm25|MAP|.4f]] | `metrics_test.csv` |
| NDCG@10 mô hình chọn trên dev | [[csv|metrics_test.csv|model=ranksvm_sampled|NDCG@10|.4f]] | `metrics_test.csv` |
| recall@100 gộp của tầng 1 | [[csv|candidate_recall_summary.csv|split=test|recall@100_micro|.4f]] | `candidate_recall_summary.csv` |
| số cặp hợp lệ khi dùng tất cả | [[json|pair_stats.json|n_valid_pairs_all|,d]] | `pair_stats.json` |

---

## 5.5 Kiểm tra kết quả tái lập

> **Bạn cần nhớ:** Các cột chỉ số phải trùng hoàn toàn; chỉ cột thời gian và bộ nhớ được khác.

1. Chạy lại và so sánh các cột **chỉ số** (MAP, NDCG, độ chính xác theo cặp) của `metrics_test.csv` trước/sau: chúng phải **trùng đến mọi chữ số** vì mọi hạt giống đã cố định (`SEED` = [[cfg|SEED|d]] trong `config.py`).
2. Các cột **thời gian và bộ nhớ** (`train_s`, `predict_s`, `peak_mb`) sẽ khác nhau chút ít giữa các lần chạy; đó là bình thường.
3. Kiểm thử tự động:

```bash
python -m pytest -q tests
```

4. Kiểm tra số trong bộ tài liệu tiếng Việt:

```bash
python -I docs_vi/tools/build_docs.py
python -I docs_vi/tools/check_numbers.py
```

Bước đầu dựng lại các file `docs_vi/*.md` từ khuôn mẫu, đọc số từ `results/`; bước sau kiểm tra độc lập từng số và báo lỗi nếu lệch.

---

## 5.6 Lộ trình đọc code (đọc theo thứ tự này)

> **Bạn cần nhớ:** Đọc theo thứ tự từ cấu hình, dữ liệu, BM25, đặc trưng, cặp, mô hình, độ đo đến các chương trình chính.

Với mỗi file, vừa đọc vừa tìm đúng những thứ ghi dưới đây.

### 1) `src/config.py` (5 phút)
- Tìm các hằng: hạt giống, số ứng viên `TOP_K`, tham số BM25, lưới `C_GRID`, `PAIR_MEM_BUDGET_GB`, `PAIR_CAP_PER_QUERY`.
- Nhận ra: chỉ một nơi định nghĩa tham số; các file khác chỉ import.

### 2) `src/data.py` (10 phút)
- `load()`: nạp tài liệu, truy vấn, nhãn của ba tập.
- `assert_query_disjoint()`: bảo vệ chống rò rỉ giữa các tập (không truy vấn nào ở hai tập).
- `dataset_stats()`: các con số trong `dataset_stats.json`.

### 3) `src/bm25.py` (15 phút)
- `tokenize()`: hạ chữ thường, bỏ từ dừng, không stemming.
- `bm25_weight_matrix()`: công thức BM25 trên ma trận thưa (tìm `k1`, `b`, `idf`).
- `top_k()`: lấy ứng viên; chỉ tài liệu có điểm > 0; hòa điểm thì theo chỉ số tài liệu.

### 4) `src/features.py` (15 phút)
- `TextIndex`: dựng chỉ mục cho tiêu đề, nội dung, gộp; có tham số `analyzer` để đổi bộ phân tích từ (dùng cho stemming).
- `stage1_and_features()`: chọn ứng viên rồi tính từng đặc trưng. Hàm **không có tham số nhãn**: đặc trưng không thể dùng nhãn.
- Danh sách `FEATURE_NAMES`.

### 5) `src/pairs.py` (15 phút), file quan trọng nhất của phần Ranking SVM
- `build_pairs()`: tạo cặp trong từng nhóm truy vấn, chỉ khi nhãn khác nhau; có `assert` cùng truy vấn và nhãn khác nhau; lật đúng một nửa số cặp.
- `count_pairs()` và `estimate_memory_gb()`: đếm và ước lượng bộ nhớ **trước** khi dựng.
- Thử: so ví dụ 4 tài liệu (nhãn 2, 2, 1, 0) trong `01_kien_thuc_chuong8.md` với tư duy của hàm này.

### 6) `src/models.py` (10 phút)
- `fit_linreg()`, `fit_logreg()`: hai mô hình pointwise.
- `fit_ranksvm()`: `LinearSVC(fit_intercept=False, ...)` trên cặp; tìm tham số `loss="squared_hinge"`.
- `fit_ranksvm_hinge()`: bản hinge chuẩn (`loss="hinge"`, `dual=True`).

### 7) `src/metrics.py` (20 phút), đọc kỹ
- `ndcg_at_k()`: IDCG lấy từ **toàn bộ** nhãn (tham số `all_rels`).
- `average_precision()`: mẫu số là tổng số tài liệu liên quan thật.
- `pairwise_counts()`: ma trận so sánh từng cặp; hòa điểm tính nửa đúng.
- `evaluate_scores()`: cuối hàm có đoạn thêm các truy vấn **không có ứng viên** với điểm 0.

### 8) `src/stats.py` (10 phút)
- `paired_stats()`: kiểm định t, Wilcoxon, bootstrap, Cohen's d_z.
- `significance_table()`: so từng mô hình với BM25 và hiệu chỉnh Holm.

### 9) `src/run_models.py` (20 phút)
- Tìm các `assert`: tập không giao nhau, chuẩn hóa fit trên train, cặp cân bằng.
- Vòng chọn C trên dev, rồi mới chấm test.
- Cách chọn "mô hình tốt nhất" bằng dev (`best_model.json`).

### 10) `src/analysis.py` (15 phút)
- Bảng trọng số, `EXPECTED` (dấu kỳ vọng đặt trước), VIF.
- Danh sách truy vấn tốt lên/xấu đi và phần sinh `query_evidence`.
- Phần độ chính xác theo cặp so với NDCG (`pairacc_vs_ndcg.csv`).

### 11) Phần mở rộng (đọc khi cần)
- `src/porter.py`: hàm `stem()`, các bước 1a đến 5b của thuật toán Porter.
- `src/features_v2.py`: `proximity()` (khoảng cách từ), `compute_v2_new()`.
- `src/ltr_runner.py`: `tune()` (chọn C trên tập validation), `vif_prune()` (bỏ đặc trưng VIF cao).
- `src/run_cv.py`: `assign_folds()` (chia theo truy vấn), `inner_split()` (tập validation trong).
- `src/run_stemmed_stage1.py` và `src/stage1_eval.py`: tầng 1 có stemming, tóm tắt recall và giới hạn trên.

### 12) `tests/` (15 phút)
- `test_metrics.py`: các ví dụ tính tay; so với thư viện.
- `test_pairs_and_leakage.py`: các bảo vệ chống lỗi trong slide.
- Mỗi test là một câu "tôi chắc chắn rằng ..." viết thành mã.

---

## Tự kiểm tra

**Câu 1.** Vì sao chạy `--smoke` trước?
*Đáp án:* Để kiểm tra mọi bước chạy được trong thời gian ngắn trên tập con nhỏ và ghi vào `results_smoke/`, không phá `results/`.

**Câu 2.** Sau khi chạy lại, cột nào của `metrics_test.csv` được phép khác lần trước?
*Đáp án:* Các cột thời gian/bộ nhớ (`train_s`, `predict_s`, `peak_mb`); các cột chỉ số phải trùng hoàn toàn.

**Câu 3.** Hàm nào đảm bảo đặc trưng không dùng nhãn?
*Đáp án:* `stage1_and_features()` không có tham số nhãn; nhãn chỉ được ghép sau khi đặc trưng đã tính xong (trong `run_features.py`).

---

## Nếu chỉ có 10 phút

1. Cài môi trường bằng `environment.yml`.
2. Chạy `bash run_all.sh --smoke`, rồi `bash run_all.sh`.
3. Kiểm tra: chỉ số trùng, thời gian có thể khác; chạy `python -m pytest -q tests`.
4. Đọc code theo thứ tự: `config`, `data`, `bm25`, `features`, `pairs`, `models`, `metrics`, `stats`, `run_models`, `analysis`.
5. Ba file quan trọng nhất: `pairs.py`, `metrics.py`, `run_models.py`.
