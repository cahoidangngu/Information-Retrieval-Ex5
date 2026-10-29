# Đọc trước (1 trang)

> **Bạn cần nhớ:** Dự án này là một thí nghiệm "xếp hạng hai tầng". Kết luận trung thực: mô hình học máy **không** chứng minh được là tốt hơn BM25 trên bộ test chính thức; phần lớn lợi ích nhỏ thấy được đến từ việc **stemming** (đưa từ về gốc).

## Dự án làm gì? (5 câu)

1. Có một kho văn bản (3,633 bài báo dinh dưỡng, xem `dataset_stats.json`) và nhiều câu truy vấn (tiêu đề video dinh dưỡng).
2. **Tầng 1:** thuật toán BM25 chọn nhanh 100 bài "có vẻ liên quan" cho mỗi truy vấn (giống thủ thư lấy 100 cuốn sách từ kho).
3. **Tầng 2:** một mô hình học máy (hồi quy tuyến tính, hồi quy logistic, Ranking SVM) **xếp hạng lại** 100 bài đó bằng 14 "đặc trưng" tính từ văn bản.
4. Đánh giá bằng MAP, NDCG@10, độ chính xác theo cặp; kiểm định thống kê xem khác biệt có đáng tin không.
5. Sau đó thêm các thí nghiệm mở rộng: hinge chuẩn, kiểm định chéo 5 phần, bộ đặc trưng v2, stemming ở tầng 1.

## Kết quả trong một dòng

Trên 323 truy vấn test: BM25 đạt NDCG@10 = 0.3134 (`metrics_test.csv`); mô hình được **chọn trên tập dev** (Ranking SVM lấy mẫu cặp) đạt 0.3117. Kiểm định thống kê **không** thấy khác biệt đáng tin (`significance.csv`).

## Thứ tự đọc

| Thứ tự | File | Nội dung | Thời gian |
|---|---|---|---|
| 1 | `00_doc_truoc.md` (file này) | Bức tranh tổng thể | 5 phút |
| 2 | `01_kien_thuc_chuong8.md` | Kiến thức Chương 8 từ số 0 | 45 phút |
| 3 | `07_bang_thuat_ngu.md` | Tra cứu thuật ngữ khi cần | tra khi cần |
| 4 | `03_doc_ket_qua.md` | Cách đọc các bảng kết quả | 30 phút |
| 5 | `04_theo_doi_mot_truy_van.md` | Theo dõi 2 truy vấn thật từ đầu đến cuối | 20 phút |
| 6 | `02_ban_do_file.md` | Mỗi file/thư mục để làm gì | 20 phút |
| 7 | `05_chay_va_doc_code.md` | Cách chạy và đọc code | 30 phút |
| 8 | `06_hoi_dap_bao_ve.md` | 25 câu hỏi giáo viên hay hỏi | 30 phút |

Thời gian là ước lượng đọc của người mới, **không** đến từ file kết quả nào.

## Phần nào có thể bỏ qua (khi bạn ít thời gian)

- Chi tiết phần cài đặt Porter stemmer (`porter.py`) và đặc trưng v2 (`features_v2.py`): chỉ cần biết ý tưởng.
- Bảng tuning đầy đủ (`tuning_dev.csv`): chỉ cần biết "C nằm trong lưới [0.001, 0.01, 0.1, 1, 10, 100] và chọn trên dev".
- Thí nghiệm hinge chuẩn (mục 9.1 của `report.md`): chỉ cần nhớ "hinge chuẩn khó hội tụ hơn, kết quả gần như bằng squared hinge".

**Không** nên bỏ qua: phần **kiểm định** (mục 1h trong `01_kien_thuc_chuong8.md`) và phần **điều kết quả KHÔNG chứng minh** (`03_doc_ket_qua.md`). Giáo viên nghiêm khắc thường hỏi đúng chỗ này.

## Một lưu ý về cách các con số được tạo

Mọi con số trong bộ tài liệu này được **đọc tự động** từ file trong `results/` khi dựng tài liệu (`docs_vi/tools/build_docs.py`) và được kiểm tra lại độc lập (`docs_vi/tools/check_numbers.py`). Nếu một con số không có trong kết quả, tài liệu ghi rõ "không có trong kết quả".

## Nếu chỉ có 10 phút

1. Hai tầng: BM25 lấy 100 ứng viên, mô hình học máy xếp hạng lại.
2. Mô hình học máy dùng 14 đặc trưng; Ranking SVM học từ **cặp** tài liệu (cùng truy vấn, nhãn khác nhau).
3. Đo bằng MAP, NDCG@10 và độ chính xác theo cặp; chọn tham số C trên dev, chạm test **một lần**.
4. Kết quả: không có cải thiện được kiểm định ủng hộ so với BM25 trên bộ test chính thức.
5. Stemming (đưa từ về gốc) mới là thứ giúp; mô hình học máy thêm ra gần như không đáng kể.
