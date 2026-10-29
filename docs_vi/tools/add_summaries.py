"""One-off helper: inserts a one-line 'Bạn cần nhớ:' summary after section headings of the templates (idempotent)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
T = Path(__file__).resolve().parents[1] / "templates"
PREFIX = "> **Bạn cần nhớ:** "

SUMMARIES = {
    "02_ban_do_file.tmpl.md": [
        ("## 2.1 ", "`src/` là mã, `results/` là kết quả, `report.md` là báo cáo, `docs_vi/` là tài liệu học."),
        ("## 2.2 ", "Dữ liệu đi qua một chuỗi bước từ `run_features.py` đến `make_report.py`; mỗi bước ghi ra một nhóm file trong `results/`."),
        ("## 2.3 ", "Mỗi file trong `src/` có đúng một việc; ba file cốt lõi là `pairs.py`, `models.py`, `metrics.py`."),
        ("## 2.4 ", "Tên file trong `results/` cho biết nó trả lời câu hỏi gì; tiền tố `ext_` là phần mở rộng thêm sau."),
        ("## 2.5 ", "Mỗi test là một điều dự án cam kết là đúng và được máy kiểm tra."),
    ],
    "03_doc_ket_qua.tmpl.md": [
        ("## 3.1 ", "Mỗi yêu cầu của đề có một việc đã làm và một file làm bằng chứng."),
        ("## 3.2 ", "Kết quả trung thực là không có cải thiện nào của mô hình học máy được kiểm định ủng hộ."),
        ("## 3.3 ", "Mỗi bảng có một con số chính để nhìn trước; các cột còn lại giải thích con số đó."),
        ("### Bảng A", "Recall thấp của tầng 1 là trần của mọi thứ ở tầng 2."),
        ("### Bảng B", "Nhìn hàng BM25 làm mốc, rồi nhìn hàng được chọn trên dev."),
        ("### Bảng C", "Chi phí chỉ để tham khảo; điểm xếp hạng mới là kết quả chính."),
        ("### Bảng D", "Khoảng tin cậy chứa 0 nghĩa là chưa chứng minh được khác biệt."),
        ("### Bảng E", "Trọng số chỉ so sánh được trong cùng một mô hình và dễ bị đa cộng tuyến làm sai lệch."),
        ("### Bảng F", "Hinge chuẩn cho kết quả gần như bằng hinge bình phương nhưng khó hội tụ."),
        ("### Bảng G", "Mẫu lớn hơn cho khoảng tin cậy chặt hơn, nhưng mức tăng vẫn rất nhỏ."),
        ("### Bảng H", "Đặc trưng v2 tốt hơn v1 nhưng không tốt hơn việc chỉ dùng BM25 có stemming."),
        ("### Bảng I", "Rút gọn VIF không làm dấu trọng số diễn giải được."),
        ("### Bảng J", "Stemming ở tầng 1 là thứ thật sự giúp; phần học máy thêm vào gần như không đáng kể."),
        ("## 3.4 ", "Tầng 2 chỉ xếp lại các ứng viên nên recall của tầng 1 chặn trần kết quả."),
        ("## 3.5 ", "Phần này liệt kê những điều **không** được phép khẳng định từ kết quả."),
    ],
    "04_theo_doi_mot_truy_van.tmpl.md": [
        ("## Phần A.", "Truy vấn tốt lên nhờ vài tài liệu liên quan được kéo vào top 10."),
        ("## Phần B.", "Truy vấn xấu đi vì một hai tài liệu liên quan bị đẩy ra khỏi top 10."),
        ("## Phần C.", "Hai ví dụ chỉ giúp hiểu cơ chế, không đại diện cho toàn bộ."),
        ("## Phụ lục", "Hai bảng top 10 được trích tự động và được kiểm tra lại bằng một cách dựng độc lập."),
    ],
    "05_chay_va_doc_code.tmpl.md": [
        ("## 5.1 ", "Cài môi trường từ `environment.yml` rồi mới chạy."),
        ("## 5.2 ", "Chạy thử nhanh trước để biết mọi bước chạy được."),
        ("## 5.3 ", "Một lệnh `bash run_all.sh` dựng lại mọi thứ từ dữ liệu gốc."),
        ("## 5.4 ", "Chạy đúng thì bạn thấy lại chính các con số chính trong bảng dưới."),
        ("## 5.5 ", "Các cột chỉ số phải trùng hoàn toàn; chỉ cột thời gian và bộ nhớ được khác."),
        ("## 5.6 ", "Đọc theo thứ tự từ cấu hình, dữ liệu, BM25, đặc trưng, cặp, mô hình, độ đo đến các chương trình chính."),
    ],
    "06_hoi_dap_bao_ve.tmpl.md": [
        ("## Phần 1.", "Câu trả lời tốt luôn có kết luận, con số kèm tên file, và giới hạn của kết luận."),
        ("## Phần 2.", "Với câu hỏi bẫy, câu trả lời đúng là \"dữ liệu không ủng hộ điều đó\" kèm lý do."),
    ],
    "07_bang_thuat_ngu.tmpl.md": [
        ("## Dữ liệu và bài toán", "Hiểu truy vấn, tài liệu, nhãn, ứng viên và hai tầng là hiểu đề bài."),
        ("## Tầng 1 và xử lý văn bản", "Tầng 1 chỉ chọn ứng viên; recall của nó là trần của tầng 2."),
        ("## Đặc trưng và mô hình", "Mô hình chỉ thấy các con số đặc trưng; Ranking SVM học trên hiệu đặc trưng của cặp."),
        ("## Dữ liệu huấn luyện và chống lỗi", "Chia theo truy vấn và fit chuẩn hóa trên train là hai chốt chặn rò rỉ quan trọng nhất."),
        ("## Độ đo", "MAP và NDCG chấm vị trí; độ chính xác theo cặp chấm mọi cặp như nhau."),
        ("## Thống kê", "Kiểm định cho biết chênh lệch có đáng tin không; ý nghĩa thực tế là chuyện khác."),
    ],
}

for name, items in SUMMARIES.items():
    p = T / name
    lines = p.read_text(encoding="utf8").split("\n")
    out, i = [], 0
    while i < len(lines):
        out.append(lines[i])
        for pref, text in items:
            if lines[i].startswith(pref):
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                if "Bạn cần nhớ" not in nxt and "Bạn cần nhớ" not in (lines[i + 2] if i + 2 < len(lines) else ""):
                    out.extend(["", PREFIX + text])
                break
        i += 1
    p.write_text("\n".join(out), encoding="utf8")
    print(name, "ok")
