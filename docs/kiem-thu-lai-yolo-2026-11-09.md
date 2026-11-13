# Chạy lại 200 ảnh sau khi nối YOLO — 09/11/2026

## Thiết kế đối chiếu

Chạy thật lại đúng 200 ảnh đã dùng ngày 08/11: 100 NORMAL, 100 PNEUMONIA. Giữ nguyên byte ảnh, nhãn nguồn, checkpoint Hybrid và ngưỡng 0,5. Bổ sung checkpoint YOLO VinDr đã kiểm tra, không hiệu chỉnh ngưỡng hoặc trọng số. Đây là phép đối chiếu trên tập test đã xem trước đó, không phải đánh giá độc lập mới.

Protocol, manifest và hash checkpoint được ghi trước khi chạy, trong `.runtime/external-eval-yolo-2026-11-09/`. Ảnh được gọi tuần tự qua `/v3/predict`, cùng pipeline mà web sử dụng; không tạo 200 hồ sơ bệnh nhân.

Nguồn ảnh: Kermany, Zhang & Goldbaum (2018), [Mendeley Data V2](https://data.mendeley.com/datasets/rscbjbr9sj/2), ảnh theo CC BY 4.0; bản sao và commit cố định được ghi trong protocol gốc.

## Kết quả nhãn Pneumonia

| Chỉ số classifier ở ngưỡng 0,5 | Trước khi có YOLO | Chạy lại có YOLO |
|---|---:|---:|
| Accuracy | 52.00% | 52.00% |
| Độ nhạy | 4.00% | 4.00% |
| Độ đặc hiệu | 100.00% | 100.00% |
| AUROC | 0.8819 | 0.8819 |

Confusion matrix mới: TP 4, TN 100, FP 0, FN 96. Điểm classifier chênh lớn nhất 1.1920928955078125e-07; số quyết định Pneumonia đổi ở ngưỡng 0,5: 0.

Giao diện mới đọc đầy đủ các nhãn vượt ngưỡng: accuracy Pneumonia 52.00%, độ nhạy 4.00%. Danh sách top-six heatmap cũ là một đầu ra khác, không dùng làm danh sách kết luận đầy đủ nữa. Không quy cải thiện do sửa lỗi top-six cho YOLO.

## Hoạt động YOLO và API

- API v3 nhận thành công 200/200 ảnh; classifier chấm được 200/200. Fallback legacy: 0. Lỗi/từ chối: 0.
- 50/200 ảnh có ít nhất một khung sau Cascade; tổng 76 khung. 150 ảnh không có khung sau lọc.
- Thời gian HTTP: median 4.864 giây/ảnh, p95 17.715 giây/ảnh, tổng 1350.5 giây. Đây là lượt chạy tuần tự trên CPU.

Trong lượt chạy có Android Studio và giả lập Android cùng hoạt động; một lần quan sát hệ thống còn khoảng 1 GB RAM trống. Các số thời gian là quan sát trên máy đang đa nhiệm, không phải phép đo hiệu năng trong môi trường được cô lập.

Số ảnh có khung không phải tỷ lệ chẩn đoán đúng. Bộ Kermany này chỉ có nhãn NORMAL/PNEUMONIA, không có bounding box chuẩn hoặc nhãn đầy đủ cho các bệnh khác; không thể tính mAP/độ chính xác định vị YOLO từ phép thử này. YOLO VinDr hiện tại không có lớp riêng Pneumonia; nó định vị các nhóm VinDr, không thay thế đầu phân loại Pneumonia của Hybrid.

## Bằng chứng

- `results.jsonl`: 200 phản hồi rút gọn, đủ 15 xác suất, kết luận đầy đủ, khung YOLO và thời gian mỗi ảnh.
- `comparison.json`, `metrics.json`, `independent-check.json`: chỉ số được đối chiếu độc lập bằng sklearn.
- `health-before.json`, `health-after.json`: trạng thái dịch vụ trước/sau.
- `six-fixed-cases.png`: ba ảnh đầu tiên của mỗi lớp theo manifest, không chọn ảnh dựa trên kết quả đẹp.
- [Báo cáo gốc](danh-gia-200-anh-2026-11-08.md), [kiểm chứng nối YOLO](noi-yolo-2026-11-09.md).
