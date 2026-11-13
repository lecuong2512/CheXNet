# Đối chiếu tên bệnh và kết quả hiển thị — 09/11/2026

## Hai vấn đề khác nhau

Tên tiếng Việt trên web đã được đồng bộ theo 15 dòng của bảng danh mục trong README. Giữ nguyên khóa lớp và thứ tự kênh khi train/suy luận; `Pleural Thickening` chỉ là cách viết hiển thị của `Pleural_Thickening`. Kiểm thử đối chiếu README, danh mục FE/adapter, dataset loader, trainer, fine-tune trainer, CLI và backend.

Điểm dự đoán yếu của checkpoint không được sửa bằng cách đổi tên hoặc hạ ngưỡng dựa trên tập test. Checkpoint hiện có metadata epoch 13, fine-tune SOTA_v2, best_auroc 0,86565085. Chưa có tệp `Results/optimal_thresholds.npy`; pipeline dùng ngưỡng 0,5. Các số AUROC trong README thuộc các lần thực nghiệm, không chứng minh accuracy của file trọng số đang phục vụ ảnh người dùng.

## Lỗi nối dữ liệu đã sửa

Pipeline gốc chỉ tạo tối đa 6 bản đồ chú ý. Trước đây FE lấy chính danh sách này làm danh sách bệnh, làm mất các nhãn vượt ngưỡng nhưng đứng hạng 7 trở đi. Adapter mới thêm `inferenceMeta.classification` với đủ 15 quyết định, điểm và ngưỡng theo tên lớp; không thay `hybridAnalysis`, trọng số hoặc xác suất gốc.

Dashboard, hồ sơ bệnh nhân và phiếu/PDF dùng danh sách phân loại đầy đủ. Bản đồ chú ý và mô tả gốc vẫn được giữ, ghi rõ khi không bao phủ mọi nhãn. Khi No Finding đồng thời vượt ngưỡng với bệnh, web/PDF báo bất đồng, không tự xóa một đầu ra để làm đẹp kết quả. Phiếu cũ không lưu ngưỡng được đánh dấu là dữ liệu cũ; không suy diễn ngưỡng lịch sử hoặc tự viết lại kết luận đã lưu.

Đã bỏ ngưỡng 35% và mô tả/khuyến nghị tự tạo ở FE cho dữ liệu thiếu kết quả Hybrid. Không còn gọi danh sách dự đoán trống là “Ổn định / Bình thường”.

## Đối chiếu lại 200 kết quả đã đóng băng

Đây là chạy lại cách đọc quyết định trên **các xác suất đã lưu**, không phải một bộ test độc lập mới hoặc cải thiện trọng số.

| Cách lấy kết luận viêm phổi | TP | TN | FP | FN | Accuracy | Độ nhạy |
|---|---:|---:|---:|---:|---:|---:|
| Danh sách tối đa 6 bản đồ chú ý trước đây | 2 | 100 | 0 | 98 | 51% | 2% |
| Đủ quyết định từ classifier ở ngưỡng 0,5 | 4 | 100 | 0 | 96 | 52% | 4% |

Hai ảnh `person1664_virus_2877.jpeg` và `person19_virus_50.jpeg` có Pneumonia lần lượt 57,87% và 51,19% nhưng bị loại khỏi danh sách cũ vì đứng hạng 7. Bản mới giữ lại chúng. Trong 200 ca, 18 ca có ít nhất một nhãn vượt ngưỡng thiếu bản đồ chú ý; 50 ca có No Finding đồng thời vượt ngưỡng với nhãn bệnh.

Việc sửa web khôi phục kết quả đã có; **96/100 ảnh viêm phổi vẫn dưới ngưỡng Pneumonia**. Chưa thể kết luận nguyên nhân là thiếu epoch. Muốn cải thiện model cần ảnh validation có nhãn đủ 14 bệnh, xác minh nguồn gốc checkpoint, đánh giá theo từng bệnh, hiệu chỉnh trên validation và chấm lại trên test độc lập. Workspace hiện có CSV train/val/test nhưng không có thư mục ảnh validation tương ứng; không dùng 200 ảnh test này để chọn ngưỡng rồi báo lại accuracy.

## Cách kiểm chứng

```powershell
.\.venv\Scripts\python.exe -m pytest Backend/tests -q
npm --prefix Frontend run build
npm --prefix Backend run build
.\.venv\Scripts\python.exe Backend/tools/audit_classification.py
# Sau khi khởi động lại AI service để nhận adapter mới:
.\.venv\Scripts\python.exe Backend/tools/audit_classification.py --api http://127.0.0.1:8000
```

Audit dùng corpus `.runtime/external-eval-2026-11-08`, lưu báo cáo tại `.runtime/classification-audit-2026-11-09.json`. Kiểm tra live gồm hai ảnh từng bị rơi nhãn, một ảnh NORMAL có đầu ra bất đồng, một ảnh NORMAL không có bệnh vượt ngưỡng. Điểm classifier mới phải khớp các điểm đã lưu với sai số tối đa dưới 0,00001.

Các file kỹ thuật gốc theo manifest 25 file được đối chiếu SHA-256; không sửa mã huấn luyện, pipeline gốc hoặc trọng số. Báo cáo này không khẳng định đã tăng accuracy của model.

## Kết quả xác nhận sau khi sửa

- 26 kiểm thử Python đạt; build FE và BE đạt.
- 4 ca API live HTTP 200, đủ 15 quyết định. Sai khác xác suất lớn nhất so với điểm đã lưu là khoảng 0,0000000596, trong sai số số thực; không có thay đổi trọng số.
- Hai ca từng mất nhãn được tải qua gateway Node, lưu và đọc lại từ MongoDB, đều có Pneumonia trong danh sách vượt ngưỡng và vẫn không có bản đồ chú ý của lớp này. PDF trả HTTP 200.
- Hồ sơ mẫu để đối chiếu trên web: `PT-715BEC` — “KIỂM THỬ NHÃN - HỒ SƠ MẪU”. Tuổi và giới tính chỉ là dữ liệu thử phần mềm; hai ảnh nguồn không thuộc cùng bệnh nhân. Chưa xác minh kết luận bác sĩ và không lưu vào CBIR.
- Bằng chứng gateway/PDF tại `.runtime/classification-verification/`; bằng chứng API tại `.runtime/classification-audit-2026-11-09.json`.

## Khung định vị YOLO trên ảnh

Backend gốc đã gọi YOLO và Cascade khi nạp được `Trainedmodel/yolov11m.pt`. Ở lượt kiểm tra ban đầu, workspace chỉ có `hybrid_model.pth`, trạng thái API `yolo_loaded=false`; không có dự đoán YOLO thật để vẽ. Sau khi người dùng bổ sung trọng số, đã nối và kiểm tra khung thật theo [báo cáo nối YOLO](noi-yolo-2026-11-09.md). Các kết quả đánh giá trước đó vẫn thuộc lượt chạy chưa có YOLO.

Viewer được nối với `hybridAnalysis.yolo_detections`, thêm chế độ “Khung YOLO”. Khi có kết quả hợp lệ, khung và chú giải xuất hiện trên **ảnh gốc**, cùng phóng to với ảnh. Tọa độ dùng kích thước tự nhiên của ảnh và SVG `preserveAspectRatio`, không áp tọa độ ảnh gốc lên heatmap đã cắt. Khung từ bản đồ chú ý được gọi rõ tên, không gán thành kết quả YOLO. Khi thiếu mô hình hoặc không có kết quả sau lọc, giao diện giải thích và không tự tạo khung.

Kiểm thử hình học bằng 4 test Node: ảnh không vuông, cắt phần ngoài ảnh, tọa độ không hữu hạn/đảo chiều/rỗng, điểm không hợp lệ. Kiểm tra trình duyệt bằng hình và khung **giả lập**, đánh dấu rõ không phải kết quả model: sai lệch vị trí dưới 0,001 pixel khi zoom 100%/125% và ở chiều rộng 390px. Chuyển ảnh gốc/bản đồ chú ý tắt overlay; chế độ khung bật lại đúng tọa độ. Trang kiểm thử tạm đã xóa. Chưa kiểm chứng định vị tổn thương thật do chưa có trọng số.

```powershell
node --test Frontend/tests/yoloBoxes.test.cjs
```
