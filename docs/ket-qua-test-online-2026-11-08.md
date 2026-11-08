# Kiểm thử web và Hybrid bằng ảnh X-quang công khai

Ngày kiểm thử: 08/11/2026. Workspace: `D:\NCKH\CheXNet`.

## Kết quả chính

Web đang chạy ở http://127.0.0.1:5173/. Gateway ở cổng 3006, AI ở cổng 8000, MongoDB đang chạy. Hybrid và bộ kiểm tra ảnh CLIP đã nạp trên CPU. YOLO chưa nạp vì chưa có checkpoint. Các kết quả dưới đây là đầu ra thực tế, không sửa trọng số, ngưỡng hay nhãn để khớp mẫu.

Luồng đăng nhập → tạo hồ sơ thử → tải JPEG → chạy model → lưu hai lần chụp → xem ảnh gốc, attention và 15 xác suất → tải lại trang → đổi lần chụp → tải PDF đều đã thực hiện qua giao diện. API đọc lại cho 15 xác suất trùng kết quả gọi AI trực tiếp, sai khác lớn nhất bằng 0 trên cả hai ảnh.

**Chưa thể kết luận model phân loại đúng với dữ liệu bên ngoài.** Hai mẫu có kết quả không khớp hoàn toàn mô tả của nguồn. Đây là kiểm tra chức năng và đối chiếu mẫu, không phải phép đo accuracy, AUC hoặc đánh giá lâm sàng.

## Ảnh và kết quả

| Ảnh nguyên bản | Mô tả nguồn | Kết quả thực tế | AI trực tiếp | Qua web, thời gian xử lý được lưu |
|---|---|---|---:|---:|
| normal-pa.jpg, 1929×2207, 906.335 byte | X-quang PA bình thường | No Finding 90,52%; Infiltration 55,91%. Pipeline vẫn báo thâm nhiễm. | 3,751 giây | 8,482 giây |
| pneumonia-ap.jpg, 700×545, 27.014 byte | X-quang AP, viêm phổi | Fibrosis 95,33%; Nodule 91,96%; Cardiomegaly 90,82%; Pleural_Thickening 84,53%; Effusion 67,38%. Pneumonia chỉ 22,52%, không có trong các nhãn phát hiện. | 2,001 giây | 6,546 giây |

Thời gian là từng lượt đo trên CPU, không phải median/p95 và không bao gồm cùng một phạm vi chi phí trong mọi đường gọi. Hai JPEG được tải nguyên bản, không tự đổi màu, cắt ảnh hoặc tăng cường trước khi gửi.

Nguồn ảnh bình thường: [Wikimedia Commons — Normal posteroanterior (PA) chest radiograph](https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg), tác giả Mikael Häggström, CC0. Trang mô tả ảnh ngực bình thường.

Nguồn ảnh viêm phổi: [Wikimedia Commons — Pneumonia x ray](https://commons.wikimedia.org/wiki/File:Pneumonia_x_ray.jpg). Trang mô tả viêm phổi và dẫn CDC Public Health Image Library, giấy phép public domain; các trường tác giả và nguồn chi tiết trên trang còn thiếu. Nhãn này được dùng để đối chiếu theo mô tả trang, chưa phải nhãn do bác sĩ của đề tài xác nhận độc lập.

SHA-256 ảnh bình thường: `4cbbcf805291db949e4ff085ca3c7258b2823de21b2857ae684e6c91ff9a38a4`.

SHA-256 ảnh viêm phổi: `5e54048331f6e2007e5c95df9ecee19d78095b417f1ad804d47acc03dcd833ae`.

## Bằng chứng và hồ sơ thử

Hồ sơ `ONLINE TEST Wikimedia 2026-11-08`, mã `PT-D82498`, thông tin hành chính giả phục vụ kiểm thử. Hai ảnh thuộc hai nguồn độc lập, được gom trong một hồ sơ thử để kiểm tra nhiều lần chụp; không diễn giải chúng thành diễn tiến của một người bệnh.

| Loại | Scan ID | Diagnosis ID |
|---|---|---|
| PA, ảnh bình thường | 6af03702ae0e87f84d081036 | 6af0370bae0e87f84d081037 |
| AP, ảnh viêm phổi | 6af03721ae0e87f84d081038 | 6af03728ae0e87f84d081039 |

Đã kiểm tra ảnh AP hiển thị đúng 700×545, attention Fibrosis hiển thị 384×384. Khi chọn lại phim PA, panel chuyển đúng Infiltration 55,9%; khi chọn AP chuyển về Fibrosis 95,3%. Reload giữ được hai phim và kết quả đã lưu. Các chỉ số sinh hiệu chưa nhập hiển thị “Chưa nhập”.

PDF tải qua nút **Tải PDF** cho đúng scan AP, phiên bản 0, trạng thái chưa xác minh; không nhập kết luận bác sĩ cho hai ảnh này. PDF một trang đã được trích xuất chữ, render và kiểm tra trực quan, tiếng Việt và bố cục không bị cắt.

File JPEG có nội dung hỏng gửi qua gateway trả HTTP 422, thông báo “Không đọc được ảnh”. Số scan hoàn tất vẫn là 2 sau yêu cầu lỗi.

Tất cả file minh chứng trong `D:\NCKH\CheXNet\.runtime\online-test-2026-11-08\`:

- `normal-result.json`, `pneumonia-result.json`: đầu ra đầy đủ của AI trực tiếp.
- `web-scan-*.json`: dữ liệu đọc lại từ gateway cho từng phim.
- `web-verification.json`: đối chiếu 15 xác suất, preview HTTP 200, thử file hỏng HTTP 422.
- `verify-web.cjs`: script gọi lại các kiểm tra API trên hồ sơ thử này; dùng tài khoản demo có sẵn, không lưu token.
- `web-comparison.png`: ảnh chụp giao diện hiển thị hai phim nguồn.
- `web-heatmap.png`: ảnh chụp attention và trạng thái YOLO.
- `pneumonia-report.pdf`, `report-page-1.png`: phiếu tải qua giao diện và bản render đã kiểm tra.
- `health.json`: trạng thái AI đầu lượt kiểm thử.

Đã đối chiếu SHA-256 của 25 tệp kỹ thuật gốc theo `.runtime/pre-web-merge/protected-sha256.json`: không tệp nào thay đổi.

## Giới hạn và việc cần xử lý tiếp

1. Kiểm tra riêng ngưỡng phân loại và quy tắc báo cáo khi No Finding cùng cao với một bệnh. Ảnh PA cho thấy tình huống này rõ ràng. Log hiện báo dùng ngưỡng mặc định 0,5 do chưa có ngưỡng tối ưu; không tự điều chỉnh từ hai mẫu.
2. Đánh giá checkpoint trên một bộ kiểm định có nhãn độc lập, đối chiếu thứ tự 15 lớp và preprocessing với huấn luyện. Bất đồng trên ảnh viêm phổi cần được điều tra trước khi khẳng định chất lượng model. Không suy ra nguyên nhân cụ thể chỉ từ hai ảnh.
3. Attention hiển thị được nhưng vị trí đúng sai chưa có nhãn vùng hoặc bác sĩ đánh giá. Độ bất định đang trả 0 trong các mẫu này, không diễn giải thành chắc chắn tuyệt đối.
4. YOLO/Cascade chưa được kiểm chứng suy luận thật. Web ghi rõ chưa nạp; người dùng tải ảnh không thay thế nhu cầu có trọng số detector nếu muốn bật nhánh này.
5. Kho CBIR hiện có một ca thử tổng hợp từ lượt kiểm thử trước. Hai ảnh mới chưa được xác minh hoặc thêm vào kho. Chưa đánh giá chất lượng truy hồi từ dữ liệu tham chiếu thật.
6. Có chữ mô tả model cũ trong lúc tải ở modal (“DenseNet-121 / ConvNeXtV2”), trong khi phiếu và dữ liệu thật ghi Hybrid ConvNeXtV2–SwinV2. Khi chọn phim PA cũ, nhãn phim đối chiếu vẫn ghi “Phim tham chiếu cũ” cho phim AP mới hơn; nên đổi nhãn thành “Phim đối chiếu” hoặc chọn phim theo thời gian. Đây là vấn đề diễn đạt giao diện được phát hiện, chưa sửa trong lượt chỉ chạy và kiểm thử này.
7. Chưa đo tải đồng thời, độ ổn định dài hạn, median/p95 hoặc bộ DICOM từ máy chụp trong lượt này.

Chạy lại dịch vụ bằng `D:\NCKH\CheXNet\start-web.ps1`. Hồ sơ thử và hai phim được giữ lại để người dùng mở kiểm tra; không thay đổi hồ sơ thật hoặc huấn luyện lại model.
