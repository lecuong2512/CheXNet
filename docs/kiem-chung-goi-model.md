# Kiểm chứng một ảnh đã được gọi vào model

## Trên web

Khởi động bản web tích hợp bằng `start-web.ps1`; script dùng adapter `Backend.ai_service.main` tại cổng 8000 cùng gateway Node tại cổng 3006.

Mở lần chụp mới → tab **Báo cáo** → **Thông tin lần phân tích**. Phiếu kết quả cũng có cùng mục này. Mỗi lượt có mã `inf_…`, thời điểm hoàn tất, thiết bị, thời gian xử lý và trạng thái quan sát thực thi riêng cho Hybrid/YOLO.

Mã được tạo ở AI service, lưu trong `inferenceMeta.trace` cùng diagnosis ở MongoDB, sau đó được đọc lại để hiển thị và xuất PDF. Mở lại ảnh không tạo lượt gọi mới; hồ sơ cũ chưa lưu trace không được gán một mã giả.

## Đối chiếu log

Nhật ký riêng ghi nối tiếp và được giữ qua lần khởi động server:

```powershell
Get-Content D:\NCKH\CheXNet\.runtime\inference-audit.log -Tail 20 -Wait
```

Tìm bằng mã đã sao chép từ phiếu:

```powershell
Select-String -Path D:\NCKH\CheXNet\.runtime\inference-audit.log* -SimpleMatch 'inf_MA_DA_SAO_CHEP'
```

Một lượt hoàn tất có các sự kiện `request_started`, `model=hybrid event=forward_completed`, `model=yolo event=forward_completed` khi quan sát được các model tương ứng, và `request_completed`. Lỗi được ghi `request_failed`. File hiện tại giới hạn khoảng 5 MB, có ba bản xoay vòng; các log quá cũ có thể hết thời gian lưu.

## Bằng chứng được lấy ở đâu

`Backend/ai_service/inference_trace.py` đăng ký PyTorch forward pre-hook và forward hook trên chính model Hybrid đã nạp và DetectionModel bên trong YOLO. Hook chỉ ghi nhận thời điểm, số lần gọi và kích thước tensor; không thay input, output, ngưỡng hay trọng số. Mỗi hook chỉ nhận sự kiện ở thread xử lý yêu cầu đó và luôn được gỡ sau khi hoàn tất/lỗi, tránh lẫn yêu cầu hoặc giữ tensor trong RAM.

Ví dụ thực tế: Hybrid nhận `[1, 3, 384, 384]`, trả logits `[1, 15]`; YOLO nhận `[1, 3, 928, 1024]`, trả tensor thô `[1, 18, 19488]` trước khi lọc thành khung. Kích thước tensor thô không phải số khung cuối cùng. Metadata không ghi tên bệnh nhân, nội dung ảnh, giá trị tensor hoặc thông tin đăng nhập vào nhật ký mới.

Nếu observer không hoạt động hoặc model không có forward được ghi nhận, giao diện hiển thị **Chưa ghi nhận thực thi**. Không suy ra đã gọi model chỉ từ cờ `model_loaded`, HTTP 200 hoặc một ảnh có sẵn.

Thông tin này kiểm chứng việc thực thi, không chứng minh dự đoán bệnh chính xác. Kết quả phân loại vẫn phải đọc cùng báo cáo đánh giá model.

## Kiểm tra triển khai

- Unit test: output không đổi, hook được dọn sau thành công/lỗi, không lẫn thread, observer lỗi không làm hỏng suy luận, log ghi nối tiếp và không lặp handler.
- Kiểm tra thật qua gateway → AI → MongoDB → PDF: cùng trace ID, Hybrid/YOLO đều ghi nhận forward; 15 xác suất trùng kết quả trước khi thêm tracing.
- Bằng chứng cục bộ: `.runtime/inference-trace-e2e/` và `.runtime/inference-trace-tests.log`.
- Lượt kiểm tra mới nhất có mã `inf_7c51a84698f54fbb90fbd88095b5857b`, Hybrid/YOLO đều ghi nhận một forward, đủ 15 điểm phân loại và 6 khung. Mã khớp trong phiếu, PDF, bản ghi đã đọc lại từ database và file audit. 33 kiểm thử Python đạt; build FE/BE đạt. 25 file kỹ thuật gốc giữ nguyên theo manifest.
