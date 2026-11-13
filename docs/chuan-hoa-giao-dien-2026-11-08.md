# Chuẩn hóa giao diện và kiểm thử ngày 08/11/2026

Workspace: `D:\NCKH\CheXNet`. Web đang chạy tại http://127.0.0.1:5173/.

## Thay đổi đã thực hiện

- Thống nhất điều hướng: Phân tích X-quang, Hồ sơ & lịch sử, Thống kê, Trạng thái hệ thống, Hướng dẫn sử dụng. Bỏ các liên kết trợ giúp không hoạt động, thêm trang hướng dẫn thật.
- Dùng chung nhãn tiếng Việt cho đủ 15 lớp, trạng thái phiếu, tư thế chụp và thiết bị. Giữ nguyên khóa lớp tiếng Anh khi gửi/nhận API. Dùng dấu phẩy thập phân cho số hiển thị. Cấu hình ngôn ngữ tiếng Việt cho các thành phần Ant Design, bao gồm phân trang và hiện/ẩn mật khẩu.
- Trang phân tích có một khung ảnh, ảnh gốc hiển thị trước. Chuyển sang bản đồ chú ý bằng nút rõ ràng. Giữ phóng to, thu nhỏ, tương phản, đảo màu, đặt lại; đưa thông số mô hình và định vị bổ sung vào mục mở rộng.
- Bỏ chiều cao ép cứng và các vùng cuộn lồng nhau ở trang phân tích/hồ sơ. Khung ảnh máy tính đo được cao 513 px ở viewport 1440×900. Trên điện thoại, các nhóm điều khiển tự xuống hàng.
- Tách phần xem ảnh, đủ 15 điểm AI, báo cáo và ca tương tự thành bốn thẻ. Cột tóm tắt giữ kết quả và nút mở phiếu; khuyến nghị có thể mở khi cần.
- Trang hồ sơ thu gọn sinh hiệu và so sánh phim. Bộ chọn phim có mã phân biệt những lần chụp cùng ngày/giờ; chuyển từ trang phân tích sang hồ sơ giữ đúng scan đang chọn. Sửa lỗi khóa React trùng giữa khung phân tích và phiếu, đã kiểm tra lại sau tải mới trang.
- Trang lịch sử hiển thị bảng trên máy tính, thẻ hồ sơ trên điện thoại. Ảnh thu nhỏ dùng đường tải có xác thực. Tìm kiếm trên thanh đầu trang dẫn tới lịch sử và truyền từ khóa. Bộ lọc thời gian dùng createdAt thật; bộ lọc trạng thái dùng trạng thái phiếu đã lưu.
- CSV xuất đúng danh sách đã lọc, có tiêu đề tiếng Việt, UTF-8 BOM và escape dấu nháy. Hồ sơ chưa có phim được ghi “Chưa phân tích”, không tự gán bình thường.
- Hai cửa sổ thêm bệnh nhân/tải phim dùng Modal có quản lý focus và phím Esc. Sắp lại nút thao tác trên màn hình nhỏ, có bước hồ sơ/ảnh, có chọn tệp bằng bàn phím. Thay tiến độ phần trăm giả bằng trạng thái đang xử lý.
- Bổ sung chỉ dẫn cho quên mật khẩu và thông báo phiên hết hạn; thống nhất giao diện tối và cách hiển thị thông báo theo chủ đề.
- Công cụ nghiên cứu cũ được dịch các nhãn còn tiếng Anh, sửa trạng thái kết nối tương thích health hiện tại và mở điều hướng ở màn hình nhỏ. Thay các số hiệu năng cứng không gắn với checkpoint bằng thông tin kết nối/thiết bị thực. Ảnh minh họa được ghi rõ là dùng thử giao diện.

## Kiểm thử đã quan sát trực tiếp

| Phép thử | Kết quả |
|---|---|
| Đăng nhập/đăng xuất | Thành công bằng tài khoản dùng thử có sẵn |
| Trang đăng ký để trống | Báo đủ lỗi bắt buộc bằng tiếng Việt; không tạo tài khoản |
| Quên mật khẩu | Hiện chỉ dẫn liên hệ quản trị viên; không còn liên kết rỗng |
| Tìm PT-31C124 từ thanh đầu trang | Chuyển tới lịch sử, tìm đúng hai phim |
| Lọc Đã xác minh trong PT-31C124 | Kết quả rỗng đúng trạng thái; nút xuất được vô hiệu hóa |
| Ảnh thu nhỏ lịch sử | Tải được cả hai ảnh có xác thực, naturalWidth > 0 |
| Xem ảnh/bản đồ chú ý | Ảnh chú ý tải được 384×384; zoom 125%, tương phản 1,8 và đặt lại hoạt động |
| Thẻ điểm AI | Đủ 15 điểm, tên lớp tiếng Việt; không thay giá trị |
| Bệnh nhân có hai phim cùng phút | Hiển thị hai mã khác nhau, mở đúng phim 08103C trong hồ sơ qua tham số scan |
| Tạo hồ sơ và tải ảnh thật trên điện thoại | Hoàn thành từ form tới model và trang hồ sơ, giữ đúng kết quả sau reload |
| Tệp .txt ở cửa sổ tải phim | Bị từ chối với thông báo tiếng Việt; nút phân tích vẫn bị khóa |
| Phím Esc khi chưa xử lý ảnh | Đóng cửa sổ tải phim, số dialog về 0 |
| Xuất PDF từ phiếu | Tải được PDF một trang, đúng PT-AD6289 và có chữ tiếng Việt |
| CSV trên điện thoại | Lọc còn một thẻ, không render bảng desktop; CSV có đúng một dòng hồ sơ mẫu |
| Thống kê/trạng thái/hướng dẫn | Mở được, nhãn bệnh và nội dung đã Việt hóa, không tràn ngang ở 390 px |
| Giao diện tối | Nền menu/card/input và màu chữ theo chủ đề, chuyển lại sáng được |
| Viewport 1440×900 | Một khung ảnh; chiều cao ảnh 513 px; chiều rộng trang 1440 px |
| Viewport 390×844 | Một khung ảnh, main không rộng hơn viewport; thao tác form và xuất tệp được |
| Viewport 320×740 | Chiều rộng trang 320 px, main 314 px, một khung ảnh |
| Công cụ nghiên cứu cũ | Báo dịch vụ AI sẵn sàng CPU, mở được cấu hình với nhãn tiếng Việt |
| Console lượt cuối sau tải mới | Không ghi nhận lỗi trong browser-checks.json |

Đã chạy `npm --prefix Frontend run build`: TypeScript và build Vite thành công. Còn cảnh báo không chặn build về kích thước bundle và dữ liệu Browserslist cũ; không thay dependency trong đợt chỉnh này.

Không thực hiện lại ghi kết luận bác sĩ, lưu ca tham chiếu hoặc in qua hộp thoại máy in trong lượt giao diện này. Đã kiểm tra mở phiếu và tải PDF. Chưa có fixture giao diện với phim cũ ngoài khoảng thời gian để chấm lại trường hợp loại trừ ngày; bộ lọc thời gian đã được nối với ngày tạo thực của bản ghi.

## Dữ liệu thử và bảo toàn model

Tạo hồ sơ **KIỂM THỬ GIAO DIỆN - Hồ sơ mẫu**, mã **PT-AD6289**, dùng thông tin hành chính giả. Ảnh là normal-pa.jpg công khai đã dùng ở lượt kiểm thử trước. Hồ sơ không được xác minh thành kết luận bác sĩ và không thêm vào CBIR.

- Scan: `6af04601ae0e87f84d081041`.
- Diagnosis: `6af04610ae0e87f84d081042`.
- Đọc lại từ gateway: trạng thái done, đủ 15 xác suất, sai khác tối đa so với đầu ra mẫu trước sửa bằng **0**.
- Frontend, gateway và AI đều trả HTTP 200 trong lần đối chiếu.
- Đã đối chiếu SHA-256 **25 tệp kỹ thuật gốc**, số tệp thay đổi **0**. Giữ nguyên checkpoint, ngưỡng, huấn luyện, tiền xử lý và thuật toán suy luận. Đợt này không làm thay đổi các chỉ số đánh giá model trên 200 ảnh.

## File bàn giao

- Nhãn dùng chung: `Frontend/src/utils/display.ts`.
- Khung ảnh dùng chung: `Frontend/src/components/common/XrayViewer.tsx`.
- Phân tích: `Frontend/src/modules/dashboard/pages/DashboardPage.tsx`, `Frontend/src/components/common/HybridAnalysisPanel.tsx`.
- Hồ sơ/lịch sử: `Frontend/src/modules/patients/pages/PatientProfilePage.tsx`, `Frontend/src/modules/history/pages/PatientHistoryPage.tsx`.
- Hướng dẫn: `Frontend/src/modules/help/pages/HelpPage.tsx`.
- Bằng chứng: `D:\NCKH\CheXNet\.runtime\ui-vietnamese-2026-11-08\`, gồm desktop-vietnamese.jpg, desktop-dark.jpg, mobile-analysis.jpg, mobile-upload.jpg, phieu-mau.pdf, lich-su-mau.csv, browser-checks.json, narrow-layout.json, api-verification.json, protected-check.json và build.log.
- Snapshot trước sửa của Frontend/src và hai trang HTML: `.runtime/ui-vietnamese-before.zip`. Snapshot này không bao gồm toàn bộ thư mục legacy.

Không xóa dữ liệu thử cũ hoặc hồ sơ người dùng. Trang phân tích đã được giữ mở để tiếp tục xem và sử dụng.
