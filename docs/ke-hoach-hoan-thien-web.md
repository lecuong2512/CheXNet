# Kế hoạch nối FE, BE và model CheXNet

Mục tiêu của lần triển khai là để bác sĩ thao tác hoàn toàn trong web chính. Từ một lần chụp, bác sĩ có thể xem ảnh, kiểm tra kết quả Hybrid và YOLO, xác minh phiếu, thêm ca tham chiếu và mở lại mọi kết quả đã lưu.

## Luồng dữ liệu

React gửi ảnh và mã bệnh nhân đến API scan của Node. Node gọi FastAPI `POST /v3/predict`. Adapter kiểm tra ảnh rồi gọi `Backend.main.predict_image`. Hybrid trả 15 xác suất qua `all_probabilities` và danh sách phát hiện qua `predictions`. Node lưu lần chụp ở `Scan`, lưu kết quả ở `Diagnosis.hybridAnalysis` và liên kết bằng `scanId`. Khi bác sĩ xem lại hồ sơ, React đọc đúng kết quả đã lưu. Luồng này đã chạy với PNG.

Các đầu việc dưới đây chỉ sửa FE, BE nghiệp vụ và adapter phục vụ suy luận. Mô hình, trọng số Hybrid, ngưỡng, bộ sinh heatmap, Cascade, báo cáo gốc và thuật toán FAISS thuộc phần kỹ thuật đã bảo toàn. Những trường và API mới dưới đây là thiết kế đề xuất để triển khai.

## Danh sách triển khai theo thứ tự

### 1. Dùng chung dữ liệu của một lần chụp

- **FastAPI** giữ hợp đồng hiện tại của `/v3/predict`. Bổ sung trạng thái YOLO tại thời điểm phân tích, dựa trên lần thực thi thực tế. Nếu pipeline chỉ cho biết đã nạp trọng số, ghi đúng mức thông tin đó. Tách tình trạng dịch vụ hiện tại qua `/health`.
- **Node** mở rộng hợp đồng trong `src/modules/scans` để lưu trạng thái này cùng kết quả, phiên bản model và thời gian xử lý. Sửa tình huống đã tạo Scan nhưng tạo Diagnosis thất bại: dọn bản ghi cùng tệp của yêu cầu và chỉ phát sự kiện hoàn thành khi cả hai đã lưu thành công.
- **React** khai báo kiểu chung cho dữ liệu Hybrid trong `src/stores/baseApi.ts` và panel. Dashboard và hồ sơ bệnh nhân dùng cùng ánh xạ. Trường thiếu trên hồ sơ cũ hiển thị “Chưa ghi nhận”. Lấy 15 xác suất từ `all_probabilities`, lấy nhãn vượt ngưỡng từ `hybridAnalysis.predictions`.
- **Kiểm tra** mở hai scan cùng bệnh nhân. Đổi scan và tải lại trang vẫn hiển thị đúng ảnh, đúng kết quả và đúng trạng thái của lần chụp đó.

### 2. Nối upload DICOM và ảnh xem lại

- **FastAPI** thêm bộ đọc DICOM bên cạnh `ai_service/inference.py`. Giải mã pixel, áp dụng rescale/window và MONOCHROME1 khi phù hợp. Sinh một bản PNG hiển thị từ kết quả giải mã, dùng cùng ảnh cho bộ xác thực và pipeline gốc. Trả lỗi 422 có nguyên nhân cho tệp hỏng hoặc định dạng chưa hỗ trợ.
- **Node** cập nhật bộ nhận upload để nhận cả tệp DICOM trình duyệt gửi với MIME trống hoặc chung. Dựa vào kết quả giải mã để xác nhận định dạng. Giữ liên kết với tệp nguồn, lưu bản xem dưới dạng tệp ảnh, trả `previewUrl` theo scan. Tệp DICOM gốc có thể chứa thông tin bệnh nhân nên lưu ngoài thư mục tĩnh `/uploads`.
- **React** sửa phần chọn và xem ảnh trong `PatientProfilePage.tsx`. Sau upload, dùng `previewUrl` cho DICOM và dùng `imageUrl` có sẵn cho ảnh cũ. Hiển thị tiến trình, lỗi dịch vụ và nút thử lại.
- **Kiểm tra** chạy PNG mốc, DICOM có ảnh đối chiếu, tệp hỏng và ảnh sai loại. Kiểm tra chiều sáng, đường dẫn ảnh sau tải lại và tình trạng hồ sơ sau yêu cầu thất bại.

### 3. Đưa kết quả Hybrid và YOLO lên cùng trình xem

- **Node** bổ sung API trạng thái AI cho web qua gateway, đọc FastAPI `/health`. Dùng trạng thái này cho màn hình hệ thống. Lịch sử bệnh án dùng trạng thái đã lưu ở bước 1.
- **React** hoàn thiện `HybridAnalysisPanel.tsx`. Khi chọn một bệnh, hiển thị heatmap, vị trí, diện tích, mức tin cậy và độ bất định đúng theo kết quả tương ứng. Cho chuyển qua ảnh gốc của chính lần chụp. Với `yolo_detections`, hiển thị lớp VinDr, tọa độ, điểm YOLO và điểm Cascade. Dùng ảnh Display Fusion do Python sinh.
- **Kiểm tra** chọn qua lại hai nhãn có phát hiện, mở ca không có nhãn vượt ngưỡng, thử khi thiếu YOLO. Khi có trọng số chuyên dụng, chạy thêm hai ca để phân biệt có khung với đã chạy nhưng không có khung vượt ngưỡng. UI phải phản ánh được cả trường hợp trạng thái chưa rõ.

### 4. Nối dự thảo, xác minh và in đúng phiếu

- **Node** dùng lại `PATCH /api/v1/diagnoses/:id/verify`. Trả đủ trạng thái, kết luận bác sĩ, người xác minh và thời điểm cùng `diagnosisId`. Giữ `hybridAnalysis.clinical_report` làm bản dự thảo ban đầu.
- **React** dùng một phiếu chung cho Dashboard và Patient Profile. Ánh xạ phần mô tả, kết luận AI và khuyến nghị từ dữ liệu model. Hiển thị thông tin kỹ thuật khi scan có dữ liệu. Bác sĩ ghi kết luận, xác minh hoặc đánh dấu cần xem lại trên đúng `diagnosisId`. Thay nút in đang chỉ hiện thông báo bằng bản in của phiếu được chọn.
- **Kiểm tra** xác minh lần chụp thứ nhất, đánh dấu lần thứ hai, mở lại và in cả hai. So sánh mã bệnh nhân, mã scan, nội dung AI, kết luận và người xác minh.

### 5. Nối thao tác lưu ca đã xác minh và tra cứu FAISS

- **Node** thêm `POST /api/v1/diagnoses/:id/reference-case` cho vai trò có quyền. Tra ngược từ chẩn đoán đến scan, lấy ảnh chuẩn hóa đúng lần chụp và gọi chức năng `/api/add_reference_case` của Python. Dùng mã tham chiếu ẩn danh để ánh xạ từ kết quả FAISS về scan. Định danh lần lưu bằng `diagnosisId` và xử lý cả nhấn đôi lẫn trường hợp Python đã lưu nhưng gateway mất phản hồi.
- **FastAPI** dùng lại chức năng trích embedding và thêm ca gốc. Ở adapter bổ sung kiểm tra mã tham chiếu đã có trước khi ghi lại, tuần tự hóa thao tác ghi chỉ mục khi cần. Với DICOM, dùng cùng bản ảnh đã xử lý ở bước 2.
- **React** thêm nút “Lưu ca tham chiếu” cho chẩn đoán đã xác minh, cho bác sĩ chọn nhãn xác nhận. Hiển thị đã lưu, đang lưu hoặc lỗi. Kết quả `similar_cases` hiển thị độ tương đồng, nhãn và liên kết mở đúng ca nếu người dùng có quyền.
- **Kiểm tra** gửi lại yêu cầu chỉ tạo một ca, truy vấn bằng ảnh khác, mở đúng ca đối chiếu và lặp lại sau khi khởi động lại FAISS.

### 6. Dùng kết quả model thật trên trang nghiên cứu

- **Node** sửa `research.service.ts`. Những biểu đồ đếm ca phát hiện dùng nhãn vượt ngưỡng từ `hybridAnalysis.predictions`. Mỗi bệnh chỉ được đếm một lần cho một scan. Lần chụp cũ thiếu danh sách phát hiện cần được nhận diện rõ khi tổng hợp. Kiểm tra cả API xu hướng và heatmap thống kê.
- **React** dùng cùng định nghĩa cho số trên màn hình và file CSV. Tách số liệu vận hành khỏi benchmark nghiên cứu. Tên phương pháp heatmap và AUROC phải dẫn về đúng kỹ thuật, tập đánh giá và phiên bản tài liệu.
- **Kiểm tra** với một scan có hai phát hiện và một scan không có phát hiện, biểu đồ tăng đúng cho hai nhãn liên quan. Trang hệ thống hiển thị giá trị thực hoặc trạng thái chưa có dữ liệu khi không đo được.

## Hoàn tất và bàn giao

1. Build cả React và Node, chạy test adapter, rồi kiểm thử bằng giao diện từ đăng nhập đến xem và in lại phiếu.
2. Dùng một bệnh nhân thử có PNG và DICOM, một ca lỗi, các ca YOLO, một ca FAISS đã xác minh và một ảnh truy vấn riêng. Lưu mã scan và kết quả mong đợi để đối chiếu UI với dữ liệu đã lưu.
3. Khởi động lại bằng `start-web.ps1`, mở lại lịch sử, phiếu và ca tương tự. Đối chiếu SHA-256 của 25 file kỹ thuật gốc với manifest trước khi bàn giao.

**Đầu vào cần cho kiểm thử cuối.** DICOM ngực ẩn danh có ảnh đối chiếu để kiểm tra cách hiển thị. Trọng số YOLO của đề tài để kiểm tra định vị thật. Ca tham chiếu có nhãn để kiểm tra tra cứu. Có thể triển khai đầu việc 1–6 với checkpoint Hybrid hiện có và ghi riêng các phép thử còn chờ đầu vào.

## Bàn giao cho agent thực hiện

**Trạng thái tại thời điểm lập kế hoạch:** sáu đầu việc trên đang chờ triển khai. Các giao diện API mới trong bảng dưới đây là đề xuất để FE và BE cùng bám. Đánh dấu một đầu việc hoàn thành khi có kết quả kiểm tra từ đường vào bình thường của người dùng. Ghi riêng phần đã viết mã và phần đã kiểm chứng bằng model thật.

### Bắt đầu từ bản đang chạy

1. Làm việc tại `D:\NCKH\CheXNet`. Đọc [biên bản hợp nhất](web-merge.md), chạy `git status --short` và giữ các thay đổi đang có, gồm cả FE/BE chưa được track. Repo nguồn `D:\Projects\Science_Of_AI\CheXNet` chỉ là bản tham chiếu. Báo cáo chính thức nằm ở Git commit `5f6c424`, thư mục `docs`.
2. Dùng `Trainedmodel/hybrid_model.pth` và môi trường của dự án. Chạy các lệnh kiểm tra ở cuối tài liệu. Đọc nội dung health trên cổng 8000. Cổng 3006 là gateway, 5173 là FE. Ghi lại trạng thái thực tế của Hybrid, validator, YOLO, thiết bị và DB trước khi sửa.
3. Dùng ảnh gốc `Results/EX/Cardiomegally.png` làm mốc. Ghi 15 xác suất theo tên nhãn, tập nhãn phát hiện, phiên bản model và thời gian trên cùng máy. Hai lần chạy cùng ảnh phải có cùng tập nhãn và sai số xác suất tối đa mục tiêu `1e-5` trên cùng runtime. Nếu khác, xác định nguyên nhân trước khi thay đổi ngưỡng kiểm thử.
4. Sử dụng database thử riêng và thư mục tạm riêng cho upload, ảnh xem và FAISS. Legacy dùng các đường dẫn FAISS tương đối nên phải kiểm tra đường dẫn hiệu lực trước khi chạy ca ghi dữ liệu. Bản khởi chạy local hiện dùng một tiến trình AI. Kiểm tra process và đường dẫn trước khi restart. `start-web.ps1` bỏ qua dịch vụ đã trả HTTP 200, nên chỉ gọi lại script sẽ không tự nạp backend mới.
5. Thứ tự phụ thuộc: hoàn tất hợp đồng scan (1), sau đó DICOM (2) và trình xem (3). Phiếu (4) dùng hợp đồng (1). FAISS (5) dùng ảnh chuẩn từ (2) và kết quả xác minh từ (4). Thống kê (6) dùng ngữ nghĩa nhãn ở (1). Có thể thực hiện phiếu và thống kê trong khi chờ checkpoint YOLO.

### Hợp đồng giữa các lớp

Các đường Node dưới đây có tiền tố `/api/v1`. Dùng envelope thành công/lỗi, middleware xác thực và vai trò hiện có. Thêm trường tùy chọn vào dữ liệu cũ. Cập nhật interface, schema, validator, tài liệu API và kiểu React trong cùng hạng mục.

| Thao tác | Hợp đồng cần giữ hoặc bổ sung | Quy tắc cho client |
|---|---|---|
| Upload hiện có | `POST /scans/:patientId/upload`, multipart tên `image`. Node gọi AI `POST /v3/predict`, multipart tên `file`. Thành công trả scan và diagnosis. Thêm `previewUrl` khi ảnh nguồn cần chuyển đổi. | Dùng Mongo ID cho API upload và giữ patientCode cho điều hướng như hiện tại. Cả `AddPatientModal` và `PatientProfilePage` dùng chung quy tắc chọn tệp. Tải ảnh lỗi sau khi đã tạo bệnh nhân thì thử lại trên bệnh nhân đó. |
| Chi tiết lần chụp | Giữ nguyên `Diagnosis.predictions` là 15 xác suất và toàn bộ `hybridAnalysis` gốc. Bổ sung trên Diagnosis một object riêng `inferenceMeta` có phiên bản hợp đồng và trạng thái detector khi chạy. | Dùng `scanId` để chọn ảnh và `diagnosisId` để sửa phiếu. Không dựa vào vị trí phần tử hay lần chụp gần nhất. Nếu chưa có diagnosis, hiển thị “Chưa có kết quả”. |
| Health đề xuất | `GET /system/ai-health` đọc AI `GET /health`, trả trạng thái model, validator, YOLO, device và thời điểm đọc. Backend mất kết nối trả lỗi dịch vụ. | Trang cấu hình hiển thị thời điểm và trạng thái hiện tại. Trạng thái quá khứ đọc từ diagnosis. Giá trị thiếu để trống. Các slider ngưỡng hiện lưu localStorage, nên ghi rõ chỉ là cấu hình hiển thị nếu thực sự dùng hoặc ẩn khi chưa có tác dụng. |
| Xác minh hiện có | `PATCH /diagnoses/:id/verify` gửi `status: verified hoặc flagged` và `notes`. Server xác định `verifiedBy`, `verifiedAt`. Đề xuất thêm điều kiện phiên bản dựa vào `updatedAt` cho client mới. | Nếu hai bác sĩ sửa cùng lúc, client có phiên bản cũ nhận 409 và có thể xem lại thay đổi. Không hiển thị thành công trước khi server xác nhận. |
| Ca tham chiếu đề xuất | `POST /diagnoses/:id/reference-case` gửi các nhãn đã xác nhận. Server kiểm tra `verified`, quyền, nhãn hợp lệ và đúng scan. Trả `referenceId` ổn định. Lần đầu 201, gửi lại cùng dữ liệu 200; nếu nhãn mâu thuẫn với ca đã lưu trả 409. | Khóa nút trong lúc gửi chỉ để hỗ trợ thao tác. Quy tắc chống trùng phải thực thi ở server. Ca nhập sẵn không có liên kết bệnh án vẫn hiển thị kết quả tương đồng với thông tin hiện có. |
| Lỗi upload/AI | 413 quá dung lượng, 422 dữ liệu hoặc định dạng không dùng được, 503 dịch vụ chưa sẵn sàng, 504 hết thời gian chờ. Lỗi lập trình hoặc vi phạm hợp đồng model trả lỗi server và có mã yêu cầu để tra log. | FE giữ bệnh nhân và lần thao tác, hiển thị lỗi đúng nguyên nhân, cho chọn lại tệp hoặc thử lại. Lịch sử không xuất hiện bản phân tích hoàn tất khi còn thiếu kết quả. |

### Ghi an toàn, xử lý lỗi và phục hồi

- **Scan + Diagnosis + tệp.** Mongo local cần hoạt động ở chế độ standalone. Dùng `processing` làm trạng thái nội bộ khi bắt đầu ghi, chỉ chuyển `done` sau khi tệp xem và diagnosis đã lưu. Các API danh sách, chi tiết và thống kê phải cùng lọc đúng trạng thái. Ràng buộc unique trên `Diagnosis.scanId` đã có. Nếu tiến trình chết giữa chừng, có bước đối soát bản ghi dang dở theo ID, tuổi yêu cầu và sự tồn tại của tệp. Hoàn tất bản ghi có đủ dữ liệu hoặc đánh dấu failed và dọn đúng tệp thuộc yêu cầu. Chỉ phát sự kiện sau khi commit nghiệp vụ. Lỗi socket sau commit không được xóa ảnh thành công.
- **Retry upload.** FE sinh một khóa cho một lần bấm phân tích và giữ khóa khi thử lại sau mất kết nối. Node đặt ràng buộc duy nhất cho khóa trong phạm vi bệnh nhân/người gửi, với chỉ mục partial cho bản ghi có khóa. Cùng khóa và nội dung trả lại scan đã tạo hoặc trạng thái đang xử lý. Cùng khóa khác nội dung trả 409. Chọn ảnh mới hoặc chủ động tạo lần chụp khác sinh khóa mới. Không khử trùng chỉ theo hash ảnh vì hai lần khám hợp lệ có thể dùng cùng ảnh.
- **DICOM.** Multer hiện ghi vào `UPLOAD_DIR` trước khi dịch vụ AI xử lý. Chuyển nơi tiếp nhận mới sang staging riêng ngoài static ngay từ đầu. Dùng tên do server sinh, giới hạn kích thước file theo cấu hình và số pixel trước giải nén. Chốt phạm vi ban đầu: ảnh ngực đơn khung grayscale MONOCHROME1/2, transfer syntax đã kiểm thử với codec thực sự cài. Áp dụng modality LUT/rescale trước VOI LUT/window, xử lý padding và ảnh hằng, đảo sáng đúng một lần. Ngoài phạm vi trả 422. Bản PNG xem lại chỉ chứa pixel. Đặt dưới route có xác thực; FE lấy blob bằng request mang token và thu hồi object URL khi đổi scan. Check cả chuyển ảnh nhanh và mất kết nối trong khi tải. Giữ đường đọc ảnh cũ. Xác định đơn vị chuyển preview giữa Python và Node: ưu tiên trả bytes PNG qua kênh nội bộ hoặc đường dẫn staging chỉ khi cùng volume và có kiểm tra đường dẫn. Không lưu base64 preview thêm một lần vào Mongo.
- **YOLO.** `YOLODetector.predict` có nhánh bắt exception rồi trả mảng rỗng. Từ mảng rỗng đơn thuần không thể kết luận detector chạy thành công. Với mã gốc hiện tại chỉ cam kết các trạng thái có bằng chứng: chưa nạp, có detection, đã nạp nhưng chưa xác nhận kết quả rỗng, hoặc chưa ghi nhận với hồ sơ cũ. Chỉ bổ sung trạng thái “đã chạy, không có khung vượt ngưỡng” khi adapter có tín hiệu thực thi chắc chắn. Không chạy YOLO lần hai chỉ để lấy trạng thái. FE sử dụng ảnh fusion đã sinh, không tự dựng hộp từ kết quả phân loại.
- **FAISS.** `cbir.py` cập nhật RAM, ghi file index, rồi ghi JSON metadata. Mutex không khắc phục mất điện giữa hai lần ghi. Dùng adapter sở hữu một writer cho cả đường mới và đường thêm ca legacy. Chạy thuật toán thêm ca gốc trên bản sao trong thư mục thế hệ mới, xác minh `ntotal == len(metadata)` và mã tham chiếu duy nhất, sau đó công bố bằng đổi con trỏ manifest nguyên tử trên cùng volume. Đổi finder đang phục vụ sau khi tạo thành công; reader đang chạy dùng snapshot cũ tới hết request. Khởi động lại phải đọc được thế hệ đã công bố, kể cả sau khi tiến trình chết ngay trước đổi con trỏ. Nhận diện cặp file legacy để nhập lần đầu mà không ghi đè. Nếu dữ liệu hỏng, giữ thế hệ tốt gần nhất và báo trạng thái. Thực hiện ở lớp adapter, bảo toàn `Backend/cbir.py`.
- **Đồng bộ Mongo–FAISS.** Node lưu ánh xạ duy nhất `diagnosisId → referenceId` với trạng thái pending hoặc ready. Nếu Python đã công bố thế hệ rồi mất phản hồi, retry kiểm tra mã trong thế hệ hiện hành và hoàn tất ánh xạ. Kiểm tra cả `success` trong JSON vì API gốc có thể trả HTTP 200 kèm `success: false`. Giữ đồng nhất phần mã ẩn danh trong FAISS và liên kết có phân quyền trong ứng dụng. Với ca từ bộ tham chiếu ngoài hệ thống không có ánh xạ, không tạo liên kết giả.

### Tối ưu có số đo

1. **Chi phí inference.** Dùng một model đã nạp trên CPU cho cả dịch vụ. Đo riêng thời gian giải mã, chờ xử lý, suy luận và toàn request. Bắt đầu với một inference đang chạy, đặt giới hạn hàng chờ và timeout nhất quán qua Node và FE. Khi quá tải, trả trạng thái bận có thể thử lại. Kiểm tra health vẫn phản hồi trong lúc xử lý model đồng bộ (lên lịch trên worker/thread phù hợp thay vì chặn event loop). Chỉ tăng song song nếu phép đo RAM và thời gian cho thấy cải thiện.
2. **Mongo và dung lượng trả về.** Payload gốc chứa heatmap base64 cho từng nhãn. Dùng projection tóm tắt khi tải lịch sử và chỉ lấy phần phân tích đầy đủ cho scan được chọn. Giữ endpoint và caller cũ tương thích trước khi thu hẹp payload. Đo kích thước BSON ở ca phát hiện nhiều bệnh. Nếu có nguy cơ chạm giới hạn tài liệu, chuyển phần ảnh sang tệp được quản lý có URL, giữ metadata kết quả, và có adapter đọc cả hai kiểu lưu.
3. **React và mạng.** Dùng RTK Query theo ID; sau khi xác minh chỉ cập nhật scan, hồ sơ, danh sách và thống kê liên quan. Mutation hiện tải lại toàn bộ hồ sơ để dò diagnosis; dùng ID đã nhận từ chi tiết scan. Cache chi tiết đang xem, tải heatmap khi cần, giải phóng blob URL. Không hiển thị số đo mặc định như dữ liệu bệnh nhân thực. Kiểm tra số request trong DevTools khi đổi hai scan, xác minh một scan và quay lại lịch sử.
4. **Nghiên cứu.** Thay tám phép tổng hợp tuần bằng một truy vấn theo khoảng thời gian và nhóm tuần, giữ biên thời gian nhất quán với UI. Đếm distinct `(scanId, disease)`, loại `No Finding` khỏi số bệnh và báo số lần chụp không có dữ liệu phát hiện. Heatmap thống kê hiện gán lớp bệnh theo vùng rồi lặp theo hàng; chỉ hiển thị bản đồ giải phẫu khi có dữ liệu định vị thực đủ để tổng hợp. Nếu thiếu, hiển thị phân bố theo nhãn hoặc “Chưa có dữ liệu vị trí”. Dùng `explain` trước khi thêm index.
5. **Tiêu chí so sánh.** Đo trên cùng phần cứng và phiên bản checkpoint: thời gian nạp lạnh, median/p95 request sau warm-up (ghi số mẫu), peak RAM, kích thước phản hồi lịch sử/chi tiết và số truy vấn. Mục tiêu nghiệm thu kiến trúc: một lần bấm dẫn tới một lần infer trong luồng upload bình thường, tải lại scan dẫn tới 0 lần infer, danh sách không truyền ảnh heatmap, health vẫn phản hồi khi tải. So sánh trước/sau và điều tra hồi quy trên cùng dữ liệu, không lấy số benchmark trong báo cáo làm SLA của máy CPU.

### Ma trận nghiệm thu cho từng bước

| Mã | Thử bằng đường vào của người dùng hoặc điểm gây lỗi có kiểm soát | Điều kiện đạt |
|---|---|---|
| A1 | Upload PNG qua web rồi đổi hai scan, tải lại, mở dashboard và hồ sơ | 15 xác suất hợp lệ, nhãn phát hiện ổn định so với mốc, ảnh/phiếu đúng ID. Hồ sơ cũ thiếu metadata vẫn mở được. |
| A2 | Hai đường upload: thêm bệnh nhân và thêm lần chụp. DICOM thật + bản đối chiếu, MONOCHROME1/2, sai MIME, tệp hỏng, quá giới hạn, codec chưa hỗ trợ | Ảnh hợp lệ xem đúng và cùng pixel nguồn cho AI. Input lỗi có thông báo phù hợp, staging sạch, không tạo ca thành công. Fixture tổng hợp được dùng để kiểm tra phép chuyển đổi nhưng ghi riêng kết quả DICOM thật. |
| A3 | Gửi đôi cùng khóa; ngắt kết nối sau khi lưu; gây lỗi sau Scan trước Diagnosis; ngắt tiến trình sau Diagnosis trước done | Chỉ một scan hoàn chỉnh cho một lần bấm. Có thể khôi phục sau restart, không mất ảnh của ca đã lưu hoặc để lịch sử trỏ tới tệp bị xóa. |
| A4 | YOLO thiếu, lỗi trả mảng rỗng, ca có khung và ca không có khung với checkpoint phù hợp | UI dùng đúng mức bằng chứng. Tọa độ/điểm khớp API. Ca đổi thiết bị hoặc trạng thái sau restart vẫn hiển thị trạng thái lịch sử ban đầu. |
| A5 | Hai phiếu, hai người mở cùng phiếu, một người sửa trước; mở bản in và thử tài khoản không có quyền xác minh | Dự thảo AI nguyên vẹn, kết luận và người xác minh đúng; xung đột được báo; bản in chỉ có lần đang chọn; server chặn quyền sai. |
| A6 | Hai yêu cầu lưu tham chiếu đồng thời; mất phản hồi sau khi FAISS công bố; lỗi giữa ghi index và metadata; restart rồi truy vấn bằng ảnh khác | Đúng một vector và một ánh xạ. Tìm kiếm vẫn dùng thế hệ hợp lệ. Ca legacy không ánh xạ, ca chưa xác minh và truy cập sai quyền đều được xử lý rõ ràng. |
| A7 | Một scan có hai bệnh, một scan No Finding, một hồ sơ cũ thiếu dữ liệu và DB trống | Hai bệnh tăng đúng một lần; biểu đồ và CSV nhất quán; vùng và latency chưa đo không được sinh từ mặc định. |
| A8 | Chạy ứng dụng từ đầu, tải đồng thời hai ảnh, xem health khi đang phân tích rồi mở lại sau restart | Dịch vụ không nạp trùng checkpoint; request thứ hai thành công hoặc nhận trạng thái bận rõ ràng; mở lại ca không suy luận lại. Có số đo trước/sau và manifest gốc khớp. |

### Lệnh kiểm tra và cách tiếp tục khi gián đoạn

Từ PowerShell tại repo đích. Build trước khi chạy. Khi đổi Node hoặc Python, chỉ khởi động lại đúng tiến trình của repo đích rồi chờ health thực sự sẵn sàng. Xem `.runtime/*.error.log` nếu có lỗi.

```powershell
$root = 'D:\NCKH\CheXNet'
Set-Location -LiteralPath $root
Push-Location Backend
try { yarn.cmd build; if ($LASTEXITCODE) { throw 'Backend build failed' } } finally { Pop-Location }
Push-Location Frontend
try { yarn.cmd build; if ($LASTEXITCODE) { throw 'Frontend build failed' } } finally { Pop-Location }
& .\.venv\Scripts\python.exe -m unittest discover -s Backend/tests -v
if ($LASTEXITCODE) { throw 'Adapter tests failed' }
.\start-web.ps1
Invoke-RestMethod 'http://127.0.0.1:8000/health'
Invoke-RestMethod 'http://127.0.0.1:3006/health'
```

Đối chiếu file gốc từ thư mục repo sau mỗi thay đổi có khả năng ảnh hưởng pipeline và trước bàn giao:

```powershell
$manifest = Get-Content .runtime/pre-web-merge/protected-sha256.json -Raw | ConvertFrom-Json
foreach ($entry in $manifest.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256 -ErrorAction Stop).Hash -ne $entry.Value) {
        throw "Protected file changed: $($entry.Name)"
    }
}
```

**Thứ tự phát hành và phục hồi.** Triển khai backend chấp nhận dữ liệu cũ và trường tùy chọn trước, sau đó FE, cuối cùng bật thao tác ghi mới. Chạy riêng các ca lỗi trên DB và thư mục thử. Nếu bản mới lỗi, tắt thao tác ghi mới, quay về bản ứng dụng ổn định, giữ dữ liệu dạng mở rộng và thế hệ FAISS đã xác minh. Không reset repo có thay đổi của người dùng. Nếu checkpoint YOLO hoặc ca tham chiếu thật chưa sẵn sàng, hoàn tất đường xử lý và ghi phép thử tương ứng là “chờ đầu vào”.

**Ghi bàn giao sau mỗi bước.** Ghi số bước, file đã sửa, thay đổi hợp đồng, lệnh chạy, mã ca A1–A8 đã đạt/chưa đạt, dữ liệu thử, kết quả đo và việc tiếp theo ngay trong mục này. Phân biệt kiểm thử adapter bằng fixture, kiểm thử với Hybrid thật và kiểm thử bằng giao diện. Agent kế tiếp bắt đầu từ phép thử chưa đạt đầu tiên, sau khi xem thay đổi hiện có.

### Nhật ký thực hiện ngày 07 tháng 11 năm 2026

Đã bắt đầu bước 1. Thêm requestKey/requestHash và partial unique index theo bệnh nhân, người upload và khóa vào Scan. Gateway dùng trạng thái processing trong quá trình suy luận và lưu chẩn đoán; chỉ công bố done sau khi đã lưu cả hai. Lỗi socket sau commit không xóa dữ liệu. Khi khởi động gateway, đối soát các scan processing/failed trước khi mở cổng.

Adapter trả thêm inferenceMeta riêng, gồm contractVersion, device và detectorStatus. Giá trị loaded-unconfirmed diễn đạt giới hạn của detector gốc khi trả mảng rỗng. React có kiểu Hybrid chung, giữ khóa gửi lại theo File/bệnh nhân/loại scan, chuyển metadata đến panel và hiển thị trạng thái detector theo lần chụp. Hồ sơ bệnh nhân và danh sách scan chỉ lấy scan done. Chưa kết luận hoàn tất A1 vì còn cần kiểm tra nhiều lần chụp qua giao diện, việc nhất quán ở tất cả API chẩn đoán/thống kê và kiểm thử ảnh cũ.

Đã kiểm chứng bằng model thật: ảnh Results/EX/Cardiomegally.png có đủ 15 xác suất; sau khi restart adapter mới, sai số từng xác suất so với mốc tối đa 1e-5. Mốc lưu ở .runtime/execution-baseline.json; kết quả adapter mới ở .runtime/step1-prediction.json. Lần lấy mốc đo 8,149 giây trên CPU; đây là một mẫu đo, chưa phải median hoặc p95.

Đã chạy 5 test Python adapter, 5 test Node lưu nhất quán trên database tạm và đối chiếu 25 SHA-256 gốc thành công. Các test Node dùng model giả để gây lỗi có kiểm soát, kiểm tra replay, xung đột nội dung, trùng đồng thời, lỗi lưu Diagnosis, lỗi socket và phục hồi startup. Lệnh bổ sung:

```powershell
Push-Location Backend
try {
    node --test tests/upload-consistency.test.cjs
    node --test tests/upload-api.test.cjs
} finally { Pop-Location }
```

Test upload-api chạy một gateway riêng cổng 13006 và database Mongo tạm, gọi Hybrid thật ở 8000. Script tự dọn database và tệp ca thử. Phải chờ AI health sẵn sàng trước khi chạy. Đã chạy đạt, bằng chứng ở .runtime/upload-api-results.log. Test này kiểm tra đăng nhập, tạo bệnh nhân, upload qua HTTP, replay, xung đột và đọc lại kết quả. API đọc chẩn đoán đã thêm lọc scan done trước khi đếm/phân trang và đọc chi tiết. Các đầu việc DICOM, phiếu, FAISS, thống kê và đo tải vẫn chờ thực hiện.

Lượt build cuối và 5 test nhất quán sau sửa API đọc chẩn đoán đều đạt. Lệnh restart gateway chính bị bộ kiểm duyệt tự động từ chối ở bước tạo process PowerShell. Cổng 3006 vì vậy có thể vẫn dùng mã đã nạp trước lượt build cuối; cần xác minh và nạp bản mới trong lần triển khai kế tiếp. Gateway thử 13006 đã đóng sau kiểm thử. Chưa có phép kiểm tra giao diện cho metadata mới.

### Bước 2 đã triển khai và kiểm tra bổ sung

Đã kiểm tra lại PID, dừng đúng các process gateway/AI quan sát được và khởi động lại bằng start-web.ps1 thành công. Trạng thái chặn restart ghi ở trên thuộc lần trước; bản có bộ đọc DICOM hiện đã chạy ở 8000 và gateway có route preview ở 3006. Gateway thử của test dùng build mới nhất và database riêng. Cần restart gateway chính khi có thay đổi backend tiếp theo.

- Thêm image_input.py với pydicom 3.0.1. Nhận DICOM đơn khung grayscale không nén (Implicit/Explicit Little Endian, Explicit Big Endian), xử lý modality LUT/rescale trước VOI/window, chiều sáng và padding. Giới hạn 20 triệu pixel, từ chối ảnh hằng hoặc dữ liệu hỏng. PNG/JPEG đi vào pipeline bằng tệp gốc.
- Adapter dùng cùng pixel DICOM đã giải mã cho CLIP và Hybrid. Trả previewPng qua kênh nội bộ. Node ghi ảnh PNG riêng, không lưu preview base64 vào Mongo; giữ hybridAnalysis gốc. Tệp upload mới nằm trong private-uploads; chỉ bản pixel xem lại được cung cấp qua GET /api/v1/scans/:scanId/preview có đăng nhập. Tệp DICOM nguồn không có route static. Đã bổ sung previewPath vào Scan, ghi đường dẫn khi còn processing để phục hồi được sau restart.
- Cả AddPatientModal và PatientProfilePage nhận .dcm/.dicom kể cả MIME chung, hiển thị tệp đã chọn trong lúc chờ giải mã. Dashboard và ảnh so sánh dùng ảnh blob lấy với token, hủy request và thu hồi object URL khi đổi scan. Ảnh cũ ở /uploads vẫn đọc được.
- Kiểm tra đạt: build Backend, build Frontend, 10 test Python (adapter + decoder), 5 test Node nhất quán và 2 test HTTP dùng Hybrid thật. Test HTTP đã kiểm tra PNG, DICOM với MIME application/octet-stream, 15 xác suất, ảnh preview khớp PNG chuẩn hóa sau đọc lại hồ sơ, truy cập preview thiếu đăng nhập trả 401, tệp nguồn qua static trả 404. Bằng chứng ở .runtime/upload-api-results.log.
- Kiểm tra UI đạt đường thêm bệnh nhân → chọn DICOM → Hybrid → hồ sơ → tải lại trang. Ảnh xem lại có naturalWidth 1060, tải bằng blob và hiển thị được trên mobile. Ảnh minh chứng .runtime/dicom-ui.jpg. Bệnh nhân thử giữ tạm cho các bước phiếu và đổi nhiều scan tiếp theo: patientCode PT-0A98CA, tên UI TEST DICOM Synthetic. Chỉ dọn đúng hồ sơ thử này khi chốt nghiệm thu.
- DICOM dùng trong kiểm thử toàn luồng được tạo từ pixel ảnh mẫu X-quang Results/EX/Cardiomegally.png và metadata giả, lưu tại .runtime/ui-synthetic-chest.dcm. Chưa có DICOM thu nhận trực tiếp từ máy chụp hoặc các transfer syntax nén để kiểm chứng. Không ghi A2 đạt toàn bộ. Chưa kiểm tra thêm lần chụp thứ hai qua UI của PatientProfilePage.

Việc tiếp theo: bước 3 trình xem chi tiết và health thật, bước 4 phiếu và xác minh có xử lý xung đột, bước 5 FAISS ghi an toàn, bước 6 thống kê/loại dữ liệu mặc định, sau đó đo tải và hoàn tất A1–A8. UI còn hiển thị vitals mặc định cho hồ sơ chưa nhập; xử lý ở bước loại dữ liệu mặc định trước nghiệm thu.

### Bước 3 và phần xác minh của bước 4

- Thêm GET /api/v1/system/ai-health có xác thực, đọc trạng thái từ FastAPI và đo latency của chính lời gọi health. Trang hệ thống và badge ở Dashboard/Profile lấy dữ liệu thật qua RTK Query. Trang hệ thống đã thay phần slider lưu localStorage, số đo mặc định và nút restart mô phỏng bằng trạng thái model, thiết bị, validator và hàng chờ. Các endpoint cấu hình client cũ còn tồn tại để giữ tương thích caller cũ.
- FastAPI dùng InferenceRunner trong runtime.py: một worker suy luận, tối đa ba yêu cầu đang chạy/chờ, timeout hàng chờ 30 giây. Phần CPU chạy trong thread, event loop vẫn phục vụ health. Quá tải trả 503. Kiểm tra bằng model thật xác nhận health phản hồi lúc active = 1. Đối chiếu 15 xác suất sau chuyển sang worker với baseline đều đạt sai số 1e-5; bằng chứng .runtime/worker-prediction.json.
- HybridAnalysisPanel có chọn ảnh gốc chuẩn hóa hoặc attention/fusion, hiển thị tin cậy hiệu chỉnh, độ bất định, mức độ từ báo cáo gốc, lớp VinDr và tọa độ khung khi có. YOLO chưa có trọng số nên phần khung thật vẫn chờ kiểm chứng. Attention_quality là object thông số trong hợp đồng TypeScript, đã đối chiếu response thật.
- Backend xác minh dùng revision tăng nguyên tử. FE gửi phiên bản được giữ lúc mở editor, tránh tự nhận phiên bản mới khi refetch trong lúc đang sửa. Notes tối đa 20.000 ký tự, lưu verifiedByName do server tra từ tài khoản. Nếu scan chưa done thì không thể xác minh. API cũ thiếu expectedRevision vẫn được hỗ trợ.
- Thêm DiagnosisReport dùng chung cho Profile và Dashboard. Hiển thị đúng mã scan, mô tả AI, kết luận AI, khuyến nghị, kết luận bác sĩ và người xác minh. Nút print của Profile và thao tác xác minh Dashboard mở phiếu này. Phiếu có thể in qua window.print và portal CSS chỉ chứa lần chụp đang chọn, khóa nút in khi đang sửa chưa lưu. Chưa có bằng chứng file PDF xuất thành công: thao tác print trong IAB báo timeout CDP, DOM vẫn hoạt động và phiếu còn mở. Cần hoàn tất/kiểm chứng xuất PDF ổn định trước khi ghi A5 đạt toàn bộ.
- Kiểm tra đạt: Backend build, Frontend build, 12 test Python, 6 test Node nhất quán, 4 test HTTP có Hybrid thật. HTTP kiểm tra hai lần chụp được xác minh riêng, stale revision trả 409, role user bị chặn 403, AI health trong lúc inference. Hệ thống thử riêng tự dọn dữ liệu. UI đã mở phiếu PT-0A98CA, lưu kết luận có nhãn KIỂM THỬ GIAO DIỆN, mở lại thấy Đã xác minh, phiên bản 1 và tên người cập nhật. Minh chứng .runtime/doctor-report-ui.jpg. Dữ liệu thử này vẫn được giữ cho các bước tiếp.

Việc tiếp theo ưu tiên: hoàn thiện xuất PDF có thể kiểm chứng, thêm lần chụp thứ hai qua UI và thử bản in theo scan, rồi triển khai FAISS ghi theo thế hệ và thống kê đúng nhãn. Cần kiểm tra thêm A3 ngắt tiến trình thật, A6 phục hồi FAISS, A8 đo RAM và median/p95. Mục tiêu tổng thể vẫn active.

### Xuất PDF và nối ca tham chiếu

Người dùng đã làm rõ rằng ảnh sẽ do người dùng web tải lên, hiện không có thêm trọng số YOLO, DICOM máy chụp hay bộ tham chiếu có nhãn để cung cấp. Phạm vi triển khai thực tế là nhận ảnh hợp lệ qua web và tăng dần CBIR từ ca bác sĩ xác nhận. Dùng fixture đúng định dạng và ảnh mẫu để kiểm tra phần mềm. Kho trống là trạng thái hợp lệ trước ca tham chiếu đầu tiên. Ghi rõ giới hạn kiểm chứng dữ liệu thực và trạng thái nhánh YOLO trong bàn giao.

- Thêm GET /api/v1/diagnoses/:id/report.pdf?revision=n. Gateway đọc kết quả đã lưu, kiểm tra phiên bản rồi gọi script report_pdf.py độc lập với model. Dùng reportlab 4.5.1, nhúng font tiếng Việt Arial trên Windows hoặc DejaVu Sans trên Linux; có thể cấu hình REPORT_FONT_PATH và PYTHON_EXECUTABLE. Nút Tải PDF dùng bearer token, tải đúng phiên bản đã lưu, tách khỏi nút In phiếu của trình duyệt.
- Đã tải PDF qua UI của PT-0A98CA vào E:/Downloads, sao chép bằng chứng tại .runtime/doctor-report-sample.pdf. Đã render và xem trang .runtime/report-page-1.png, tiếng Việt và kết luận đúng, bố cục một trang không bị cắt. Hai test PDF kiểm tra Unicode, escape markup và kết luận dài qua nhiều trang. HTTP test kiểm tra riêng hai scan và từ chối revision cũ. Cài dependency kiểm tra bằng Backend/tests/requirements.txt.
- Thêm ReferenceStore bọc SimilarCaseFinder gốc. Mỗi lần thêm dùng bản sao index/metadata, gọi add_case gốc, fsync cặp file rồi công bố manifest nguyên tử. Reader giữ snapshot, khôi phục thế hệ trước khi cặp hiện hành hỏng. Chặn writer thứ hai bằng khóa OS. Giữ hai thế hệ để giới hạn dung lượng. Ca ref_* gửi lại cùng nhãn chỉ có một vector, khác nhãn bị từ chối.
- FastAPI nạp store ở Backend/data/reference-generations, có thể cấu hình CBIR_STORE_ROOT. POST /v3/reference-cases và đường legacy thêm ca đều dùng store và worker chung. GET /v3/reference-cases/:id hỗ trợ đối soát. /predict legacy cũng dùng worker chung để tránh chạy model đồng thời với thao tác thêm ca. Báo trạng thái/count CBIR trong health.
- Node POST /diagnoses/:id/reference-case kiểm tra phiếu verified, khóa nhãn và mã ref_ theo diagnosisId, lưu pending/ready trong Diagnosis. Retry sau mất phản hồi Python đối soát bằng mã idempotent của store. Bổ sung người lưu và phiên bản phiếu tại thời điểm lưu cho các yêu cầu mới. UI chọn nhãn xác nhận và khóa thao tác khi phiếu còn sửa chưa lưu.
- Đã kiểm tra UI lưu ca PT-0A98CA với nhãn thử Effusion. Health trả CBIR available=true, count=1. Đây là ca kiểm thử có kết luận gắn nhãn KIỂM THỬ, được giữ cho bước tra cứu tiếp theo. Minh chứng .runtime/reference-saved-ui.jpg. Dữ liệu FAISS chứa ref_6aeeef685bdfa21f4002df93, đã đọc từ metadata của thế hệ công bố. Khi dọn, đối chiếu lại referenceCase trong DB và manifest hiện hành.
- Thêm GET /diagnoses/:id/similar-cases để nối mã FAISS với scan và patientCode nếu có ánh xạ; ca legacy không có ánh xạ vẫn hiển thị. Link mở profile có ?scan=...; thêm GET /scans/detail/:scanId và hook để đọc đúng scan ngoài 10 phim gần nhất. Hai scan đối chiếu phải khác ID. Phần click link tới ca trả về từ một truy vấn mới vẫn cần kiểm thử toàn luồng.
- Build FE/BE và typecheck đã qua trước bổ sung các trường audit cuối. Test Python đã có 18 ca qua trước test bổ sung, và 5 test ReferenceStore mới nhất đều qua, gồm concurrent replay, lỗi trước publish, corruption fallback và giới hạn hai thế hệ. Chưa kiểm tra đầy đủ Node pending/ready khi mất phản hồi hoặc restart toàn service sau lưu bằng model thật.

Việc tiếp theo: build/typecheck lại sau các trường audit, kiểm thử các API ca tham chiếu (quyền, conflict, retry) bằng database riêng, thực hiện truy vấn qua một ảnh upload mới và mở đúng scan tham chiếu sau restart. Sau đó bước 6 sửa thống kê và vitals mặc định, tối ưu payload, chạy A1–A8 và ghi số đo. Mục tiêu tổng thể vẫn active.
