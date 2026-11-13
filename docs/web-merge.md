# Hợp nhất FE/BE với pipeline Hybrid

Nguồn giao diện và nghiệp vụ: `D:\Projects\Science_Of_AI\CheXNet`, commit `41e5c84`.
Đích: `D:\NCKH\CheXNet`. Kế thừa React 19/TypeScript, Node.js/Express,
đăng nhập/phân quyền, quản lý bệnh nhân, scan, chẩn đoán, nghiên cứu và cấu hình.
Bản nguồn có thay đổi cục bộ ở `Backend/src/common/config/redis.ts`; nội dung
này được kế thừa. Thư mục nguồn không bị chỉnh sửa bởi lần hợp nhất này.

## Giữ nguyên kỹ thuật gốc

Không thay thế `HybridCNNViTModel` bằng DenseNet hoặc ConvNeXt độc lập của
bản nguồn. Giữ nguyên byte của 25 file thuộc `Models/`, `Finetune/`, `Dataset/`,
các script Python gốc và tám file phục vụ suy luận gốc trong `Backend/`.
Bao gồm tiền xử lý, attention/FPN, thresholds, YOLO/cascade/display fusion,
phân vùng phổi, báo cáo, tương quan và CBIR. Không thay trọng số mô hình.

`Backend/ai_service/main.py` dùng lại chính app `Backend.main` và thêm:

- `/health`: trạng thái mô hình và bộ xác thực ảnh CLIP kế thừa từ bản nguồn.
- `/v3/predict`: xác thực ảnh, gọi pipeline gốc và chuyển 15 xác suất sang
  hợp đồng `className/probability` mà API Node yêu cầu.
- `hybridAnalysis`: giữ nguyên toàn bộ phản hồi gốc, được lưu trong chẩn đoán
  MongoDB và hiển thị trên Dashboard/hồ sơ bệnh nhân.

`/predict` và `/api/add_reference_case` gốc tiếp tục hoạt động.
Demo FE cũ được giữ tại `/legacy.html`, bao gồm công cụ CBIR và báo cáo cũ.
Giao diện mới dùng ảnh attention thật, không dùng bounding box mô phỏng.
Ngưỡng hiển thị nhãn Hybrid lấy từ kết quả pipeline gốc.

## Chạy trên máy hiện tại

```powershell
cd D:\NCKH\CheXNet
.\start-web.ps1
```

- FE: <http://127.0.0.1:5173/>
- API nghiệp vụ: <http://127.0.0.1:3006/docs>
- AI Hybrid: <http://127.0.0.1:8000/health>
- Demo Hybrid cũ: <http://127.0.0.1:5173/legacy.html>

Database riêng: `chexnet_hybrid_local`. Không dùng database của thư mục nguồn.
MongoDB phải đang chạy. Redis là tùy chọn; API tiếp tục hoạt động khi không có
Redis. Không tự xóa bệnh án khi khởi động.

Tài khoản phát triển kế thừa: `admin@chexnet.vn` / `Admin@123456`,
hoặc `bacsi@chexnet.vn` / `Doctor@123456`.
Secrets cục bộ được sinh riêng trong `Backend/.env` và không đưa vào Git.
FE cấu hình API bằng `Frontend/.env.local`; có file `.env.example` tương ứng.

## Cài lại thư viện và build

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r Backend/requirements.txt -r Backend/ai_service/requirements.txt
cd Backend
yarn install --frozen-lockfile
yarn build
cd ..\Frontend
yarn install --frozen-lockfile
yarn build
```

Máy hiện tại dùng `.venv` có `include-system-site-packages=true` để tái sử dụng
PyTorch CPU đã cài. Hai thư mục Node dependencies được sao chép từ bản nguồn
cùng máy; lockfile Yarn được giữ nguyên. FE build bao gồm cả trang mới và legacy.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Backend/tests -v
```

Đã kiểm tra build FE/BE, 4 test hợp đồng adapter, đối chiếu SHA-256 của 25 file
gốc và toàn luồng đăng nhập → bệnh nhân → upload → Hybrid → lưu/đọc MongoDB →
xác minh. Kiểm tra ảnh không phải X-quang bị từ chối HTTP 422 và không tạo
scan/chẩn đoán. Kiểm tra API người dùng và nghiên cứu.
Đã dọn hồ sơ, scan và ảnh upload được tạo riêng cho kiểm thử tích hợp.
Trạng thái chưa có ảnh không được hiển thị như kết luận phổi bình thường.

Hiện chạy CPU với `Trainedmodel/hybrid_model.pth`. Chưa có
`Trainedmodel/yolov11m.pt`, nên YOLO báo chưa nạp; mã và cơ chế YOLO vẫn được
giữ nguyên. CBIR hiện chưa có ca tham chiếu.

Bản sao trước hợp nhất và manifest SHA-256 nằm trong
`.runtime/pre-web-merge/` trên máy hiện tại. Log chạy nằm trong `.runtime/`.
