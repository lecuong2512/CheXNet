# 🫁 CheXNet V3 — Hướng Dẫn Cài Đặt & Chạy Dự Án

> Hệ thống chẩn đoán ảnh X-quang phổi sử dụng Deep Learning (DenseNet-121 / ConvNeXtV2), tích hợp đầy đủ Frontend, Backend API, AI Inference Service và MongoDB.

---

## 📁 Cấu Trúc Dự Án

```text
CheXNet/
├── Frontend/           # React + Vite (Giao diện bác sĩ)
├── Backend/            # Node.js + Express (API Gateway)
│   ├── ai_service/     # FastAPI Python (AI Inference Service)
│   ├── src/            # TypeScript source
│   └── uploads/        # Ảnh X-quang đã upload
├── Models/             # Python model code (Training & Evaluation)
├── Trainedmodel/       # Chứa file trọng số chexnetmodel.pth
├── Dataset/            # Danh sách train/val/test
├── Database/           # Script tải dữ liệu NIH
└── main.py             # Entry point huấn luyện / kiểm thử
```

---

## ⚙️ Yêu Cầu Hệ Thống

| Thành phần | Phiên bản tối thiểu |
|---|---|
| **Python** | 3.10+ (khuyến nghị 3.13) |
| **Node.js** | 20+ |
| **Yarn** | 1.22+ |
| **MongoDB** | 6.0+ (standalone, không cần Docker) |
| **Redis** | 7.0+ (standalone) |
| **GPU CUDA** | Tùy chọn (CPU cũng chạy được) |

---

## 🚀 Hướng Dẫn Chạy Dự Án (Native Windows — Không Dùng Docker)

Dự án chạy **4 tiến trình song song** trên 4 terminal riêng biệt.

---

### Bước 0: Cài Đặt MongoDB & Redis (Chỉ làm lần đầu)

**MongoDB:**
1. Tải về tại: https://www.mongodb.com/try/download/community
2. Cài đặt dạng Windows Service → MongoDB sẽ tự khởi động cùng Windows
3. Mặc định chạy ở cổng `27017`

**Redis:**
1. Tải về tại: https://github.com/tporadowski/redis/releases (bản Windows)
2. Cài đặt, chạy dưới dạng Service
3. Mặc định chạy ở cổng `6379`

Kiểm tra:
```powershell
# Kiểm tra MongoDB
mongosh --eval "db.adminCommand('ping')"

# Kiểm tra Redis
redis-cli ping   # Kết quả: PONG
```

---

### Bước 1: Cài Đặt Dependencies (Chỉ làm lần đầu)

**Python AI dependencies:**
```powershell
# Từ thư mục gốc CheXNet/
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
py -m pip install fastapi uvicorn python-multipart timm Pillow numpy
```

> 💡 Nếu có GPU NVIDIA, thay `cpu` bằng `cu124` (CUDA 12.4) để tận dụng GPU tăng tốc.

**Node.js Backend dependencies:**
```powershell
cd Backend
yarn install
```

**Frontend dependencies:**
```powershell
cd Frontend
npm install
```

---

### Bước 2: Cấu Hình Biến Môi Trường

File `Backend/.env` (đã có sẵn, kiểm tra lại nếu cần):

```env
PORT=3005
NODE_ENV=development
MONGODB_URI=mongodb://localhost:27017/chexnet_v3
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=
JWT_PRIVATE_KEY=chexnet-v3-super-secret-jwt-key-2024
JWT_EXPIRES_IN=15m
JWT_REFRESH_EXPIRES_IN=7d
AI_SERVICE_URL=http://localhost:8000
UPLOAD_DIR=./uploads
MAX_FILE_SIZE_MB=10
ALLOWED_ORIGINS=http://localhost:5173
LOG_LEVEL=debug
```

---

### Bước 3: Chạy AI Inference Service (Terminal 1)

```powershell
cd Backend\ai_service
py main.py
```

**Kết quả mong đợi:**
```
[AI Model] Sử dụng thiết bị: cpu
[AI Model] Đang tải mô hình từ: ...\Trainedmodel\chexnetmodel.pth
[AI Model] 🔍 Phát hiện kiến trúc: DenseNet-121 (Classical CheXNet)
[AI Model] ✅ Tải mô hình thành công! (Image Size: 224x224)
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Kiểm tra dịch vụ AI hoạt động:
```powershell
curl http://localhost:8000/health
# {"status":"healthy","modelLoaded":true,"device":"cpu","service":"CheXNet AI Inference Service"}
```

---

### Bước 4: Chạy Backend API Gateway (Terminal 2)

```powershell
cd Backend
yarn dev
```

**Kết quả mong đợi:**
```
✅ MongoDB kết nối thành công: mongodb://localhost:27017/chexnet_v3
✅ Redis kết nối thành công: 127.0.0.1:6379
✅ Socket.IO đã khởi động
✅ Cron jobs đã khởi động
[Seed] ✅ Hoàn thành khởi tạo tài khoản hệ thống
🚀 Máy chủ CheXNet V3 đang chạy tại cổng http://localhost:3005
```

Kiểm tra backend API:
```powershell
curl http://localhost:3005/health
```

---

### Bước 5: Chạy Frontend Client (Terminal 3)

```powershell
cd Frontend
npm run dev
```

**Kết quả mong đợi:**
```
  VITE v8.0.13  ready in 470 ms
  ➜  Local:   http://localhost:5173/
```

---

### Bước 6: Truy Cập Ứng Dụng

Mở trình duyệt và truy cập: **http://localhost:5173**

**Tài khoản mặc định (được seed tự động):**

| Vai trò | Email | Mật khẩu |
|---|---|---|
| **Bác sĩ** | `bacsi@chexnet.vn` | `Doctor@123456` |
| **Admin** | `admin@chexnet.vn` | `Admin@123456` |

> ⚠️ Chỉ tài khoản **Bác sĩ** có thể sử dụng tính năng thêm bệnh nhân và quét ảnh AI.

---

## 🏥 Hướng Dẫn Sử Dụng Tính Năng Quét AI

### 1. Đăng nhập với tài khoản Bác sĩ

Truy cập `http://localhost:5173` → Nhập email `bacsi@chexnet.vn` / mật khẩu `Doctor@123456`.

### 2. Thêm bệnh nhân mới

Click nút **"+ New Analysis"** ở Sidebar (hoặc nút **"+ Thêm ca bệnh"** ở Dashboard).

**Bước 1 — Thông tin bệnh nhân:**
- Nhập tên, tuổi, giới tính, khoa chẩn đoán
- (Tùy chọn) Mở rộng phần **Sinh hiệu lâm sàng** để nhập nhịp tim, huyết áp, SpO2

**Bước 2 — Tải ảnh X-quang:**
- Chọn loại chụp: PA, AP, Lateral, hoặc CT Scan
- Kéo thả hoặc chọn file ảnh ngực định dạng JPG/PNG
- Click **"Bắt đầu phân tích AI"**

**Bước 3 — Chờ AI chẩn đoán:**
- Hệ thống gửi ảnh đến DenseNet-121 CheXNet model
- Nhận kết quả xác suất 15 nhãn bệnh lý trong vài giây
- Tự động chuyển hướng đến trang hồ sơ bệnh nhân để xem kết quả

### 3. Xem kết quả chẩn đoán

Kết quả bao gồm xác suất % của 15 bệnh lý:
`No Finding`, `Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Hernia`

---

## 🧪 Kiểm Thử Suy Luận Trực Tiếp (Không Qua UI)

Chạy script Python để test model trực tiếp trên một ảnh:

```python
# test_single.py
import sys
sys.path.append(r".\Backend\ai_service")

from inference import CheXNetInference

infer = CheXNetInference()  # Tự động tải chexnetmodel.pth
predictions = infer.predict(r".\Backend\uploads\<tên_file_ảnh>.jpg")

for pred in predictions:
    print(f"  {pred['className']}: {pred['probability'] * 100:.2f}%")
```

```powershell
py test_single.py
```

---

## 🎓 Huấn Luyện / Kiểm Thử Mô Hình (Tuỳ Chọn Nâng Cao)

### Chuẩn bị dữ liệu NIH Chest X-Ray

```powershell
cd Database
py batch_download_zips.py
```

> Tải ~112,000 ảnh X-quang từ NIH (~40GB). Cần kết nối mạng ổn định.

### Huấn luyện từ đầu

```powershell
py main.py
# Chọn mode: train
```

### Đánh giá mô hình đã huấn luyện

```powershell
py main.py
# Chọn mode: test
# Nhập đường dẫn dữ liệu và model path
```

### Tiếp tục từ checkpoint

```powershell
py main.py
# Chọn mode: resume
# Nhập đường dẫn checkpoint: Trainedmodel/chexnetmodel.pth
```

---

## 🌐 Bảng Tổng Hợp Cổng Dịch Vụ

| Dịch vụ | URL | Mô tả |
|---|---|---|
| **Frontend** | http://localhost:5173 | Giao diện lâm sàng Bác sĩ |
| **Backend API** | http://localhost:3005 | REST API Gateway |
| **AI Service** | http://localhost:8000 | FastAPI DenseNet-121 |
| **API Docs** | http://localhost:3005/docs | Swagger UI tài liệu API |
| **MongoDB** | localhost:27017 | Database `chexnet_v3` |
| **Redis** | localhost:6379 | Cache & Session Store |

---

## 🗄️ Xem Dữ Liệu Trong MongoDB

Kết nối bằng **MongoDB Compass**: `mongodb://localhost:27017`

Các collection quan trọng trong database `chexnet_v3`:

| Collection | Mô tả |
|---|---|
| `users` | Tài khoản bác sĩ / admin |
| `patients` | Hồ sơ bệnh nhân |
| `scans` | Bản ghi ảnh X-quang đã upload |
| `diagnoses` | Kết quả chẩn đoán AI (15 nhãn xác suất) |

---

## 🔧 Xử Lý Sự Cố Thường Gặp

### ❌ Lỗi "AI service không khả dụng"
→ AI Inference Service chưa chạy. Mở terminal và chạy:
```powershell
cd Backend\ai_service
py main.py
```

### ❌ Lỗi "MongoDB connection failed"
→ MongoDB Service chưa khởi động:
```powershell
net start MongoDB
```

### ❌ Cổng 5173 đã bị chiếm
→ Tìm và đóng tiến trình chiếm cổng:
```powershell
netstat -aon | findstr 5173
taskkill /F /PID <PID_TÌM_THẤY>
```

### ❌ Lỗi Python modules not found
→ Cài lại đầy đủ thư viện:
```powershell
py -m pip install -r Backend\ai_service\requirements.txt
```

### ❌ Build Frontend lỗi TypeScript
→ Kiểm tra lỗi trước khi build:
```powershell
cd Frontend
npm run lint
npm run build
```

---

## 📚 Tài Liệu Tham Khảo

- 📄 [CheXNet Paper (Stanford)](https://arxiv.org/abs/1711.05225)
- 🗂️ [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- 🔥 [PyTorch Documentation](https://pytorch.org/docs/stable/)
- ⚡ [FastAPI Documentation](https://fastapi.tiangolo.com/)
- 🍃 [Mongoose ODM](https://mongoosejs.com/)
