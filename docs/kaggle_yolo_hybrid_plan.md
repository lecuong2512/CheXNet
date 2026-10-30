# 🛠 Kế Hoạch Chuẩn Bị Dữ Liệu Hybrid YOLO Trên Kaggle

Mục tiêu: Xây dựng tập dữ liệu YOLO gồm 14 classes kết hợp:
- **VinDr-CXR**: Lấy BBox chuẩn từ người (10 bệnh).
- **NIH**: Dùng CheXNet sinh BBox ảo (Pseudo-labels) cho 4 bệnh còn thiếu.
- **Xử lý đặc thù Kaggle**: Dữ liệu ảnh nằm rải rác trong 10 folder `data_1` -> `data_10`.

---

## 1. Bản Đồ Class IDs (14 Classes)

Để model YOLO mapping chuẩn 1-1 với thứ tự của CheXNet:
```python
# CHEXNET_CLASSES (Bỏ 'No Finding')
TARGET_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',    # 0-4 (VinDr)
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',      # 5-9 (Pneumonia, Edema từ NIH)
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'               # 10-13 (Emphysema, Hernia từ NIH)
]
```

## 2. Xây Dựng Thuật Toán Tối Ưu (Kaggle Data Prep)

### A. Giải thuật tra cứu đường dẫn ảnh O(1)
Thay vì lặp qua 130,000 dòng CSV rồi dùng `os.path.find` trong 10 thư mục (cực kỳ chậm, làm sập IO Kaggle), ta quét 1 lần duy nhất:

```python
# Khởi tạo bản đồ đường dẫn O(1)
image_path_map = {}
data_dirs = glob.glob("/kaggle/input/chexnet-hybrid-dataset/data_*") # 10 thư mục

for d in data_dirs:
    for img_name in os.listdir(d):
        if img_name.endswith(('.png', '.jpg')):
            image_path_map[img_name] = os.path.join(d, img_name)

# Tra cứu siêu tốc: 
# full_path = image_path_map.get("00015679_015.png")
```

### B. Tách nhánh xử lý trong Vòng lặp CSV
Khi duyệt qua từng dòng của `train_list.csv` / `val_list.csv`:
- Nếu `Source == 'VinDr-CXR'`: Đọc `x_min, y_min, x_max, y_max`, tính công thức chuẩn hóa YOLO. Ghi vào file `.txt`.
- Nếu `Source == 'NIH'` **VÀ** có nhãn = 1 của (`Pneumonia`, `Edema`, `Emphysema`, `Hernia`):
  - Gửi đường dẫn ảnh vào một mảng `nih_pseudo_queue`.

### C. Sinh Pseudo-Labels bằng CheXNet (Batch Processing)
Sau khi duyệt xong CSV, ta có mảng `nih_pseudo_queue`.
- Load mô hình `HybridCNNViTModel` lên GPU.
- Tạo DataLoader chạy batch size 32 hoặc 64.
- Lấy `attention_maps`.
- Chạy hàm biến Heatmap thành BBox (Otsu threshold).
- Trích xuất tọa độ, ghi file `.txt` YOLO cho 4 classes này.

## 3. Cấu Trúc Khối Code Trong Kaggle Notebook

Bạn sẽ chỉnh sửa file `yolov11m_fixed.ipynb` trên Kaggle thành các Cell (khối) như sau:

**Cell 1: Cài đặt và Import**
```bash
!pip install ultralytics timm scikit-learn
```

**Cell 2: Định nghĩa Script Xử lý Dữ liệu**
Tạo một file `prepare_kaggle_data.py` (tôi sẽ viết code hoàn chỉnh này). Script này chứa toàn bộ logic phần 2 ở trên.

**Cell 3: Chạy Script Tiền xử lý (Khoảng 20-40 phút)**
```bash
!python prepare_kaggle_data.py \
    --csv_dir /kaggle/input/chexnet-csv-data \
    --img_dirs /kaggle/input/chexnet-image-data/data_* \
    --model_weight /kaggle/input/chexnet-model/hybrid_model_light.pth \
    --output_dir /kaggle/working/yolo_dataset
```

**Cell 4: Khởi chạy YOLO11m Training**
```python
from ultralytics import YOLO

# Khởi tạo mô hình pre-train
model = YOLO('yolov11m.pt')

# Train bằng bộ dataset hybrid vừa sinh ra
results = model.train(
    data='/kaggle/working/yolo_dataset/dataset.yaml',
    epochs=50,
    imgsz=1024,
    batch=8,           # Kaggle P100/T4 chịu được batch 8 với ảnh 1024
    device=0,
    project='CheXNet_YOLO_Hybrid',
    name='train_phase1'
)
```

---

## 4. Các Vấn Đề Cần Lưu Ý Trên Kaggle

1. **Dung lượng Disk (/kaggle/working/)**: Kaggle chỉ cho phép tối đa ~20GB ở thư mục working. Các file `.txt` của YOLO rất nhẹ (vài MB), nhưng nếu bạn copy cả ảnh vào đây sẽ bị đầy ổ cứng!
   - **Giải pháp**: Trong file `.txt` YOLO của tập train (vd: `train.txt`), ta sẽ ghi thẳng đường dẫn TỐI THƯỢNG (Absolute path) đến các folder `/kaggle/input/.../data_x/`. YOLO Ultralytics hỗ trợ đọc đường dẫn từ file danh sách mà không cần copy ảnh về chung 1 thư mục!

2. **Giới hạn 12 Tiếng**: Việc train YOLO 50 epochs ảnh 1024x1024 có thể tốn hơn 12 tiếng. Hãy đảm bảo bạn sử dụng cờ `resume=True` nếu cần train nối tiếp.
