# 🚀 Hướng dẫn Build FAISS Index trên Google Colab

Bạn hoàn toàn có thể mang việc nặng nhọc này lên Google Colab để làm! Môi trường Linux trên Colab hỗ trợ đa luồng (multiprocessing) của PyTorch cực kỳ hoàn hảo (`num_workers=8` thoải mái không bị lỗi) và có sẵn GPU T4/L4 cực mạnh.

Tuy nhiên, rào cản duy nhất là **dữ liệu ảnh (Dataset 140.000 tấm)**. Bạn sẽ không cần upload 50GB ảnh từ máy tính lên mạng (việc này mất hàng tuần). Giải pháp tối ưu là:

## Bước 1: Chuẩn bị file lên Google Drive
Tạo một thư mục trên Google Drive của bạn (ví dụ: `CheXNet_Project`) và upload 4 thứ sau lên đó:
1. `Trainedmodel/hybrid_model_light.pth` (Checkpoint model)
2. `Backend/cbir.py` (Script tính toán)
3. `Models/Model.py` (File định nghĩa mô hình)
4. `Dataset/train_list.csv` (File danh sách bệnh)

## Bước 2: Khởi tạo Notebook trên Google Colab
1. Vào [Google Colab](https://colab.research.google.com/) tạo sổ tay mới (New Notebook).
2. Vào **Runtime** > **Change runtime type** > Chọn **T4 GPU**.

## Bước 3: Chạy các Cell sau trên Colab

### Cell 1: Kết nối Google Drive và cài đặt thư viện
```python
from google.colab import drive
drive.mount('/content/drive')

!pip install faiss-gpu faiss-cpu pandas tqdm Pillow
```

### Cell 2: Tải Dataset siêu tốc từ Kaggle (Khuyên dùng)
*Thay vì upload ảnh từ máy bạn, hãy tải trực tiếp từ Kaggle về Colab (chỉ mất 2-3 phút cho 40GB ảnh).*
```bash
# Cấu hình Kaggle API (bạn cần file kaggle.json tải từ tài khoản Kaggle của bạn)
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/CheXNet_Project/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Đổi link dưới thành bộ dataset chính xác bạn đang dùng (NIH hoặc VinDr)
!kaggle datasets download -d nih-chest-xrays/data -p /content/dataset --unzip
```

### Cell 3: Chạy script Build Index với sức mạnh tối đa
```python
import sys
import os

# Thêm thư mục code vào path để import được Model
project_root = '/content/drive/MyDrive/CheXNet_Project'
sys.path.append(project_root)

from cbir import SimilarCaseFinder
from Models.Model import HybridCNNViTModel # Đảm bảo file Model.py nằm trong thư mục Models trên Drive
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Khởi tạo và Load Model
model = HybridCNNViTModel(num_classes=15)
checkpoint_path = os.path.join(project_root, 'hybrid_model_light.pth')

checkpoint = torch.load(checkpoint_path, map_location=device)
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

# 2. Xây dựng Index (Chạy siêu nhanh với num_workers=8)
SimilarCaseFinder.build_index(
    model=model,
    dataset_csv_path=os.path.join(project_root, 'train_list.csv'),
    image_root='/content/dataset', # Trỏ tới thư mục vừa tải từ Kaggle
    output_dir=os.path.join(project_root, 'output_faiss'),
    batch_size=128, # Colab T4 GPU (16GB VRAM) cân thoải mái batch 128
    device=device
)
```

## Bước 4: Tải kết quả về máy tính
Sau khi tiến trình trên chạy xong 100%, trên Google Drive của bạn (trong thư mục `CheXNet_Project/output_faiss`) sẽ xuất hiện 2 file:
- `chest_xray.faiss`
- `chest_xray_meta.json`

Việc của bạn chỉ là **tải 2 file đó về** và chép đè vào thư mục `D:\temp\tempnckh\CheXNet\Backend\data\` trên máy tính cục bộ. Sau đó mở Backend API lên và tận hưởng!
