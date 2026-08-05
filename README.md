# CheXNet — Hybrid CNN-ViT

Dự án Deep Learning nhận diện 15 loại bệnh phổi từ ảnh X-ray, sử dụng kiến trúc lai ConvNeXtV2 + SwinV2 với Residual Masking và huấn luyện 2 giai đoạn (NIH Chest X-ray + VinDr-CXR).

## Mục tiêu
- Huấn luyện và đánh giá mô hình phân loại các bệnh phổi từ ảnh X-ray.
- Tái hiện lại kết quả nghiên cứu CheXNet với tập dữ liệu NIH Chest X-ray.

## Yêu cầu hệ thống
- Python >= 3.8
- PyTorch >= 1.10
- torchvision, scikit-learn, numpy
- GPU (khuyến nghị)

## Chuẩn bị dữ liệu
1. Tải và giải nén dữ liệu NIH Chest X-ray bằng script:
   ```bash
   cd Database
   python batch_download_zips.py
   ```
   > Dữ liệu sẽ được giải nén vào thư mục riêng trong CheXNet/Database.

2. Tạo các file danh sách:  
   - `Dataset/train_list.txt`
   - `Dataset/val_list.txt`
   - `Dataset/test_list.txt`  
   (Mỗi file chứa đường dẫn ảnh và label; xem ví dụ trong tài liệu NIH hoặc liên hệ tác giả.)

## Huấn luyện mô hình
Chỉnh sửa và chạy file `main.py`:
```bash
python main.py
```
- Hàm `runTrain()` sẽ huấn luyện mô hình và lưu checkpoint vào `Trainedmodel/chexnetmodel.pth`.
- Hàm `runTest()` sẽ kiểm tra mô hình đã huấn luyện, in ra chỉ số AUROC cho từng loại bệnh.

## Kiến trúc code
- **main.py**: Quản lý quá trình train/test.
- **Models/TrainModel.py**: Định nghĩa class HybridTrainer (train 2 giai đoạn, AMP BF16, tính AUROC).
- **Models/Model.py**: Định nghĩa mô hình Hybrid ConvNeXtV2 + SwinV2 với Residual Masking.
- **Models/read_data.py**: Chuẩn bị dữ liệu cho DataLoader, HybridBatchSampler.
- **Database/batch_download_zips.py**: Script tải và giải nén dữ liệu.

## Kết quả
- Chỉ số AUROC cho từng loại bệnh sẽ được in ra khi kiểm tra mô hình.
- Checkpoint lưu lại mô hình tốt nhất theo loss.

## Tài liệu tham khảo
- [CheXNet Paper](https://arxiv.org/abs/1711.05225)
- [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
