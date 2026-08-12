# CheXNet — Hybrid CNN-ViT Multi-Label Chest X-ray Classification

Dự án Deep Learning nhận diện 15 loại bệnh phổi từ ảnh X-ray, sử dụng kiến trúc lai tiên tiến kết hợp giữa **ConvNeXtV2** và **SwinV2** cùng cơ chế **Residual Masking**. Mô hình được thiết kế để phân loại chính xác đa nhãn (multi-label) và tối ưu hóa qua các bộ dữ liệu lớn như NIH Chest X-ray.

## 🌟 Tính năng nổi bật
- **Kiến trúc Hybrid CNN-ViT**: Kết hợp ưu điểm trích xuất đặc trưng cục bộ xuất sắc của ConvNeXtV2 và khả năng nắm bắt ngữ cảnh toàn cục của Swin Transformer V2.
- **Hỗ trợ Multi-Label**: Phân loại đồng thời 15 nhãn (14 bệnh lý phổi và No Finding): *Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia*.
- **Test-Time Augmentation (TTA)**: Áp dụng trong quá trình kiểm thử để cải thiện độ ổn định và tăng độ chính xác (AUROC).
- **Dự đoán và Trực quan hóa (Inference)**: Chẩn đoán trên một ảnh X-quang bất kỳ, xuất ra biểu đồ xác suất và bản đồ chú ý (Heatmap/Attention Map) để giải thích vùng bệnh lý mà mô hình tập trung vào.
- **Tối ưu hóa Huấn luyện**: Tích hợp AMP (Automatic Mixed Precision - BF16/FP32), tận dụng Tensor Cores, và hỗ trợ `torch.compile` để tăng tốc độ huấn luyện tối đa.

## 🛠 Yêu cầu hệ thống
- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- **Thư viện khác**: `scikit-learn`, `numpy`, `opencv-python`, `matplotlib`, `seaborn`, `pandas`, `tqdm` (chi tiết xem tại `requirements.txt`).
- **Phần cứng**: Khuyến nghị sử dụng GPU (VRAM ≥ 12GB cho cấu hình Base, ≥ 24GB cho cấu hình Large).

## 📂 Cấu trúc dự án tiêu biểu
- `main.py`: Điểm đầu vào chính để chạy Huấn luyện (Train) và Kiểm thử (Test) mô hình qua menu tương tác.
- `predict_single.py`: Công cụ dự đoán trên một ảnh X-ray đơn lẻ và trực quan hóa kết quả (sinh ra heatmap).
- `Models/`: Thư mục cốt lõi chứa mã nguồn mô hình.
  - `Model.py`: Định nghĩa kiến trúc Hybrid ConvNeXtV2 + SwinV2.
  - `TrainModel.py`: Quản lý quy trình huấn luyện, tính toán hàm loss và metrics.
  - `read_data.py`: Các lớp xử lý dữ liệu và DataLoader tối ưu.
  - `checkpoint_utils.py`: Hỗ trợ lưu và tải checkpoint an toàn.
- `Database/` & `Dataset/`: Chứa script tải dữ liệu (vd: NIH Dataset) và các file phân chia train/val/test.
- `Trainedmodel/`: Thư mục mặc định lưu các file trọng số mô hình (`.pth`).
- `Results/`: Nơi xuất các biểu đồ đánh giá (AUROC, ROC, PR Curves, Confusion Matrix) và ảnh kết quả dự đoán.

## 🚀 Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu và Môi trường
Cài đặt thư viện:
```bash
pip install -r requirements.txt
```
Tải dữ liệu NIH Chest X-ray (nếu sử dụng):
```bash
cd Database
python batch_download_zips.py
```
*Dữ liệu sẽ được giải nén vào `Database/`. Hãy đảm bảo bạn đã chuẩn bị các file `train_list.csv`, `val_list.csv`, `test_list.csv` trong thư mục `Dataset/` với định dạng đúng (đường dẫn ảnh và các cột nhãn one-hot).*

### 2. Huấn luyện (Training) & Kiểm thử (Testing)
Để bắt đầu quá trình huấn luyện hoặc kiểm thử, hãy chạy `main.py`:
```bash
python main.py
```
Hệ thống sẽ hiển thị một menu để bạn lựa chọn:
1. **Train mới từ đầu**: Thiết lập các tham số (model size: Base/Large, image size: 256/384, batch size, v.v.) và bắt đầu huấn luyện.
2. **Train tiếp từ checkpoint (Resume)**: Tiếp tục quá trình huấn luyện đang dang dở từ một file `.pth`.
3. **Test / Đánh giá**: Chạy kiểm thử trên tập Test. Mô hình sẽ tính toán chỉ số AUROC, PR-AUC, F1-score và tìm ra ngưỡng tối ưu (optimal thresholds) cho từng bệnh. Kết quả và các biểu đồ chi tiết sẽ được tự động lưu vào thư mục `Results/`.

### 3. Dự đoán ảnh đơn (Inference)
Để chẩn đoán nhanh trên một tấm ảnh X-quang thực tế và xem Heatmap vùng tổn thương:
```bash
python predict_single.py
```
- Script sẽ yêu cầu bạn nhập đường dẫn đến file model (`.pth`) và đường dẫn ảnh X-quang cần kiểm tra.
- Kết quả chẩn đoán bao gồm xác suất cho 15 nhãn sẽ được in ra terminal.
- Một ảnh tổng hợp gồm: **Ảnh gốc, Bản đồ nhiệt (Heatmap) cho các bệnh dương tính, và Biểu đồ xác suất** sẽ được tạo ra và lưu trong thư mục `Results/`.

## 📈 Trực quan hóa & Đánh giá
Trong chế độ **Test**, chương trình cung cấp bộ công cụ phân tích mô hình mạnh mẽ lưu tại `Results/`:
- **Bar chart AUROC**: So sánh hiệu suất nhận diện cho từng loại bệnh.
- **ROC & Precision-Recall Curves**: Biểu đồ chi tiết thể hiện khả năng phân loại của mô hình.
- **Probability Distribution**: Đánh giá sự phân tách giữa các dự đoán Positive/Negative.
- **Confusion Matrix**: Ma trận nhầm lẫn tổng thể và phân tách riêng cho từng lớp (Grid).

## 📚 Tài liệu tham khảo
- [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/abs/1711.05225)
- [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
