# BÁO CÁO PHÂN TÍCH TOÀN DIỆN CÁC GIẢI PHÁP HÀNG ĐẦU
## Cuộc thi Kaggle: VinBigData Chest X-ray Abnormalities Detection

---

## I. TỔNG QUAN CUỘC THI & BÀI TOÁN

- **Tên cuộc thi**: VinBigData Chest X-ray Abnormalities Detection (Đăng tải trên Kaggle, 2021).
- **Đơn vị tổ chức**: Viện Nghiên cứu Dữ liệu lớn VinBigData (Vingroup).
- **Mục tiêu**: Tự động định vị (Localization - Bounding Box) và phân loại (Classification) 14 loại tổn thương/bất thường lồng ngực trên ảnh X-quang ngực (Chest Radiographs), kèm theo 1 lớp thứ 15 đại diện cho trường hợp bình thường: `No finding`.
- **Danh sách 14 tổn thương**:
  1. Aortic enlargement (Phì đại động mạch chủ)
  2. Atelectasis (Xẹp phổi)
  3. Calcification (Vôi hóa)
  4. Cardiomegaly (Tim to)
  5. Consolidation (Đông đặc phổi)
  6. ILD (Bệnh phổi kẽ)
  7. Infiltration (Thâm nhiễm)
  8. Lung Opacity (Mờ phổi)
  9. Nodule/Mass (Nốt/Khối mờ)
  10. Other lesion (Tổn thương khác)
  11. Pleural effusion (Tràn dịch màng phổi)
  12. Pleural thickening (Dày màng phổi)
  13. Pneumothorax (Tràn khí màng phổi)
  14. Pulmonary fibrosis (Xơ phổi)
  15. *No finding* (Không phát hiện tổn thương)
- **Bộ dữ liệu**: 15,000 ảnh huấn luyện và 4,394 ảnh kiểm tra định dạng DICOM, được dán nhãn bởi 17 bác sĩ chẩn đoán hình ảnh giàu kinh nghiệm tại Việt Nam.
- **Thước đo đánh giá (Evaluation Metric)**: **mAP@0.4** (mean Average Precision tại ngưỡng điểm trùng khớp IoU = 0.4).

---

## II. PHÂN TÍCH CHI TIẾT GIẢI PHÁP HẠNG NHÌ (2ND PLACE SOLUTION)

### 1. Thông tin chung
- **Tác giả chính**: **Ivan Panshin (@ivanpan)** & **ZFTurbo** (tác giả thư viện *Weighted Boxes Fusion - WBF* nổi tiếng).
- **Mã nguồn Repository**: [ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection](https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection)
- **Bài viết thảo luận Kaggle**: [2nd place solution quick overview](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/229696) | [Addition about MMDetection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/229740)

### 2. Kiến trúc Mô hình & Framework
Tác giả chuẩn hóa toàn bộ pipeline trên framework **MMDetection** (hệ sinh thái OpenMMLab), sử dụng tập hợp mô hình đa dạng (Ensemble):

- **Cascade RFP ResNet-50 (`Cascade_RFP_R50`)**: Mô hình Cascade R-CNN kết hợp *Recursive Feature Pyramid (RFP)*. Cơ chế tự lặp đặc trưng giúp trích xuất thông tin đa quy mô cực tốt đối với các vi tổn thương.
- **Generalized Focal Loss (`GFL_R101` & `GFL_X101`)**: Sử dụng Backbone ResNet-101 và ResNeXt-101. Mô hình tối ưu đồng thời việc phân loại và chất lượng Bounding Box thông qua hàm mất mát Focal Loss mở rộng.
- **RetinaNet ResNeXt-101 (`RetinaNet_X101`)**: Mô hình One-stage Detector cung cấp sự khác biệt về mặt cấu trúc đặc trưng khi hòa trộn kết quả với các mô hình Two-stage.
- **ATSS & UniverseNet**: Được thử nghiệm trong quá trình Grid Search để bổ sung độ phong phú cho Ensemble.

### 3. Kỹ thuật Huấn luyện & Đa độ phân giải
- **Chiến lược 2 giai đoạn độ phân giải**:
  - Giai đoạn 1: Huấn luyện ban đầu ở độ phân giải $1024 \times 1024$.
  - Giai đoạn 2: Fine-tune ở độ phân giải rất cao $2048 \times 2048$. Điều này giúp phát hiện các nốt mờ, vôi hóa hoặc tổn thương dính màng phổi nhỏ mà ở kích thước $512 \times 512$ hay bị mất chi tiết.
- **Kỹ thuật FP16 (Mixed Precision)**: Giúp giảm tải dung lượng VRAM và đẩy nhanh tốc độ tính toán khi huấn luyện ảnh $2048 \times 2048$.
- **Validation**: Áp dụng 5-Fold Cross-Validation nghiêm ngặt để tránh rò rỉ dữ liệu giữa các bệnh nhân.

### 4. Data Augmentation (Albumentations)
- **Biến dạng hình học**: `ShiftScaleRotate`, `IAAAffine`, `HorizontalFlip`.
- **Xử lý nhiễu & Làm mờ**: `Blur`, `GaussianBlur`, `MedianBlur`, `GaussNoise` / `IAAAdditiveGaussianNoise`.
- **Tương phản & Độ sáng**: `RandomBrightnessContrast` giúp mô hình thích nghi với các thiết bị chụp X-quang có cấu hình tia X khác nhau.

### 5. Hậu xử lý & Ensemble (WBF & Thresholding)
- **Weighted Boxes Fusion (WBF)**:
  - Thay vì dùng NMS (vốn chỉ giữ 1 box tốt nhất và xóa các box khác), WBF gom nhóm các Bounding Box đè lên nhau từ nhiều mô hình/folds, sau đó tính trung bình có trọng số vị trí và confidence score.
  - Kết quả tạo ra khung Bounding Box ôm sát tổn thương hơn nhiều.
- **Xử lý lớp `No finding`**:
  - Sử dụng bộ lọc ngưỡng (Thresholding) riêng cho từng lớp. Nếu ảnh không có tổn thương nào đạt ngưỡng tin cậy tối thiểu, hệ thống sẽ trả về dự đoán chuẩn `14 1.0 0 0 1 1`.

---

## III. PHÂN TÍCH CHI TIẾT GIẢI PHÁP HẠNG NHẤT (1ST PLACE SOLUTION)

### 1. Thông tin chung
- **Thành viên đội vô địch**: **Mohammed Rizin V K (@morizin)**, Sergio Manuel Papadakis, avsanjay, và Fatih Öztürk (@fatihozturk).
- **Bài viết thảo luận Kaggle**: [1st Place Solution Writeup](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/231511)

### 2. Kiến trúc Multi-Framework & Mô hình cốt lõi
Nhóm Hạng nhất không dùng duy nhất 1 framework mà kết hợp 3 framework khác nhau để tạo ra sự đa dạng tối đa:

1. **Detectron2 (Facebook AI Research)**:
   - Sử dụng `Faster R-CNN` và `Cascade R-CNN` với Backbone **ResNet-101**.
   - Đóng vai trò bộ định vị khung hình chuẩn xác nhất (Two-stage).
2. **YOLOv5 (Ultralytics)**:
   - Sử dụng biến thể **YOLOv5x**.
   - Đóng vai trò bộ phát hiện One-stage tốc độ cao, trích xuất đặc trưng theo cơ chế hoàn toàn khác so với R-CNN.
3. **EfficientDet (Google Brain)**:
   - Sử dụng **EfficientDet-D2 / D4** với mạng BiFPN giúp kết hợp thông tin phân cấp đa tầng.

### 3. Quy trình 2 Giai đoạn (2-Stage Pipeline: Classifier + Detector)
- **Giai đoạn 1 - Image-Level Classifier**: Huấn luyện một mô hình Phân loại ảnh (như EfficientNet/ResNet) để dự đoán xác suất ảnh đó là **Lành tính (`No finding`)** hay **Bất thường**.
- **Giai đoạn 2 - Object Detector Ensemble**: Các mô hình Detectron2, YOLOv5, EfficientDet đưa ra danh sách các Bounding Box.
- **Kết hợp (Fusion Logic)**:
  - Dự đoán của Classifier được dùng để hiệu chỉnh điểm số tin cậy (Confidence Score) của Detector.
  - Nếu Classifier khẳng định 99% ảnh là `No finding`, mọi Bounding Box điểm thấp từ Detector sẽ bị triệt tiêu, trả về `14 1.0 0 0 1 1`.
  - Nếu Classifier dự đoán chắc chắn có tổn thương, điểm số của các Bounding Box từ Detector được nhân hệ số tăng cường.

### 4. Khai thác Thước đo mAP@0.4 (Recall Maximization Strategy)
- **Phát hiện quy luật cuộc thi**: Chỉ số mAP@0.4 không phạt nặng việc đưa ra các Bounding Box thừa có điểm tin cậy thấp, nhưng phạt rất nặng nếu bỏ sót tổn thương (False Negative).
- **Chiến thuật**: Hạ thấp ngưỡng tin cậy sau khi gộp WBF để giữ lại nhiều candidate box tiềm năng. Việc tăng tối đa Độ phủ (Recall) giúp nâng điểm mAP tổng thể lên vị trí dẫn đầu bảng xếp hạng.

### 5. Class-wise Models
Huấn luyện riêng các mô hình chuyên biệt cho một số lớp tổn thương khó nhận diện (như Nodule/Mass hoặc Pneumothorax) để cải thiện điểm số cho các lớp có tỷ lệ mẫu thấp.

---

## IV. BẢNG SO SÁNH TRỰC QUAN HẠNG NHẤT VS HẠNG NHÌ

| Tiêu chí so sánh | 🏆 Giải pháp Hạng nhất (1st Place) | 🥈 Giải pháp Hạng nhì (2nd Place) |
| :--- | :--- | :--- |
| **Đội ngũ** | @morizin, @fatihozturk, S. Papadakis, A. Sanjay | @ivanpan & ZFTurbo |
| **Triết lý Framework** | **Đa Framework**: Detectron2 + PyTorch YOLOv5 + EfficientDet | **Đồng nhất**: MMDetection Framework |
| **Kiến trúc chính** | Detectron2 (ResNet-101), YOLOv5x, EfficientDet-D2/D4 | Cascade RFP R50, GFL R101/X101, RetinaNet X101 |
| **Xử lý `No finding`** | **2-Stage Pipeline**: Dùng Image-Level Classifier chuyên biệt làm bộ lọc điều chỉnh điểm Bbox | **Thresholding & Class Filtering**: Dùng ngưỡng tin cậy theo từng lớp |
| **Độ phân giải ảnh** | Đa dạng ($512\times512$, $1024\times1024$) | Siêu phân giải ($1024\times1024$ fine-tune $2048\times2048$) |
| **Tối ưu chỉ số mAP** | Hạ ngưỡng Confidence để tối đa hóa Recall (tăng lượng Bbox dự đoán) | Fine-tune ảnh $2048\times2048$ để bắt tổn thương vi mô chính xác |
| **Thuật toán Ensemble** | Weighted Boxes Fusion (WBF) | Weighted Boxes Fusion (WBF) |

---

## V. CÁC BÀI HỌC KỸ THUẬT QUAN TRỌNG (KEY TAKEAWAYS)

1. **Ứng dụng Weighted Boxes Fusion (WBF)**: Đối với các bài toán phát hiện vật thể trong y tế có độ chồng lấp cao, WBF vượt trội hoàn toàn so với NMS truyền thống nhờ cơ chế tính trung bình có trọng số vị trí Bounding Box.
2. **Đa dạng hóa Framework (Multi-Framework Ensemble)**: Kết hợp các họ kiến trúc đến từ các đại gia công nghệ khác nhau (Detectron2 của Meta, YOLOv5 của Ultralytics, EfficientDet của Google) mang lại đa dạng đặc trưng vượt trội hơn việc chỉ ensemble các backbone trong cùng 1 framework.
3. **Mô hình Phân loại 2 Giai đoạn (Image-level Classifier Filter)**: Trong bài toán vừa phát hiện vị trí vừa có lớp `No finding`, việc kết hợp bộ phân loại cấp ảnh (Classifier) để điều chỉnh điểm số của bộ phát hiện vị trí (Detector) là chiến thuật quyết định giúp loại bỏ tối đa âm tính/dương tính giả.
4. **Hiểu rõ Thước đo Đánh giá (Metric Exploitation)**: Phân tích kỹ cách tính điểm mAP@0.4 giúp thiết lập ngưỡng Confidence phù hợp để tối đa hóa điểm số mà không vi phạm quy tắc cuộc thi.
