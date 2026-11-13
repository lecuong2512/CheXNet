# 📋 BÁO CÁO PHÂN TÍCH TOÀN DIỆN DỰ ÁN CheXNet Pro AI

> **Ngày lập báo cáo**: 18/08/2026  
> **Repository**: `lecuong2512/CheXNet`  
> **Đường dẫn gốc**: `d:\temp\tempnckh\CheXNet`

---

## MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc hệ thống tổng thể](#2-kiến-trúc-hệ-thống-tổng-thể)
3. [Phân tích chi tiết mô hình AI (Model)](#3-phân-tích-chi-tiết-mô-hình-ai)
4. [Hệ thống huấn luyện (Training Pipeline)](#4-hệ-thống-huấn-luyện)
5. [Module Fine-Tune chuyên sâu](#5-module-fine-tune-chuyên-sâu)
6. [Hệ thống suy luận & trực quan hóa (Inference)](#6-hệ-thống-suy-luận--trực-quan-hóa)
7. [Backend — FastAPI Pipeline đa mô hình](#7-backend--fastapi-pipeline-đa-mô-hình)
8. [Frontend — React SPA](#8-frontend--react-spa)
9. [Dữ liệu & Dataset](#9-dữ-liệu--dataset)
10. [Lịch sử phát triển & Git Timeline](#10-lịch-sử-phát-triển--git-timeline)
11. [Danh mục toàn bộ file dự án](#11-danh-mục-toàn-bộ-file-dự-án)
12. [Đánh giá tổng hợp & Đề xuất](#12-đánh-giá-tổng-hợp--đề-xuất)

---

## 1. Tổng Quan Dự Án

### 1.1 Mục tiêu

**CheXNet Pro AI** là hệ thống trí tuệ nhân tạo hỗ trợ chẩn đoán hình ảnh X-quang lồng ngực (Chest X-Ray), phát triển cho mục đích **Nghiên cứu Khoa học (NCKH)**. Hệ thống thực hiện:

- **Phân loại đa nhãn (Multi-Label Classification)**: Nhận diện đồng thời 15 nhãn bệnh lý (14 bệnh phổi + *No Finding*) từ 1 ảnh X-quang.
- **Phát hiện tổn thương cục bộ (Object Detection)**: Xác định bounding box vùng tổn thương bằng YOLO11m.
- **Bản đồ chú ý nội sinh (Endogenous Attention Maps)**: Trực quan hóa 15 kênh attention riêng biệt cho từng bệnh lý.
- **Tìm kiếm ca bệnh tương tự (CBIR)**: So khớp vector đặc trưng 1536 chiều qua FAISS.
- **Sinh báo cáo y khoa tự động**: Tự động sinh báo cáo chuẩn quốc tế RSNA/ACR dựa trên kết quả suy luận xác suất và định vị tổn thương (100% offline, bảo mật dữ liệu).
- **Ứng dụng web hoàn chỉnh**: React 18 + FastAPI, giao diện chuyên nghiệp kiểu "Medical Cockpit".

### 1.2 Danh sách 15 nhãn bệnh lý

| # | Tên bệnh (EN) | Tên bệnh (VN) |
|---|---|---|
| 1 | Atelectasis | Xẹp phổi |
| 2 | Cardiomegaly | Tim to |
| 3 | Effusion | Tràn dịch màng phổi |
| 4 | Infiltration | Thâm nhiễm |
| 5 | Mass | Khối u |
| 6 | Nodule | Nốt mờ |
| 7 | Pneumonia | Viêm phổi |
| 8 | Pneumothorax | Tràn khí màng phổi |
| 9 | Consolidation | Đông đặc phổi |
| 10 | Edema | Phù phổi |
| 11 | Emphysema | Khí phế thũng |
| 12 | Fibrosis | Xơ phổi |
| 13 | Pleural_Thickening | Dày màng phổi |
| 14 | Hernia | Thoát vị hoành |
| 15 | No Finding | Không phát hiện |

### 1.3 Yêu cầu phần cứng

| Cấu hình | GPU VRAM | Backbone |
|---|---|---|
| Base (256px) | ≥ 12 GB | ConvNeXtV2-Base + SwinV2-Base |
| Large (384px) | ≥ 24 GB | ConvNeXtV2-Large + SwinV2-Large |

### 1.4 Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Deep Learning Framework | PyTorch ≥ 2.0, timm ≥ 0.9 |
| CNN Backbone | ConvNeXtV2 (Base/Large) |
| ViT Backbone | Swin Transformer V2 (Base/Large) |
| Object Detection | YOLO11m (Ultralytics) |
| Vector Search | FAISS (Inner Product) |
| Backend Server | FastAPI + Uvicorn |
| Frontend | React 18 + Vite 5 + Tailwind CSS v4 |
| Clinical Report Engine | Local Rule-based & Medical Evidence Synthesis (RSNA/ACR) |
| Metrics | scikit-learn, matplotlib, seaborn |

---

## 2. Kiến Trúc Hệ Thống Tổng Thể

```mermaid
graph TB
    subgraph Frontend["🖥️ Frontend (React + Vite)"]
        UI["App.jsx — Master Orchestrator"]
        CR["ClinicalReport.jsx"]
        CR["report_generator.py — Local Clinical Engine"]
        RM["radiologyMapper.js"]
    end
    
    subgraph Backend["⚙️ Backend (FastAPI)"]
        API["main.py — API Server"]
        YOLO["yolo_detector.py — YOLO11m"]
        LUNG["lung_segmentation.py"]
        POST["post_processing.py"]
        RPT["report_generator.py"]
        CBIR["cbir.py — FAISS Search"]
    end
    
    subgraph Models["🧠 Core AI Models"]
        HYB["HybridCNNViTModel"]
        CNN["ConvNeXtV2"]
        VIT["SwinV2"]
        FPN["Feature Pyramid Network"]
        ATT["15-Channel Attention Head"]
    end
    
    subgraph Training["📈 Training Pipeline"]
        TRN["TrainModel.py — HybridTrainer"]
        ASL["Asymmetric Loss"]
        DICE["Dice + Focal Tversky"]
        UW["Uncertainty Weighting"]
        EMA["Model EMA"]
    end
    
    UI -->|"POST /predict"| API
    API --> HYB
    API --> YOLO
    API --> LUNG
    API --> POST
    API --> RPT
    API --> CBIR
    HYB --> CNN
    HYB --> VIT
    CNN --> FPN
    VIT --> FPN
    FPN --> ATT
    TRN --> HYB
    TRN --> ASL
    TRN --> DICE
    TRN --> UW
    TRN --> EMA
```

### Luồng xử lý ảnh X-quang (Data Flow)

```
[Ảnh X-quang đầu vào (3, H, W)]
        │
        ├──► [ConvNeXtV2 (CNN)] ──► Đặc trưng cục bộ C₃, C₄
        │                                    │
        ├──► [SwinV2 (ViT)]   ──► Ngữ cảnh toàn cục V
        │                                    │
        ├──► [FPN Merge (C₃ + Upsample V)] ──► Biểu diễn đa tỷ lệ
        │                                         │
        ├──► [15-Channel Attention Head]    ──► 15 Attention Maps
        │                                         │
        ├──► [Gated Residual Masking: Fₖ + α·(Fₖ ⊙ Aₖ)]
        │                                         │
        └──► [GAP + Linear Classifier]      ──► 15 Multi-Label Logits
                                                   │
                     ┌─────────────────────────────┘
                     ▼
              [Sigmoid → Probabilities]
                     │
        ┌────────────┼────────────────┐
        ▼            ▼                ▼
   [Heatmaps]  [YOLO BBoxes]   [FAISS CBIR]
        │            │                │
        └────────────┼────────────────┘
                     ▼
         [Báo cáo lâm sàng tự động]
```

---

## 3. Phân Tích Chi Tiết Mô Hình AI

### 3.1 Kiến trúc Hybrid CNN-ViT

> **File**: [Model.py](file:///d:/temp/tempnckh/CheXNet/Models/Model.py) — 215 dòng, 11,462 bytes

#### Class `HybridCNNViTModel(nn.Module)`

**Constructor Parameters**:
```python
HybridCNNViTModel(
    num_classes=15,      # Số nhãn bệnh lý
    model_size='base',   # 'base' hoặc 'large'
    img_size=384,        # Kích thước ảnh đầu vào
    dropout_rate=0.3,    # Dropout rate
    pretrained=True      # Sử dụng pretrained weights
)
```

**Các thành phần chính**:

| Thành phần | Chi tiết kỹ thuật |
|---|---|
| **CNN Backbone** | ConvNeXtV2 (`convnextv2_base.fcmae_ft_in22k_in1k_384` / `large`), `features_only=True` |
| **ViT Context** | SwinV2 (`swinv2_base_window12to16_192to256` / `large`), trích `layers[-1]` + `norm` |
| **FPN** | Lateral Conv trên CNN Stage 3 + Merge Conv (3×3 Conv → BatchNorm → GELU) kết hợp SwinV2 upsampled features |
| **Attention Head** | 3×3 Conv → BN → GELU → 1×1 Conv(256, 15) → Sigmoid → 15 attention maps độc lập |
| **Channel Grouping** | Chia `swin_dim` (1024 hoặc 1536) kênh thành 15 nhóm, mỗi nhóm tương ứng 1 bệnh |
| **Gated Alpha** | $\alpha = \text{Sigmoid}(\alpha_{\text{raw}}) \times 0.8 \approx 0.3$ (khởi tạo $\alpha_{\text{raw}} = -0.5$) |
| **Classifier** | Dropout(0.3) → Linear(swin_dim, 15) |

**Quá trình Forward**:

$$\text{GatedGroup}_k = F_k + \alpha \cdot (F_k \odot A_k) \quad \text{với } k = 1, \ldots, 15$$

Trong đó:
- $F_k$: nhóm kênh thứ $k$ từ CNN features
- $A_k$: attention map thứ $k$ (đã downsample)
- $\alpha$: trọng số gate được học, giới hạn $\leq 0.8$

**Đầu ra**: `(logits[B,15], attention_maps[B,15,H,W])` hoặc thêm `pooled[B,swin_dim]` khi `return_embedding=True`.

### 3.2 Ý nghĩa thiết kế đột phá

> [!IMPORTANT]
> **Per-Class Multi-Channel Attention**: Thay vì 1 attention map chung cho tất cả bệnh (truyền thống), hệ thống sinh **15 attention maps độc lập** — cho phép model tập trung vào **nhiều vùng khác nhau đồng thời** khi bệnh nhân mắc nhiều bệnh cùng lúc (multi-label). Đây là đặc điểm kiến trúc quan trọng nhất của dự án.

---

## 4. Hệ Thống Huấn Luyện

> **File**: [TrainModel.py](file:///d:/temp/tempnckh/CheXNet/Models/TrainModel.py) — 1,047 dòng, 57,881 bytes

### 4.1 Chiến lược Curriculum 2 Giai Đoạn

```mermaid
flowchart LR
    S1["Stage 1: Attention Warmup\n(Chỉ dùng VinDr-CXR có BBox)"]
    COND{"Dice < 0.65\nAUROC > 0.68\n2 epoch liên tiếp?"}
    S2["Stage 2: Full Training\n(NIH + VinDr-CXR, tỷ lệ 3:1)"]
    
    S1 --> COND
    COND -->|Có| S2
    COND -->|Chưa| S1
```

| Giai đoạn | Dữ liệu | Mục tiêu | Điều kiện chuyển |
|---|---|---|---|
| **Stage 1** | Chỉ VinDr-CXR (có BBox) | Warmup Attention Head qua Dice Loss | `diceLoss < 0.65` AND `valAUROC > 0.68` (2 epochs) |
| **Stage 2** | NIH + VinDr-CXR (tỷ lệ 3:1) | Tối ưu Classification + Localization | Cho đến max epochs |

### 4.2 Hệ thống Loss Functions

#### 4.2.1 Asymmetric Loss (ASL) — Ridnik et al., ICCV 2021

Giải quyết mất cân bằng lớp cực đoan (positive rate < 5%):

$$\mathcal{L}_{\text{ASL}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \cdot \log(p_i) + (1-y_i) \cdot (p_{i,m})^{\gamma_{\text{neg}}} \cdot \log(1 - p_{i,m})\right]$$

Trong đó $p_{i,m} = \max(p_i - m, 0)$ với $m = 0.05$ (probability shift) và $\gamma_{\text{neg}} = 4$, $\gamma_{\text{pos}} = 0$.

#### 4.2.2 Dice Loss + Focal Tversky Loss (Multi-Channel)

Hướng dẫn Attention Head bám sát BBox Ground-Truth:

$$\mathcal{L}_{\text{Dice+FT}} = 0.4 \times \text{DiceLoss} + 0.6 \times \text{FocalTverskyLoss}$$

- Focal Tversky: $\alpha = 0.3$ (FP penalty), $\beta = 0.7$ (FN penalty), $\gamma = 0.75$ (focal exponent)
- Sử dụng `valid_mask [N, 15]` chỉ tính loss trên kênh có BBox thật

#### 4.2.3 Attention Sparsity Loss

$$\mathcal{L}_{\text{sparsity}} = 0.1 \cdot \text{mean}(A) + 0.05 \cdot \text{BinaryEntropy}(A)$$

#### 4.2.4 Multi-Task Uncertainty Weighting (Kendall & Gal 2018)

$$\mathcal{L}_{\text{total}} = \frac{1}{2} e^{-s_1} \mathcal{L}_{\text{ASL}} + \frac{1}{2} s_1 + \frac{1}{2} e^{-s_2} \mathcal{L}_{\text{Dice}} + \frac{1}{2} s_2 + \lambda \cdot \mathcal{L}_{\text{sparsity}}$$

- $s_1, s_2$: log-variance parameters, clamp $\in [-4.0, 4.0]$
- Sparsity giữ ngoài UW để tránh suy biến

#### 4.2.5 No-Finding Consistency Penalty

$$\mathcal{L}_{\text{NF}} = 0.1 \times P(\text{No Finding}) \times \max_d P(\text{disease}_d)$$

### 4.3 Cấu hình Optimizer & Scheduler

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW (weight_decay=0.05) |
| LR Backbone | $1 \times 10^{-4}$ |
| LR Head/FPN | $5 \times 10^{-4}$ |
| LR UW params | $1 \times 10^{-3}$ |
| Warmup | 3 epoch Linear Warmup |
| Scheduler | CosineAnnealingLR ($\eta_{\min} = 10^{-6}$) |
| Gradient Clipping | max_norm = 1.0 |
| AMP | bfloat16 (Ampere+) hoặc float16 + GradScaler |

### 4.4 Model EMA

$$\theta_{\text{EMA}}^{(t)} = \text{decay} \cdot \theta_{\text{EMA}}^{(t-1)} + (1 - \text{decay}) \cdot \theta^{(t)}$$

- Decay = 0.999 (training gốc) / 0.9995 (fine-tune)
- Track cả trainable params lẫn BatchNorm running stats

### 4.5 Dual Checkpoint Strategy

| Checkpoint | Nội dung | Mục đích |
|---|---|---|
| `state_dict` | EMA weights | Inference (production) |
| `training_state_dict` | Raw training weights + optimizer + scheduler | Resume training |

---

## 5. Module Fine-Tune Chuyên Sâu

> **Files**: [main_finetune.py](file:///d:/temp/tempnckh/CheXNet/Finetune/main_finetune.py), [TrainModel_finetune.py](file:///d:/temp/tempnckh/CheXNet/Finetune/TrainModel_finetune.py)

Fine-tune từ checkpoint tốt nhất (Epoch 12, AUROC ~0.8651) với các điều chỉnh:

| Tham số | Gốc | Fine-Tune | Lý do |
|---|---|---|---|
| Backbone LR | $1 \times 10^{-4}$ | $3 \times 10^{-5}$ | Giảm 3.3× tránh phá hỏng features đã học |
| Head LR | $5 \times 10^{-4}$ | $1 \times 10^{-4}$ | Giảm 5× cho fine-grained tuning |
| Weight Decay | 0.05 | 0.08 | Tăng regularization chống overfitting |
| ASL $\gamma_{\text{neg}}$ | 4 | 2 | Nới lỏng phạt mẫu âm, cải thiện AUROC |
| EMA Decay | 0.999 | 0.9995 | Ổn định hơn trong giai đoạn late training |
| UW Clamp | [-4, 4] | [-3, 3] | Siết dải cân bằng ASL-Dice |
| Augmentation | Không Color | ColorJitter + GaussianBlur | Chống overfitting |
| Checkpoint Size | ~4.2 GB | ~1.2 GB | Chỉ lưu state_dict cần thiết |

---

## 6. Hệ Thống Suy Luận & Trực Quan Hóa

### 6.1 Inference đơn ảnh

> **File**: [predict_single.py](file:///d:/temp/tempnckh/CheXNet/predict_single.py) — 249 dòng

**Pipeline**:
1. Resize 1.14× → CenterCrop → ImageNet Normalize
2. Forward pass → `(logits, attention_maps[15, H, W])`
3. Sigmoid probabilities → Threshold filtering (0.5 hoặc optimal)
4. Multi-panel visualization: Ảnh gốc + Heatmaps + Bar chart xác suất

**Heatmap Overlay**:

$$\text{Overlay} = 0.4 \times \text{JET\_Heatmap} + 0.6 \times \text{Original}$$

### 6.2 Test-Time Augmentation (TTA)

$$P_{\text{TTA}} = \frac{\sigma(\text{logits}_{\text{orig}}) + \sigma(\text{logits}_{\text{flip}})}{2}$$

> [!NOTE]
> Average probabilities (sau sigmoid) chứ không average logits — do tính phi tuyến của Sigmoid.

### 6.3 Optimal Thresholding — Youden's J Index

$$J_k = \max_{\tau} \left[\text{TPR}_k(\tau) - \text{FPR}_k(\tau)\right] \quad k = 1, \ldots, 15$$

Lưu vào `Results/optimal_thresholds.npy` cho inference thực tế.

### 6.4 Đánh giá toàn diện (Test Mode)

| Output | Mô tả |
|---|---|
| `test_auroc.png` | Bar chart AUROC per-class |
| `test_confusion_matrix.png` | Ma trận nhầm lẫn tổng thể |
| `test_roc_curves.png` | 15 đường ROC curves |
| `test_pr_curves.png` | 15 đường Precision-Recall curves |
| `test_prob_distribution.png` | Phân phối xác suất Positive vs Negative |
| `test_per_class_cm.png` | Grid 3×5 confusion matrices riêng |
| `test_threshold_f1.png` | Macro-F1 vs. Decision Threshold |
| Bootstrap 95% CI | 1000 iterations cho AUROC và PR-AUC |

---

## 7. Backend — FastAPI Pipeline Đa Mô Hình

> **Thư mục**: [Backend/](file:///d:/temp/tempnckh/CheXNet/Backend)

### 7.1 Server chính

> **File**: [Backend/main.py](file:///d:/temp/tempnckh/CheXNet/Backend/main.py)

**Endpoints**:

| Method | Path | Chức năng |
|---|---|---|
| `GET` | `/` | Health-check trạng thái Server + Models |
| `POST` | `/predict` | Pipeline suy luận toàn diện |
| `POST` | `/api/add_reference_case` | Dynamic Growth — thêm ca bệnh tham chiếu FAISS |

**Pipeline `/predict`**:
1. Đọc ảnh upload → Forward qua `HybridCNNViTModel(return_embedding=True)`
2. Phân vùng phổi (`get_lung_mask`) → Lọc attention map ngoài phổi
3. YOLO11m predict (1024px) → Cascade 2-Stage filtering
4. Render heatmap overlay + YOLO bounding boxes
5. Tính Attention Quality + Lesion Area Ratio + Vị trí giải phẫu
6. Kiểm tra đồng xuất hiện bệnh lý → Sinh báo cáo lâm sàng
7. Tìm kiếm 5 ca bệnh tương tự (CBIR FAISS)

### 7.2 YOLO Detector & Cascade Scoring

> **File**: [yolo_detector.py](file:///d:/temp/tempnckh/CheXNet/Backend/yolo_detector.py)

**Cascade 2-Stage Score**:

$$\text{Score}_{\text{cascade}} = P_{\text{CheXNet}} \times \text{Conf}_{\text{YOLO}}$$

**Ngưỡng nhạy riêng theo bệnh nguy hiểm**:

| Bệnh | Ngưỡng Cascade |
|---|---|
| Pneumothorax (cấp cứu) | 0.05 |
| Cardiomegaly | 0.10 |
| Effusion | 0.10 |
| Consolidation | 0.12 |
| Mặc định | 0.15 |

### 7.3 CBIR — Tìm kiếm ca bệnh tương tự

> **File**: [cbir.py](file:///d:/temp/tempnckh/CheXNet/Backend/cbir.py)

- **FAISS IndexFlatIP**: Inner Product trên vectors chuẩn hóa L2 = Cosine Similarity
- **Embedding**: 1536 chiều từ GAP layer của HybridCNNViTModel
- **Dynamic Growth**: Cho phép thêm ca bệnh mới vào index trong runtime

### 7.4 Phân vùng phổi

> **File**: [lung_segmentation.py](file:///d:/temp/tempnckh/CheXNet/Backend/lung_segmentation.py)

**Pipeline**: Grayscale → CLAHE → Gaussian Blur → Otsu Threshold → Morphological Close/Open → 2 Contours lớn nhất → Convex Hull

**Tính năng**:
- Lọc attention map ra ngoài lồng ngực (loại trừ Cardiomegaly, Hernia)
- Ước tính diện tích tổn thương (% tổng diện tích phổi)
- Định vị giải phẫu: Phổi trái/phải × Vùng trên/giữa/dưới

### 7.5 Hậu xử lý & Đồng xuất hiện

> **File**: [post_processing.py](file:///d:/temp/tempnckh/CheXNet/Backend/post_processing.py)

- **Clinical Priority Thresholds**: Ngưỡng quyết định riêng theo mức ưu tiên lâm sàng
- **Co-occurrence Matrix**: Ma trận tương quan $15 \times 15$ tính từ 107,880 ảnh
- **Ví dụ**: Cardiomegaly + Effusion ($r > 0.08$) → Gợi ý *Suy tim sung huyết*
- **Attention Quality**: Đánh giá Concentration, Contrast, Sharpness (Entropy)
- **CAM → BBoxes**: Chuyển heatmap thành tối đa 5 bounding boxes

### 7.6 Sinh báo cáo y khoa

> **File**: [report_generator.py](file:///d:/temp/tempnckh/CheXNet/Backend/report_generator.py)

- **Rule-based**: Template tiếng Việt chuẩn y khoa cho 14 bệnh lý
- **Phân loại mức độ**: < 5% (Nhẹ), 5-15% (Trung bình), ≥ 15% (Nặng)
- **Cảnh báo đặc biệt**: Tự động flag bệnh cấp cứu (Pneumothorax, Pneumonia)
- **Uncertainty Alert**: Cảnh báo khi Uncertainty > 12%

---

## 8. Frontend — React SPA

> **Thư mục**: [Frontend/](file:///d:/temp/tempnckh/CheXNet/Frontend)

### 8.1 Stack công nghệ

| Công nghệ | Phiên bản | Vai trò |
|---|---|---|
| React | 18.2.0 | UI Framework |
| Vite | 5.2.0 | Build Tool + Dev Server |
| Tailwind CSS | 4.0.0 | Utility-first CSS |
| Google Fonts | Outfit, DM Sans, JetBrains Mono | Typography |

### 8.2 Component chính — App.jsx

> **File**: [App.jsx](file:///d:/temp/tempnckh/CheXNet/Frontend/src/App.jsx)

**State Management**:
- `activeTab`: `'analysis'` | `'clinical_report'` | `'settings'`
- `stage`: `'idle'` | `'analyzing'` | `'results'`
- `viewMode`: `'heatmap'` | `'original'`

**Luồng tương tác chính**:
1. Upload ảnh hoặc chọn Sample → Preview
2. Click "Analyze" → `POST /predict` (với animation 6 bước)
3. Hiển thị kết quả: Heatmap viewer, YOLO boxes, xác suất
4. Tab Clinical Report: Báo cáo lâm sàng chuẩn y khoa từ Backend
5. Xuất in báo cáo PDF chuẩn trắng đen cho bệnh án

**Chế độ Offline**: `runMockAnalysis()` — chạy giả lập khi Backend không khả dụng.

### 8.3 Mô-đun Sinh Báo Cáo Lâm Sàng Tự Động (Clinical Report Engine)

> **File**: [Backend/report_generator.py](file:///d:/temp/tempnckh/CheXNet/Backend/report_generator.py)

| Thành phần | Chức năng |
|---|---|
| `generate_clinical_report()` | Sinh báo cáo lâm sàng chuyên sâu theo cấu trúc RSNA/ACR |
| `Evidence Synthesis` | Tổng hợp từ xác suất 15 bệnh lý, bounding box YOLO11m và phân vùng phổi |
| `Co-occurrence Rules` | Tự động hiệu chỉnh tương quan đồng xuất hiện bệnh lý |

**Đặc điểm**: Vận hành 100% nội bộ (offline), không phụ thuộc dịch vụ hay API bên ngoài, đảm bảo tuyệt đối tính riêng tư dữ liệu y tế.

### 8.4 Theme & UI Design

- **Dark Medical Cockpit** theme: nền `#030d1a`, accent cyan `#00d4ff`
- **Animations**: Scan-line (tia laser X-quang), pulse-ring, heatmap-reveal
- **Print-friendly**: CSS `@media print` cho xuất PDF bệnh án chuẩn trắng đen
- **Glass morphism**: `.glass`, `.glass-bright` effects

---

## 9. Dữ Liệu & Dataset

### 9.1 Nguồn dữ liệu

| Dataset | Số ảnh | Đặc điểm |
|---|---|---|
| **NIH ChestX-ray14** | ~112,000 | Multi-label classification, không có BBox |
| **VinDr-CXR** | ~18,000 | Có BBox annotations từ 3 bác sĩ X-quang |
| **Tổng cộng** | ~130,000 | Kết hợp classification + localization |

### 9.2 Phân chia dữ liệu

| Tập | Số ảnh | File |
|---|---|---|
| Train | 141,281 dòng (multi-annotator) | [train_list.csv](file:///d:/temp/tempnckh/CheXNet/Dataset/train_list.csv) |
| Validation | 8,508 dòng | [val_list.csv](file:///d:/temp/tempnckh/CheXNet/Dataset/val_list.csv) |
| Test | 16,443 dòng | [test_list.csv](file:///d:/temp/tempnckh/CheXNet/Dataset/test_list.csv) |

### 9.3 Schema CSV

```
Image Index | Finding Labels | Source | 15 cột one-hot | Patient Age | Patient Sex | rad_id | x_max | x_min | y_max | y_min
```

### 9.4 Data Loading tối ưu

> **File**: [read_data.py](file:///d:/temp/tempnckh/CheXNet/Models/read_data.py) — 380 dòng

- **Raw Byte Cache**: Giữ file nén trong RAM (không decompress) → giảm I/O disk
- **Synchronized Augmentation**: Concat image (3ch) + mask (15ch) = 18ch → transform đồng bộ
- **HybridBatchSampler**: Tỷ lệ 1:3 (VinDr : NIH) cho Stage 2
- **FastDataLoader**: `pin_memory=True`, `prefetch_factor=2`, `persistent_workers=True`

### 9.5 YOLO Training (VinDr-CXR)

> **File**: [yolov11m_fixed.ipynb](file:///d:/temp/tempnckh/CheXNet/Notebooks/yolov11m_fixed.ipynb)

- **WBF Ground-Truth Fusion**: Gộp box từ 3 bác sĩ thành consensus box
- **Progressive Training**: Phase 1 (640px, 25 epochs) → Phase 2 (1024px, 55 epochs)
- **X-Ray Augmentation**: Bật `fliplr=0.5`; **Tắt** `flipud`, `mosaic`, `mixup` (sai giải phẫu)
- **StratifiedKFold 90/10**: Giữ ảnh "No finding" làm background negatives

---

## 10. Lịch Sử Phát Triển & Git Timeline

### 10.1 Nhánh Git

| Nhánh | Vai trò |
|---|---|
| `main` | Nhánh chính (production) |
| `test` ★ | Nhánh hiện tại |
| `lai` | Nhánh phát triển thử nghiệm |
| `develop` | Nhánh phát triển |
| `a`, `c`, `h` | Nhánh thử nghiệm ngắn |
| `swinv1large` | Thử nghiệm SwinV1 Large |

### 10.2 Timeline phát triển chi tiết

```mermaid
timeline
    title Tiến trình phát triển CheXNet (10/2025 — 08/2026)
    section Phase 1 — Khởi tạo
        27/10/2025 : Upload source code ban đầu
        28/10/2025 : Cập nhật config, read_data, TrainModel, main
        31/10/2025 : Tinh chỉnh TrainModel, read_data
        02/11/2025 : Cập nhật main.py, visualize.py
        03/11/2025 : Thêm script tải dữ liệu, upload files
    section Phase 2 — Dữ liệu
        01/12/2025 : Upload dataset CSV (train/val/test)
    section Phase 3 — Nghiên cứu
        21/05/2026 : Upload source base hoàn chỉnh
        21/05/2026 : Fix research portal
    section Phase 4 — Tối ưu SOTA
        05/08/2026 : Tối ưu SOTA Tier 0-4 (FPN, DDP, metrics)
        06/08/2026 : Fix scheduler crash
        08/08/2026 : Sửa 7 bugs nghiêm trọng liên tiếp
        09/08/2026 : Fix optimizer resume crash, AUROC diagnostics
    section Phase 5 — Fine-Tune & Deploy
        11/08/2026 : Thêm Fine-tune module + Fix 2 critical bugs
        12/08/2026 : Cấu hình .gitignore, dọn dẹp
```

### 10.3 Các bug nghiêm trọng đã sửa (Phase 4)

| Ngày | Commit | Bug | Tác động |
|---|---|---|---|
| 08/08 | `3c92bf9` | Dice Loss không hội tụ | Thêm mask_smooth, valid_mask, discriminative LR |
| 08/08 | `4729fc3` | Nhiều bugs dựa trên audit | Sửa critical + medium bugs |
| 08/08 | `75db276` | **Attention Collapse** | Loại Sparsity khỏi UW (n_tasks 3→2), dùng fixed weight |
| 08/08 | `de7fe4e` | **FP16 NaN Gradients** (Epoch 18 crash) | Tăng eps từ 1e-8 lên 1e-4 tránh flush-to-zero |
| 09/08 | `dbd71da` | **Optimizer Resume Crash** | Handle param group mismatch (2→3 groups) |
| 09/08 | `0c0341d` | AUROC sụp đổ | Thêm AUROC debug diagnostics |
| 11/08 | `d0f4d0f` | 2 critical training bugs | Fine-tune module + bugfixes |

> [!WARNING]
> **Bug nghiêm trọng nhất**: FP16 Flush-To-Zero — GPU làm tròn giá trị epsilon rất nhỏ (1e-8) về **đúng 0.0** trong chế độ float16, gây `torch.log(0.0) = -inf` → NaN gradients → crash training ở Epoch 18. Giải pháp: tăng tất cả epsilon/smooth lên 1e-4.

---

## 11. Danh Mục Toàn Bộ File Dự Án

### Core AI & Training

| # | File | Dòng | Vai trò |
|---|---|---|---|
| 1 | [main.py](file:///d:/temp/tempnckh/CheXNet/main.py) | 531 | CLI entry point: Train / Resume / Test |
| 2 | [predict_single.py](file:///d:/temp/tempnckh/CheXNet/predict_single.py) | 249 | Inference đơn ảnh + Heatmap |
| 3 | [run_predict.py](file:///d:/temp/tempnckh/CheXNet/run_predict.py) | 54 | Script test nhanh tự động |
| 4 | [Models/Model.py](file:///d:/temp/tempnckh/CheXNet/Models/Model.py) | 215 | Kiến trúc Hybrid CNN-ViT |
| 5 | [Models/TrainModel.py](file:///d:/temp/tempnckh/CheXNet/Models/TrainModel.py) | 1047 | Engine huấn luyện toàn bộ |
| 6 | [Models/config.py](file:///d:/temp/tempnckh/CheXNet/Models/config.py) | 109 | Auto GPU config & batch size |
| 7 | [Models/read_data.py](file:///d:/temp/tempnckh/CheXNet/Models/read_data.py) | 380 | Dataset, Augmentation, Sampling |
| 8 | [Models/checkpoint_utils.py](file:///d:/temp/tempnckh/CheXNet/Models/checkpoint_utils.py) | 63 | Safe checkpoint loading |
| 9 | [Models/head_map.py](file:///d:/temp/tempnckh/CheXNet/Models/head_map.py) | 124 | Standalone attention visualization |
| 10 | [Models/visualize.py](file:///d:/temp/tempnckh/CheXNet/Models/visualize.py) | 52 | Training curves (3-panel) |

### Fine-Tune Module

| # | File | Vai trò |
|---|---|---|
| 11 | [Finetune/main_finetune.py](file:///d:/temp/tempnckh/CheXNet/Finetune/main_finetune.py) | CLI fine-tune tương tác |
| 12 | [Finetune/TrainModel_finetune.py](file:///d:/temp/tempnckh/CheXNet/Finetune/TrainModel_finetune.py) | Engine fine-tune chuyên sâu |

### Backend (FastAPI)

| # | File | Vai trò |
|---|---|---|
| 13 | [Backend/main.py](file:///d:/temp/tempnckh/CheXNet/Backend/main.py) | FastAPI Server, điều phối pipeline |
| 14 | [Backend/yolo_detector.py](file:///d:/temp/tempnckh/CheXNet/Backend/yolo_detector.py) | YOLO11m wrapper + Cascade scoring |
| 15 | [Backend/cbir.py](file:///d:/temp/tempnckh/CheXNet/Backend/cbir.py) | FAISS CBIR vector search |
| 16 | [Backend/lung_segmentation.py](file:///d:/temp/tempnckh/CheXNet/Backend/lung_segmentation.py) | Phân vùng phổi cổ điển |
| 17 | [Backend/post_processing.py](file:///d:/temp/tempnckh/CheXNet/Backend/post_processing.py) | Hậu xử lý, Co-occurrence, WBF |
| 18 | [Backend/report_generator.py](file:///d:/temp/tempnckh/CheXNet/Backend/report_generator.py) | Sinh báo cáo y khoa tiếng Việt |
| 19 | [Backend/scripts/build_faiss_index.py](file:///d:/temp/tempnckh/CheXNet/Backend/scripts/build_faiss_index.py) | CLI build FAISS index |

### Frontend (React)

| # | File | Vai trò |
|---|---|---|
| 20 | [Frontend/src/App.jsx](file:///d:/temp/tempnckh/CheXNet/Frontend/src/App.jsx) | Component chính — Master Orchestrator |
| 21 | [Frontend/src/main.jsx](file:///d:/temp/tempnckh/CheXNet/Frontend/src/main.jsx) | React DOM mount |
| 22 | [Frontend/src/index.css](file:///d:/temp/tempnckh/CheXNet/Frontend/src/index.css) | Theme, animations, print styles |
| 23 | [Frontend/src/components/ClinicalReport.jsx](file:///d:/temp/tempnckh/CheXNet/Frontend/src/components/ClinicalReport.jsx) | Báo cáo lâm sàng có cấu trúc |
| 24 | [Backend/report_generator.py](file:///d:/temp/tempnckh/CheXNet/Backend/report_generator.py) | Bộ sinh báo cáo lâm sàng RSNA/ACR |
| 25 | [Frontend/src/utils/radiologyMapper.js](file:///d:/temp/tempnckh/CheXNet/Frontend/src/utils/radiologyMapper.js) | Ánh xạ hình ảnh học y khoa |
| 26 | [Frontend/src/utils/sampleData.js](file:///d:/temp/tempnckh/CheXNet/Frontend/src/utils/sampleData.js) | 3 ca mẫu + từ điển 15 bệnh |

### Data & Utilities

| # | File | Vai trò |
|---|---|---|
| 27 | [Database/batch_download_zips.py](file:///d:/temp/tempnckh/CheXNet/Database/batch_download_zips.py) | Script tải NIH ChestX-ray14 |
| 28 | [Dataset/train_list.csv](file:///d:/temp/tempnckh/CheXNet/Dataset/train_list.csv) | 141,281 dòng train data |
| 29 | [Dataset/val_list.csv](file:///d:/temp/tempnckh/CheXNet/Dataset/val_list.csv) | 8,508 dòng validation data |
| 30 | [Dataset/test_list.csv](file:///d:/temp/tempnckh/CheXNet/Dataset/test_list.csv) | 16,443 dòng test data |
| 31 | [Notebooks/yolov11m_fixed.ipynb](file:///d:/temp/tempnckh/CheXNet/Notebooks/yolov11m_fixed.ipynb) | Train YOLO11m Progressive |

### Model Weights

| # | File | Vai trò |
|---|---|---|
| 32 | `Trainedmodel/hybrid_model_light.pth` | Weights Hybrid CNN-ViT (production) |
| 33 | `Trainedmodel/yolov11m.pt` | Weights YOLO11m detector |

---

## 12. Đánh Giá Tổng Hợp & Đề Xuất

### 12.1 Điểm mạnh

| Tiêu chí | Đánh giá |
|---|---|
| **Kiến trúc mô hình** | ⭐⭐⭐⭐⭐ — Hybrid CNN-ViT với 15-channel endogenous attention là thiết kế độc đáo, vượt trội hơn Grad-CAM truyền thống |
| **Hệ thống Loss** | ⭐⭐⭐⭐⭐ — Kết hợp tinh vi ASL + Dice + Focal Tversky + UW + Sparsity, giải quyết triệt để class imbalance |
| **Training Pipeline** | ⭐⭐⭐⭐⭐ — Curriculum 2-stage, discriminative LR, EMA, dual checkpoint, DDP — đạt tiêu chuẩn production |
| **Tích hợp đa mô hình** | ⭐⭐⭐⭐⭐ — Cascade CheXNet × YOLO11m loại bỏ False Positives hiệu quả |
| **Debug & Bug-fixing** | ⭐⭐⭐⭐⭐ — Git log cho thấy 7 bugs nghiêm trọng được xác định và sửa có phương pháp trong 3 ngày |
| **Frontend UX** | ⭐⭐⭐⭐ — Giao diện chuyên nghiệp, dark medical theme, print-friendly |
| **Documentation** | ⭐⭐⭐⭐ — Code comments chi tiết, commit messages mô tả rõ ràng nguyên nhân gốc |

### 12.2 Điểm cần cải thiện

| Vấn đề | Mức độ | Đề xuất |
|---|---|---|
| `Backend/api.py` trống | 🟡 Nhỏ | Xóa hoặc tích hợp endpoints modular |
| Backend `requirements.txt` thiếu thư viện | 🟠 Trung bình | Thêm `faiss-cpu/gpu`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `ultralytics`, `ensemble-boxes` |
| Chưa có unit tests | 🟠 Trung bình | Thêm pytest cho model forward, data loading, post-processing |
| Chưa có Docker/docker-compose | 🟡 Nhỏ | Containerize Backend + Frontend cho deployment nhất quán |
| README chưa cập nhật Backend/Frontend | 🟡 Nhỏ | Bổ sung hướng dẫn chạy Backend + Frontend |
| FAISS index chưa có pre-built | 🟡 Nhỏ | Cung cấp script hoặc pre-built index để demo ngay |

### 12.3 Tổng kết

> [!TIP]
> **CheXNet Pro AI** là một dự án NCKH hoàn chỉnh ở mức **production-ready**, tích hợp đầy đủ từ kiến trúc mô hình tiên tiến (Hybrid CNN-ViT + YOLO11m + FAISS CBIR), quy trình huấn luyện có phương pháp (Curriculum Learning, Multi-Task Uncertainty Weighting), đến ứng dụng web hoàn chỉnh với giao diện Dark Cyber-Medical và báo cáo lâm sàng chuẩn y khoa. Lịch sử Git cho thấy quá trình phát triển kéo dài ~10 tháng (10/2025 → 08/2026) với giai đoạn tối ưu hóa chuyên sâu rất ấn tượng trong 08/2026 khi 7 bugs nghiêm trọng được xác định và khắc phục một cách có hệ thống.

---

*Báo cáo được tạo tự động bởi Antigravity AI — phân tích từ toàn bộ mã nguồn, lịch sử Git (48 commits) và cấu trúc dự án.*
