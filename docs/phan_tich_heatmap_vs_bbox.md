# 🔍 Tại Sao SwinV2 Chỉ Học Vẽ "Heatmap Ảo" Mà Không Vẽ Được BBox?

## Tóm tắt ngắn gọn

> [!IMPORTANT]
> **Nguyên nhân gốc rễ**: Kiến trúc `HybridCNNViTModel` hiện tại **KHÔNG có detection head** (không có nhánh regression tọa độ `[x, y, w, h]`). Nó chỉ có **attention head** — một module segmentation đơn giản sinh ra **soft heatmap** (giá trị liên tục 0→1 trên mỗi pixel), KHÔNG phải bounding box. Đây là **2 bài toán hoàn toàn khác nhau** về mặt kiến trúc mạng.

---

## 1. So Sánh: Attention Map vs Bounding Box Detection

### Attention Map (cái model hiện tại làm)

```
Input Image → CNN+ViT Features → Conv 3×3 → Conv 1×1 → Sigmoid
                                                         ↓
                                              [B, 15, H, W] float ∈ [0, 1]
                                              (mỗi pixel = "xác suất chú ý")
```

**Kết quả**: Một "bức tranh nhiệt" mờ ảo, mỗi pixel có giá trị từ 0 (không quan tâm) đến 1 (rất quan tâm). **Không có tọa độ cụ thể nào**.

### Bounding Box Detection (cái bạn muốn)

```
Input Image → Backbone Features → Detection Head:
                                    ├── Classification Branch → "Đây là bệnh gì?"
                                    ├── Regression Branch    → "Tọa độ [x, y, w, h]?"
                                    └── Objectness Branch    → "Có object không?"
                                                         ↓
                                              List of Boxes: [(class, x, y, w, h, conf), ...]
```

**Kết quả**: Danh sách các hộp chữ nhật chính xác với tọa độ pixel cụ thể.

### Bảng so sánh cốt lõi

| Tiêu chí | Attention Map (hiện tại) | BBox Detection (mong muốn) |
|---|---|---|
| **Đầu ra** | Tensor 2D liên tục `[15, H, W]` | Danh sách `[(x, y, w, h, class, conf)]` |
| **Thông tin vị trí** | Mờ, lan tỏa, không có biên rõ | Hộp chữ nhật chính xác |
| **Cần anchor/prior boxes** | ❌ Không | ✅ Bắt buộc |
| **Cần regression head** | ❌ Không | ✅ Bắt buộc (`nn.Linear → [x,y,w,h]`) |
| **Cần NMS** | ❌ Không | ✅ Bắt buộc |
| **Loss function** | Dice / BCE (per-pixel) | Smooth L1 / GIoU / CIoU (per-box) |
| **Supervision signal** | Mask 2D (fill toàn bộ box) | Tọa độ box `(x₁, y₁, x₂, y₂)` |

---

## 2. Phân Tích Code — 5 Điểm Thiếu Trong Model.py

### Điểm 1: Attention Head chỉ là Segmentation Head đơn giản

```python
# Model.py dòng 77-83
self.attention_head = nn.Sequential(
    nn.Conv2d(swin_dim, 256, kernel_size=3, padding=1),  # Giảm kênh
    nn.BatchNorm2d(256),
    nn.GELU(),
    nn.Conv2d(256, num_classes, kernel_size=1),  # 256 → 15 kênh
    nn.Sigmoid()  # ← Output là XÁC SUẤT per-pixel, KHÔNG phải tọa độ
)
```

> **Vấn đề**: Output qua `Sigmoid()` cho ra giá trị `[0, 1]` trên mỗi pixel — đây là **semantic segmentation**, không phải detection. Muốn vẽ bbox cần thêm một nhánh **regression head** xuất ra `[x, y, w, h]` cho từng object.

### Điểm 2: Không có Anchor Boxes / Prior Boxes

Các detector (YOLO, Faster R-CNN, SSD) đều cần hệ thống **anchor boxes** — các hộp tham chiếu đặt sẵn trên feature map, model chỉ cần **tinh chỉnh offset** so với anchor. Model hiện tại **không có bất kỳ anchor nào**.

### Điểm 3: Không có Box Regression Loss

```python
# TrainModel.py — DiceLoss (dòng 163-236)
# Loss được tính PER-PIXEL giữa attention map và filled mask
intersection = (preds_flat * targets_flat).sum()
dice = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
```

> **Vấn đề**: Dice Loss so sánh **pixel-by-pixel** giữa attention map và mask GT. Nó tối ưu cho **overlap diện tích**, KHÔNG tối ưu cho **vị trí + kích thước hộp**. Muốn vẽ bbox cần loss như `Smooth L1`, `GIoU Loss`, hoặc `CIoU Loss` — so sánh **tọa độ box** trực tiếp.

### Điểm 4: GT Mask là "fill toàn bộ box" — thông tin bị mất

```python
# read_data.py — Rasterize bbox thành mask
# BBox (x₁, y₁, x₂, y₂) → Tô kín toàn bộ hình chữ nhật trên mask 2D
draw.rectangle([x_min, y_min, x_max, y_max], fill=1)
```

> **Vấn đề**: Tọa độ chính xác `(x₁, y₁, x₂, y₂)` bị **phá hủy** khi chuyển thành filled mask. Model chỉ thấy "vùng này sáng, vùng kia tối" — không bao giờ học được khái niệm "cạnh trái ở pixel 120, cạnh phải ở pixel 340".

### Điểm 5: Không có NMS (Non-Maximum Suppression)

Detector cần NMS để loại bỏ các box trùng lặp. Attention map là tensor liên tục — không có khái niệm "nhiều box" hay "box trùng".

---

## 3. Tại Sao Thiết Kế Lại Thành Như Vậy?

### 3.1 Mục đích THỰC SỰ của Attention Head

Nhìn vào code [Model.py dòng 164-191](file:///d:/temp/tempnckh/CheXNet/Models/Model.py#L164-L191):

```python
# Gated Residual Attention — mục đích CHÍNH của attention map
highlighted_group = group_features * group_attention  # Feature × Attention
gated_group = group_features + gate_alpha * highlighted_group  # Residual
```

> [!NOTE]
> Attention map được thiết kế để **tăng cường classification** (nhân với feature rồi cộng residual), **KHÔNG phải để vẽ bbox**. Nó đóng vai trò "bộ lọc không gian" — khuếch đại features ở vùng tổn thương, triệt tiêu features ở vùng bình thường → giúp classifier chính xác hơn.

### 3.2 Tại sao attention map bị "mờ ảo"?

| Nguyên nhân | Giải thích |
|---|---|
| **Resolution thấp** | SwinV2 xử lý ở resolution ~12×12 hoặc 24×24 (sau FPN). BBox thực tế cần precision ở resolution gốc (512×512 hoặc 1024×1024) |
| **Sigmoid output** | Giá trị liên tục `[0, 1]` tự nhiên sẽ "lan" ra xung quanh — không có cơ chế nào ép nó tạo biên sắc nét |
| **Dice Loss tối ưu overlap** | Dice chỉ cần "đủ nhiều pixel đúng" — không phạt mạnh khi biên bị lệch vài pixel |
| **Bilinear interpolation** | Khi upsample từ 12×12 lên 384×384, nội suy tạo ra gradient mượt → heatmap luôn mờ |

### 3.3 Dòng thời gian quyết định

```
Ý tưởng ban đầu: "SwinV2 vừa classify vừa vẽ bbox"
        ↓
Triển khai thực tế: Attention Head chỉ sinh soft mask (segmentation)
        ↓
Phát hiện: Attention map quá mờ, không dùng làm bbox lâm sàng được
        ↓
Giải pháp: Train YOLO11m riêng để vẽ bbox chính xác
        ↓
Kết hợp: Cascade Score = P_classifier × Conf_YOLO
```

---

## 4. Nếu Muốn SwinV2 Vừa Classify Vừa Detect — Cần Làm Gì?

### Phương án A: DETR-Style (Khuyến nghị cho nghiên cứu)

Thêm **Transformer Decoder + Hungarian Matching** lên trên SwinV2 features:

```
SwinV2 Features → Transformer Decoder (learnable queries)
                    ├── Class Head  → "Bệnh gì?"
                    └── Box Head    → "[x, y, w, h]" (normalize)
                    
Loss = λ₁·CE(class) + λ₂·L1(box) + λ₃·GIoU(box)
Matching = Hungarian Algorithm (1-to-1 assignment)
```

> **Ưu điểm**: End-to-end, không cần anchor, không cần NMS  
> **Nhược điểm**: Khó train, cần nhiều data, hội tụ chậm

### Phương án B: Faster R-CNN Hybrid

Gắn **Region Proposal Network (RPN) + ROI Head** lên SwinV2 FPN features:

```
SwinV2 + FPN Features → RPN → Proposals → ROI Align → 
                                            ├── Classifier
                                            └── Box Regressor
```

### Phương án C: Multi-Task Head (Thực tế nhất)

Giữ nguyên kiến trúc hiện tại, **thêm 1 nhánh box regression song song** với attention head:

```python
# === THÊM VÀO Model.py ===
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=9):
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, num_classes * num_anchors, 1)
        self.reg_head = nn.Conv2d(in_channels, 4 * num_anchors, 1)  # [dx, dy, dw, dh]
    
    def forward(self, features):
        cls_pred = self.cls_head(features)   # [B, 15*9, H, W]
        box_pred = self.reg_head(features)   # [B, 4*9, H, W]
        return cls_pred, box_pred
```

Và thêm loss tương ứng:
```python
# Box Regression Loss
box_loss = F.smooth_l1_loss(pred_boxes, target_boxes)
# hoặc GIoU Loss
giou_loss = 1 - generalized_iou(pred_boxes, target_boxes)
```

---

## 5. Tại Sao YOLO Vẽ BBox Tốt Hơn?

| Yếu tố | Attention Head (SwinV2) | YOLO11m |
|---|---|---|
| **Kiến trúc** | Segmentation head (per-pixel) | Detection head (per-box) |
| **Anchor system** | ❌ Không có | ✅ 9 anchors/cell tối ưu cho X-ray |
| **Box regression** | ❌ Không có | ✅ Trực tiếp predict `[x, y, w, h]` |
| **Loss function** | Dice (pixel overlap) | CIoU (box geometry) |
| **NMS** | ❌ Không có | ✅ Loại box trùng |
| **Resolution** | 12×12 → 24×24 (rất thấp) | 80×80 → 20×20 (multi-scale) |
| **Training data** | BBox → fill mask (mất thông tin) | BBox trực tiếp (giữ nguyên tọa độ) |

> [!TIP]
> **Kết luận**: Giải pháp hiện tại (CheXNet classify + YOLO detect + Cascade Score) thực ra là **hợp lý và hiệu quả** trong thực tế. Mỗi model làm đúng thế mạnh của mình. Tuy nhiên nếu yêu cầu nghiên cứu bắt buộc "1 model làm cả 2", thì cần refactor kiến trúc theo Phương án A (DETR) hoặc C (Multi-Task Head).
