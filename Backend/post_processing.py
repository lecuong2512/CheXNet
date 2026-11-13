import numpy as np
import cv2
from typing import List, Dict, Tuple, Any

try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    weighted_boxes_fusion = None

# Danh sách các bệnh lý
CLASS_NAMES = [
    'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Mức độ ưu tiên lâm sàng của từng bệnh lý
CLINICAL_PRIORITY = {
    'Pneumothorax': 'critical',
    'Pneumonia': 'critical',
    'Cardiomegaly': 'high',
    'Effusion': 'high',
    'Mass': 'high',
    'Nodule': 'high',
    'Consolidation': 'high',
    'Infiltration': 'moderate',
    'Atelectasis': 'moderate',
    'Edema': 'moderate',
    'Emphysema': 'low',
    'Fibrosis': 'low',
    'Pleural_Thickening': 'low',
    'Hernia': 'low',
    'No Finding': 'low'
}

# Ngưỡng (threshold) dựa trên mức độ ưu tiên
TARGET_RECALL = {
    'critical': 0.25,
    'high': 0.35,
    'moderate': 0.40,
    'low': 0.45
}

def get_clinical_thresholds() -> Dict[str, float]:
    """
    Lấy ngưỡng lâm sàng cho từng loại bệnh dựa trên mức độ ưu tiên.
    Ngưỡng được điều chỉnh để đạt recall mục tiêu (ví dụ: critical -> ngưỡng 0.25).
    
    Returns:
        Dict[str, float]: Dictionary ánh xạ tên bệnh với ngưỡng tương ứng.
    """
    return {disease: TARGET_RECALL[priority] for disease, priority in CLINICAL_PRIORITY.items()}

# Khởi tạo ma trận đồng xuất hiện (Co-occurrence matrix)
# Tính từ dữ liệu thực: 107,880 ảnh huấn luyện (NIH + VinDr-CXR) bằng np.corrcoef
CO_OCCURRENCE_MATRIX = np.array([
    [ 1.0,  -0.36, -0.229,-0.402,-0.496,-0.25, -0.272,-0.12, -0.237,-0.228,-0.153,-0.161,-0.182,-0.237,-0.047],
    [-0.36,  1.0,  -0.009, 0.177, 0.106, 0.027,-0.001, 0.032, 0.041, 0.112, 0.003, 0.039, 0.007, 0.018, 0.009],
    [-0.229,-0.009, 1.0,   0.102,-0.007,-0.014, 0.01, -0.007,-0.031, 0.015, 0.01, -0.017, 0.114, 0.151,-0.002],
    [-0.402, 0.177, 0.102, 1.0,   0.132, 0.076, 0.03,  0.026, 0.057, 0.113, 0.065, 0.014, 0.046, 0.132,-0.002],
    [-0.496, 0.106,-0.007, 0.132, 1.0,   0.022, 0.045, 0.079, 0.009, 0.058, 0.097, 0.005, 0.031, 0.022,-0.0  ],
    [-0.25,  0.027,-0.014, 0.076, 0.022, 1.0,   0.101,-0.0,   0.034, 0.085, 0.004, 0.028, 0.011, 0.056, 0.015],
    [-0.272,-0.001, 0.01,  0.03,  0.045, 0.101, 1.0,  -0.002, 0.01,  0.039, 0.001,-0.004, 0.066, 0.076,-0.005],
    [-0.12,  0.032,-0.007, 0.026, 0.079,-0.0,  -0.002, 1.0,  -0.007, 0.027, 0.174,-0.002,-0.012,-0.006, 0.002],
    [-0.237, 0.041,-0.031, 0.057, 0.009, 0.034, 0.01, -0.007, 1.0,   0.006,-0.02,  0.184,-0.007, 0.022, 0.001],
    [-0.228, 0.112, 0.015, 0.113, 0.058, 0.085, 0.039, 0.027, 0.006, 1.0,   0.023, 0.0,   0.032, 0.045,-0.004],
    [-0.153, 0.003, 0.01,  0.065, 0.097, 0.004, 0.001, 0.174,-0.02,  0.023, 1.0,  -0.009,-0.019,-0.011,-0.001],
    [-0.161, 0.039,-0.017, 0.014, 0.005, 0.028,-0.004,-0.002, 0.184, 0.0,  -0.009, 1.0,  -0.011, 0.009, 0.0  ],
    [-0.182, 0.007, 0.114, 0.046, 0.031, 0.011, 0.066,-0.012,-0.007, 0.032,-0.019,-0.011, 1.0,   0.245, 0.003],
    [-0.237, 0.018, 0.151, 0.132, 0.022, 0.056, 0.076,-0.006, 0.022, 0.045,-0.011, 0.009, 0.245, 1.0,  -0.002],
    [-0.047, 0.009,-0.002,-0.002,-0.0,   0.015,-0.005, 0.002, 0.001,-0.004,-0.001, 0.0,   0.003,-0.002, 1.0  ],
], dtype=np.float32)

def check_co_occurrence(probs: np.ndarray, threshold: float = 0.4) -> List[Dict[str, Any]]:
    """
    Kiểm tra sự đồng xuất hiện của các bệnh dựa trên xác suất và ma trận tương quan.
    
    Args:
        probs (np.ndarray): Mảng numpy chứa xác suất dự đoán cho 15 nhãn.
        threshold (float): Ngưỡng xác suất để xem xét bệnh có xuất hiện hay không.
        
    Returns:
        List[Dict]: Danh sách các dict chứa thông tin đồng xuất hiện và ghi chú lâm sàng.
    """
    results = []
    clinical_notes_mapping = {
        ('Cardiomegaly', 'Effusion'): 'Gợi ý Suy tim sung huyết',
        ('Edema', 'Effusion'): 'Gợi ý Suy tim sung huyết hoặc Phù phổi',
        ('Edema', 'Cardiomegaly'): 'Gợi ý Suy tim sung huyết',
        ('Pneumonia', 'Consolidation'): 'Viêm phổi nghiêm trọng hoặc đông đặc nhu mô phổi',
        ('Pneumonia', 'Infiltration'): 'Viêm phổi kèm thâm nhiễm',
        ('Mass', 'Atelectasis'): 'Khối u có thể gây xẹp phổi',
        ('Mass', 'Nodule'): 'Đa tổn thương dạng khối/nốt',
        ('Fibrosis', 'Pleural_Thickening'): 'Dấu hiệu tổn thương mạn tính/xơ hóa'
    }

    n = len(CLASS_NAMES)
    for i in range(n):
        for j in range(i + 1, n):
            if probs[i] > threshold and probs[j] > threshold:
                corr = CO_OCCURRENCE_MATRIX[i, j]
                if corr > 0.08:  # Ngưỡng hạ thấp phù hợp dữ liệu thực (r=0.10-0.25)
                    d1, d2 = CLASS_NAMES[i], CLASS_NAMES[j]
                    note = clinical_notes_mapping.get((d1, d2)) or clinical_notes_mapping.get((d2, d1)) or 'Cần lưu ý lâm sàng'
                    results.append({
                        'diseases': [d1, d2],
                        'correlation': float(corr),
                        'clinical_note': note
                    })
    return results

def cam_to_bboxes(attention_map_2d: np.ndarray, img_size: int, threshold_method: str = 'otsu', min_area_ratio: float = 0.002) -> List[Dict[str, Any]]:
    """
    Chuyển đổi bản đồ chú ý (attention map) thành danh sách các bounding boxes.
    
    Args:
        attention_map_2d (np.ndarray): Bản đồ chú ý kích thước [H_feat, W_feat] giá trị [0, 1].
        img_size (int): Kích thước ảnh gốc để quy đổi tọa độ.
        threshold_method (str): Phương pháp nhị phân hóa ('otsu' hoặc 'adaptive').
        min_area_ratio (float): Tỷ lệ diện tích tối thiểu của bounding box so với diện tích ảnh.
        
    Returns:
        List[Dict]: Danh sách bboxes chứa tọa độ và độ tin cậy. Tối đa 5 bboxes, sắp xếp giảm dần theo confidence.
    """
    # Thay đổi kích thước map về kích thước ảnh
    map_resized = cv2.resize(attention_map_2d, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    
    # Chuẩn hóa về [0, 255]
    map_normalized = np.clip(map_resized * 255.0, 0, 255).astype(np.uint8)
    
    if threshold_method == 'otsu':
        _, binary = cv2.threshold(map_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == 'adaptive':
        binary = cv2.adaptiveThreshold(map_normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        _, binary = cv2.threshold(map_normalized, 127, 255, cv2.THRESH_BINARY)
        
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = min_area_ratio * (img_size ** 2)
    bboxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Tính độ tin cậy (confidence) của bbox: giá trị kích hoạt trung bình bên trong contour
        mask = np.zeros_like(map_normalized)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        mean_act = cv2.mean(map_resized, mask=mask)[0]
        
        bboxes.append({
            'x1': int(x),
            'y1': int(y),
            'x2': int(x + w),
            'y2': int(y + h),
            'confidence': float(mean_act)
        })
        
    # Sắp xếp bboxes theo confidence giảm dần và giới hạn 5 bounding boxes
    bboxes.sort(key=lambda b: b['confidence'], reverse=True)
    return bboxes[:5]

def compute_attention_quality(att_map_2d: np.ndarray) -> Dict[str, float]:
    """
    Tính toán chỉ số chất lượng của bản đồ chú ý (attention map).
    
    Args:
        att_map_2d (np.ndarray): Bản đồ chú ý kích thước [H, W], giá trị [0, 1].
        
    Returns:
        Dict[str, float]: Từ điển chứa concentration, contrast, sharpness và quality_score.
    """
    att_map_2d = np.clip(att_map_2d, 0, 1)
    
    # 1. Concentration (Độ tập trung): Tỷ lệ pixel có kích hoạt cao
    high_act_mask = att_map_2d > 0.3
    concentration = float(np.mean(high_act_mask))
    
    # 2. Contrast (Độ tương phản)
    if np.any(high_act_mask) and np.any(~high_act_mask):
        mean_high = np.mean(att_map_2d[high_act_mask])
        mean_low = np.mean(att_map_2d[~high_act_mask])
        contrast = float(mean_high - mean_low)
    else:
        contrast = 0.0
        
    # 3. Sharpness (Độ sắc nét): dựa trên entropy
    eps = 1e-7
    p = float(np.mean(att_map_2d))
    if 0 < p < 1:
        entropy = -p * np.log(p + eps) - (1 - p) * np.log(1 - p + eps)
    else:
        entropy = 0.0
    sharpness = max(0.0, 1.0 - entropy / 0.693)
    
    # 4. Quality score: Tổng hợp các tiêu chí
    concentration_score = 1.0 - min(concentration / 0.5, 1.0)
    quality_score = 0.3 * concentration_score + 0.4 * contrast + 0.3 * sharpness
    
    return {
        'concentration': float(np.clip(concentration, 0, 1)),
        'contrast': float(np.clip(contrast, 0, 1)),
        'sharpness': float(np.clip(sharpness, 0, 1)),
        'quality_score': float(np.clip(quality_score, 0, 1))
    }

def _iou(box1: List[float], box2: List[float]) -> float:
    """Tính Intersection over Union (IoU) của 2 bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def _simple_nms(boxes: List[List[float]], scores: List[float], labels: List[int], iou_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: thuật toán NMS đơn giản nếu không có ensemble_boxes."""
    if not boxes:
        return np.array([]), np.array([]), np.array([])
        
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        rem_indices = []
        for i in indices[1:]:
            if _iou(boxes[current], boxes[i]) <= iou_thr:
                rem_indices.append(i)
        indices = rem_indices
        
    keep_boxes = np.array([boxes[i] for i in keep])
    keep_scores = np.array([scores[i] for i in keep])
    keep_labels = np.array([labels[i] for i in keep])
    
    return keep_boxes, keep_scores, keep_labels

def apply_wbf(boxes_list: List[List[List[float]]], scores_list: List[List[float]], labels_list: List[List[int]], img_size: int, iou_thr: float = 0.4, skip_box_thr: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kết hợp các bounding boxes sử dụng Weighted Boxes Fusion (WBF) hoặc Non-Maximum Suppression (NMS).
    
    Args:
        boxes_list (List[List[List[float]]]): Danh sách các bboxes từ các nguồn khác nhau.
        scores_list (List[List[float]]): Điểm số (confidence) tương ứng.
        labels_list (List[List[int]]): Nhãn bệnh tương ứng.
        img_size (int): Kích thước ảnh thực tế (để chuẩn hóa và giải chuẩn hóa [0, 1]).
        iou_thr (float): Ngưỡng IoU.
        skip_box_thr (float): Ngưỡng bỏ qua bboxes có score thấp.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: bboxes_array, scores_array, labels_array.
    """
    # Chuẩn hóa tọa độ về [0, 1]
    norm_boxes_list = []
    for boxes in boxes_list:
        norm_boxes = []
        for box in boxes:
            norm_boxes.append([
                max(0.0, min(1.0, box[0] / img_size)),
                max(0.0, min(1.0, box[1] / img_size)),
                max(0.0, min(1.0, box[2] / img_size)),
                max(0.0, min(1.0, box[3] / img_size))
            ])
        norm_boxes_list.append(norm_boxes)
        
    if weighted_boxes_fusion is not None:
        boxes, scores, labels = weighted_boxes_fusion(
            norm_boxes_list, scores_list, labels_list, 
            weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
    else:
        # Fallback to simple NMS if ensemble_boxes is not installed
        flat_boxes = []
        flat_scores = []
        flat_labels = []
        
        for bs, ss, ls in zip(norm_boxes_list, scores_list, labels_list):
            for b, s, l in zip(bs, ss, ls):
                if s > skip_box_thr:
                    flat_boxes.append(b)
                    flat_scores.append(s)
                    flat_labels.append(l)
                    
        boxes, scores, labels = _simple_nms(flat_boxes, flat_scores, flat_labels, iou_thr)

    # Giải chuẩn hóa về kích thước pixel ảnh
    if len(boxes) > 0:
        boxes = np.array(boxes) * img_size
    else:
        boxes = np.array([])
        
    return np.array(boxes), np.array(scores), np.array(labels).astype(int)
