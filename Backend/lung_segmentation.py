import numpy as np
import cv2

CLASS_NAMES = [
    'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Diseases that can appear OUTSIDE the lung field (should NOT be masked)
NON_LUNG_DISEASES = {'Cardiomegaly', 'Hernia', 'No Finding'}


def segment_lungs_classical(image_np: np.ndarray) -> np.ndarray:
    """
    Phân vùng phổi từ ảnh X-quang ngực sử dụng các phương pháp xử lý ảnh truyền thống.
    
    Args:
        image_np (np.ndarray): Ảnh RGB dưới dạng mảng numpy [H, W, 3] (uint8).
        
    Returns:
        np.ndarray: Mặt nạ nhị phân [H, W] dạng float32 (1.0 = phổi, 0.0 = ngoài phổi).
    """
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif len(image_np.shape) == 2:
        gray = image_np
    else:
        raise ValueError("Ảnh đầu vào phải có định dạng RGB hoặc Grayscale.")

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Otsu's thresholding
    # Lungs are generally dark, so we threshold to find dark regions.
    # Invert the image so lungs are bright before thresholding, or invert mask.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    H, W = gray.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    total_area = H * W

    # Keep only the 2 largest contours (left lung + right lung)
    valid_contours = []
    for cnt in contours[:2]:
        area = cv2.contourArea(cnt)
        if area >= 0.05 * total_area:
            # Optional: convex hull
            hull = cv2.convexHull(cnt)
            valid_contours.append(hull)

    if valid_contours:
        cv2.drawContours(mask, valid_contours, -1, (255), thickness=cv2.FILLED)

    return (mask / 255.0).astype(np.float32)


def get_lung_mask(image_np: np.ndarray, method: str = 'classical') -> np.ndarray:
    """
    Lấy mặt nạ phổi bằng phương pháp chỉ định.
    
    Args:
        image_np (np.ndarray): Ảnh RGB đầu vào.
        method (str): Phương pháp phân vùng ('classical' hoặc 'ones').
        
    Returns:
        np.ndarray: Mặt nạ phổi.
    """
    if method == 'classical':
        return segment_lungs_classical(image_np)
    elif method == 'ones':
        return np.ones(image_np.shape[:2], dtype=np.float32)
    else:
        raise ValueError(f"Không hỗ trợ phương pháp: {method}")


def apply_lung_mask(attention_maps: np.ndarray, lung_mask: np.ndarray, class_names: list) -> np.ndarray:
    """
    Áp dụng mặt nạ phổi lên bản đồ sự chú ý (attention maps).
    
    Args:
        attention_maps (np.ndarray): Bản đồ sự chú ý [num_classes, H_att, W_att].
        lung_mask (np.ndarray): Mặt nạ phổi [H_img, W_img].
        class_names (list): Danh sách tên các lớp bệnh.
        
    Returns:
        np.ndarray: Bản đồ sự chú ý đã được lọc.
    """
    filtered_maps = attention_maps.copy()
    
    num_classes, H_att, W_att = attention_maps.shape
    
    # Resize lung_mask to match attention_maps spatial dimensions
    if lung_mask.shape != (H_att, W_att):
        resized_mask = cv2.resize(lung_mask, (W_att, H_att), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = lung_mask

    for i, class_name in enumerate(class_names):
        if class_name not in NON_LUNG_DISEASES:
            filtered_maps[i] *= resized_mask

    return filtered_maps


def compute_lesion_area_ratio(attention_map_2d: np.ndarray, lung_mask: np.ndarray, activation_threshold: float = 0.3) -> float:
    """
    Tính tỷ lệ diện tích tổn thương trên tổng diện tích phổi.
    
    Args:
        attention_map_2d (np.ndarray): Bản đồ sự chú ý 2D [H, W].
        lung_mask (np.ndarray): Mặt nạ phổi [H, W] (0/1).
        activation_threshold (float): Ngưỡng kích hoạt tổn thương.
        
    Returns:
        float: Tỷ lệ phần trăm diện tích tổn thương (%).
    """
    H_att, W_att = attention_map_2d.shape
    
    if lung_mask.shape != (H_att, W_att):
        resized_mask = cv2.resize(lung_mask, (W_att, H_att), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = lung_mask

    lung_area = np.sum(resized_mask == 1)
    if lung_area == 0:
        return 0.0

    lesion_pixels = np.logical_and(attention_map_2d > activation_threshold, resized_mask == 1)
    lesion_area = np.sum(lesion_pixels)

    return (lesion_area / lung_area) * 100.0


def get_lesion_location(attention_map_2d: np.ndarray, lung_mask: np.ndarray, img_size: int) -> str:
    """
    Xác định vị trí giải phẫu của tổn thương trong phổi.
    
    Args:
        attention_map_2d (np.ndarray): Bản đồ sự chú ý [H, W].
        lung_mask (np.ndarray): Mặt nạ phổi [H, W].
        img_size (int): Kích thước ảnh gốc (giả định là vuông hoặc không dùng).
        
    Returns:
        str: Chuỗi mô tả vị trí (vd: 'Phổi phải, vùng giữa').
    """
    H_att, W_att = attention_map_2d.shape
    
    if lung_mask.shape != (H_att, W_att):
        resized_mask = cv2.resize(lung_mask, (W_att, H_att), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = lung_mask
        
    if np.sum(resized_mask) == 0:
        return 'Không xác định'
        
    # Split left and right half of the image
    # Note: In X-ray, left side of image is right lung and right side is left lung.
    # Here we map left image half -> Phổi phải, right image half -> Phổi trái.
    half_w = W_att // 2
    
    right_lung_mask = np.zeros_like(resized_mask)
    right_lung_mask[:, :half_w] = resized_mask[:, :half_w]  # Left of image -> Phổi phải
    
    left_lung_mask = np.zeros_like(resized_mask)
    left_lung_mask[:, half_w:] = resized_mask[:, half_w:]   # Right of image -> Phổi trái
    
    threshold = 0.3
    active_pixels = attention_map_2d > threshold
    
    right_lung_lesion = np.sum(np.logical_and(active_pixels, right_lung_mask))
    left_lung_lesion = np.sum(np.logical_and(active_pixels, left_lung_mask))
    
    total_lesion = right_lung_lesion + left_lung_lesion
    
    if total_lesion == 0:
        return 'Không phát hiện rõ ràng'
        
    location = ''
    if right_lung_lesion > 0 and left_lung_lesion > 0 and right_lung_lesion / total_lesion > 0.2 and left_lung_lesion / total_lesion > 0.2:
        location += 'Hai bên phổi'
    elif right_lung_lesion > left_lung_lesion:
        location += 'Phổi phải'
    else:
        location += 'Phổi trái'
        
    # Find vertical zones
    third_h = H_att // 3
    upper_zone = np.sum(np.logical_and(active_pixels[:third_h, :], resized_mask[:third_h, :]))
    middle_zone = np.sum(np.logical_and(active_pixels[third_h:2*third_h, :], resized_mask[third_h:2*third_h, :]))
    lower_zone = np.sum(np.logical_and(active_pixels[2*third_h:, :], resized_mask[2*third_h:, :]))
    
    zones = []
    max_zone_val = max(upper_zone, middle_zone, lower_zone)
    if max_zone_val > 0:
        if upper_zone >= 0.5 * max_zone_val:
            zones.append('vùng trên')
        if middle_zone >= 0.5 * max_zone_val:
            zones.append('vùng giữa')
        if lower_zone >= 0.5 * max_zone_val:
            zones.append('vùng dưới')
            
    if zones:
        location += ', ' + ' và '.join(zones)
        
    return location
