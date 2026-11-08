"""
YOLODetector — Module quản lý YOLO11m cho CheXNet Backend.

Chạy song song với CheXNet classifier, cung cấp:
  - Object detection (bounding boxes) cho tổn thương trên X-quang
  - Cascade scoring: Final_Score = P_classifier × Conf_YOLO
  - Vẽ boxes lên ảnh overlay

Tách riêng khỏi main.py để dễ bảo trì và test độc lập.
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[WARN] ultralytics not installed. YOLO detection disabled. Install: pip install ultralytics")


class YOLODetector:
    """
    Wrapper cho YOLO11m đã train trên VinDr-CXR.
    
    VinDr-CXR có 14 class (0-13). Một số class trùng với CheXNet 15 class,
    một số không có (Aortic enlargement, Calcification, ILD, Lung Opacity, Other lesion).
    Các class không mapping được sẽ giữ nguyên tên VinDr gốc.
    """
    
    # Tên class VinDr-CXR theo thứ tự class_id (0-13)
    VINDR_CLASS_NAMES = [
        'Aortic enlargement',   # 0
        'Atelectasis',          # 1
        'Calcification',        # 2
        'Cardiomegaly',         # 3
        'Consolidation',        # 4
        'ILD',                  # 5
        'Infiltration',         # 6
        'Lung Opacity',         # 7
        'Nodule/Mass',          # 8
        'Other lesion',         # 9
        'Pleural effusion',     # 10
        'Pleural thickening',   # 11
        'Pneumothorax',         # 12
        'Pulmonary fibrosis',   # 13
    ]
    
    # Mapping VinDr class_id → CheXNet class name (dùng cho Cascade scoring)
    # None = không có class tương ứng trong CheXNet
    VINDR_TO_CHEXNET = {
        0: None,                  # Aortic enlargement — không có trong CheXNet
        1: 'Atelectasis',         # → CheXNet idx 1
        2: None,                  # Calcification — không có trong CheXNet
        3: 'Cardiomegaly',        # → CheXNet idx 2
        4: 'Consolidation',       # → CheXNet idx 9
        5: None,                  # ILD — không có trong CheXNet
        6: 'Infiltration',        # → CheXNet idx 4
        7: None,                  # Lung Opacity — không mapping trực tiếp
        8: 'Mass',                # Nodule/Mass → CheXNet idx 5 (Mass)
        9: None,                  # Other lesion — không có trong CheXNet
        10: 'Effusion',           # Pleural effusion → CheXNet idx 3
        11: 'Pleural_Thickening', # → CheXNet idx 13
        12: 'Pneumothorax',       # → CheXNet idx 8
        13: 'Fibrosis',           # Pulmonary fibrosis → CheXNet idx 12
    }
    
    # Ngưỡng Cascade riêng cho bệnh nguy hiểm (maximize Recall)
    CASCADE_THRESHOLDS = {
        'Pneumothorax': 0.05,       # Cấp cứu
        'Cardiomegaly': 0.10,
        'Effusion': 0.10,
        'Consolidation': 0.12,
    }
    DEFAULT_CASCADE_THRESHOLD = 0.15
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Khởi tạo YOLO detector.
        
        Args:
            model_path: Đường dẫn tới file .pt đã train
            device: 'auto', 'cpu', 'cuda', hoặc '0'
        """
        self.model = None
        self.available = False
        self.model_path = model_path
        self.device = device
        
        if YOLO is None:
            print("[YOLO] ultralytics not available. Detector disabled.", flush=True)
            return
            
        if not os.path.exists(model_path):
            print(f"[YOLO] Model not found: {model_path}. Detector disabled.", flush=True)
            return
        
        try:
            import torch
            self.model = YOLO(model_path)
            self.available = True
            
            # Explicitly nạp full trọng số YOLO lên GPU
            target_device = 'cuda:0' if (self.device == 'auto' and torch.cuda.is_available()) or str(self.device).startswith('cuda') else self.device
            if target_device != 'auto' and target_device != 'cpu':
                try:
                    self.model.to(target_device)
                    print(f"[YOLO] Successfully transferred YOLO model to {target_device}", flush=True)
                except Exception as ex:
                    print(f"[YOLO] Transfer note: {ex}", flush=True)
            
            # In thông tin model
            n_params = sum(p.numel() for p in self.model.model.parameters())
            print(f"[YOLO] Loaded: {model_path}", flush=True)
            print(f"[YOLO] Parameters: {n_params/1e6:.1f}M", flush=True)
            print(f"[YOLO] Classes: {self.model.names}", flush=True)
            
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}", flush=True)
            self.model = None
            self.available = False
    
    def predict(self, image_np: np.ndarray, conf: float = 0.15, imgsz: int = 1024) -> List[Dict]:
        """
        Chạy YOLO detection trên ảnh.
        
        Args:
            image_np: Ảnh numpy (H, W, 3) RGB
            conf: Ngưỡng confidence tối thiểu
            imgsz: Kích thước inference
            
        Returns:
            List[dict]: Mỗi detection có:
                - vindr_class_id (int)
                - vindr_class_name (str)
                - chexnet_class_name (str or None)
                - confidence (float)
                - bbox (dict): {x1, y1, x2, y2} pixel coordinates
        """
        if not self.available or self.model is None:
            return []
        
        try:
            import torch
            use_half = torch.cuda.is_available() and (self.device != 'cpu')
            device_arg = 0 if (self.device == 'auto' and torch.cuda.is_available()) or self.device == 'cuda' else self.device
            
            results = self.model.predict(
                image_np,
                conf=conf,
                imgsz=imgsz,
                verbose=False,
                augment=False,
                device=device_arg if device_arg != 'auto' else None
            )
            
            if not results or len(results) == 0:
                return []
            
            result = results[0]
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()     # [N, 4] pixel coords
                confs = result.boxes.conf.cpu().numpy()      # [N]
                classes = result.boxes.cls.cpu().numpy().astype(int)  # [N]
                
                for box, c, cls_id in zip(boxes, confs, classes):
                    vindr_name = self.VINDR_CLASS_NAMES[cls_id] if cls_id < len(self.VINDR_CLASS_NAMES) else f"class_{cls_id}"
                    chexnet_name = self.VINDR_TO_CHEXNET.get(cls_id, None)
                    
                    detections.append({
                        'vindr_class_id': int(cls_id),
                        'vindr_class_name': vindr_name,
                        'chexnet_class_name': chexnet_name,
                        'confidence': float(c),
                        'bbox': {
                            'x1': int(box[0]),
                            'y1': int(box[1]),
                            'x2': int(box[2]),
                            'y2': int(box[3]),
                        }
                    })
            
            return detections
            
        except Exception as e:
            print(f"[YOLO] Prediction error: {e}", flush=True)
            return []
    
    def cascade_filter(
        self,
        detections: List[Dict],
        classifier_probs: Dict[str, float],
    ) -> List[Dict]:
        """
        Áp dụng Cascade 2 giai đoạn: Final_Score = P_classifier × Conf_YOLO.
        
        Loại bỏ False Positives bằng cách nhân confidence YOLO với xác suất
        từ CheXNet classifier. Nếu classifier cho xác suất bệnh thấp, 
        YOLO boxes cho bệnh đó bị giảm score mạnh.
        
        Args:
            detections: Output từ self.predict()
            classifier_probs: Dict {disease_name: probability} từ CheXNet classifier
                              Ví dụ: {'Atelectasis': 0.8, 'Cardiomegaly': 0.3, ...}
        
        Returns:
            List[dict]: Detections đã lọc, thêm field 'cascade_score'
        """
        filtered = []
        
        for det in detections:
            chexnet_name = det.get('chexnet_class_name')
            yolo_conf = det['confidence']
            
            if chexnet_name and chexnet_name in classifier_probs:
                # Cascade: nhân confidence YOLO với xác suất classifier
                p_class = classifier_probs[chexnet_name]
                cascade_score = yolo_conf * p_class
            else:
                # Không có mapping → giữ nguyên confidence YOLO
                cascade_score = yolo_conf
            
            # Lấy ngưỡng dựa trên disease name
            display_name = chexnet_name or det['vindr_class_name']
            threshold = self.CASCADE_THRESHOLDS.get(display_name, self.DEFAULT_CASCADE_THRESHOLD)
            
            if cascade_score >= threshold:
                det_copy = det.copy()
                det_copy['cascade_score'] = float(cascade_score)
                det_copy['display_name'] = display_name
                filtered.append(det_copy)
        
        # Sắp xếp theo cascade_score giảm dần
        filtered.sort(key=lambda x: x['cascade_score'], reverse=True)
        return filtered
    
    def draw_boxes(
        self,
        image_np: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 128),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Vẽ YOLO bounding boxes lên ảnh.
        
        Args:
            image_np: Ảnh numpy (H, W, 3) RGB
            detections: Output từ cascade_filter() hoặc predict()
            color: Màu box (R, G, B) — mặc định xanh lá
            thickness: Độ dày đường viền
            
        Returns:
            np.ndarray: Ảnh đã vẽ boxes
        """
        vis = image_np.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            display_name = det.get('display_name', det.get('chexnet_class_name') or det['vindr_class_name'])
            score = det.get('cascade_score', det['confidence'])
            label = f"{display_name}: {score:.0%}"
            
            # Vẽ box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
            
            # Vẽ label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        return vis
    
    def scale_detections(
        self,
        detections: List[Dict],
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> List[Dict]:
        """
        Scale bounding boxes từ kích thước ảnh gốc sang kích thước target.
        Dùng khi cần vẽ boxes lên ảnh đã resize (vd: 384x384 cho heatmap overlay).
        
        Args:
            detections: List detections với bbox ở kích thước original
            original_size: (width, height) của ảnh gốc
            target_size: (width, height) của ảnh đích
            
        Returns:
            List[dict]: Detections với bbox đã scale
        """
        ow, oh = original_size
        tw, th = target_size
        sx = tw / ow
        sy = th / oh
        
        scaled = []
        for det in detections:
            det_copy = det.copy()
            bbox = det['bbox']
            det_copy['bbox'] = {
                'x1': int(bbox['x1'] * sx),
                'y1': int(bbox['y1'] * sy),
                'x2': int(bbox['x2'] * sx),
                'y2': int(bbox['y2'] * sy),
            }
            scaled.append(det_copy)
        
        return scaled
