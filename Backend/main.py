import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import io
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# ── Thêm thư mục gốc dự án và Backend vào sys.path ──
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
models_dir = os.path.join(project_root, 'Models')
for p in [backend_dir, models_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Model import HybridCNNViTModel
from checkpoint_utils import load_checkpoint_safe, extract_state_dict
from post_processing import cam_to_bboxes, compute_attention_quality, check_co_occurrence, get_clinical_thresholds
from lung_segmentation import (
    get_lung_mask, apply_lung_mask, compute_lesion_area_ratio,
    get_lesion_location, NON_LUNG_DISEASES,
)
from report_generator import ClinicalReportGenerator
from cbir import SimilarCaseFinder
from yolo_detector import YOLODetector, draw_dashed_rectangle
import torch.nn.functional as F

app = FastAPI(title="CheXNet Local Inference API")

# Cấu hình CORS để frontend có thể gọi được
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khai báo biến toàn cục cho model và modules
model = None
device = None
transform_inference = None  # Pipeline tiền xử lý hoàn chỉnh (PIL → Tensor), thống nhất với predict_single.py
yolo_detector = None  # YOLO11m object detector
current_model_info = {}
report_gen = ClinicalReportGenerator()
cbir_finder = SimilarCaseFinder()

CLASS_NAMES = [
    'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# Ngưỡng quyết định theo mức độ ưu tiên lâm sàng (post_processing.py đã định nghĩa
# sẵn hàm này nhưng trước đây KHÔNG được import/gọi ở đâu cả — /predict dùng cứng
# 0.5 cho MỌI bệnh, kể cả Pneumothorax/Pneumonia (critical, lẽ ra ngưỡng 0.25 để
# ưu tiên recall, tránh bỏ sót). Tính 1 lần lúc khởi động, dùng lại cho mọi request.
CLINICAL_THRESHOLDS = get_clinical_thresholds()

@app.on_event("startup")
def load_model():
    global model, device, transform_inference, yolo_detector, optimal_thresholds, current_model_info
    import gc
    import os
    import numpy as np
    
    # Load optimal thresholds giống predict_single.py
    thresholds_file = os.path.join(project_root, 'Results', 'optimal_thresholds.npy')
    optimal_thresholds = None
    if os.path.isfile(thresholds_file):
        optimal_thresholds = np.load(thresholds_file)
        print(f"[BACKEND] Loaded per-class optimal thresholds from {thresholds_file}")
    else:
        print("[BACKEND] No optimal thresholds found. Using default 0.5")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"\n[BACKEND] Starting Backend - Target device: {device} ({device_name})", flush=True)
    
    # Mọi đường dẫn tính từ thư mục CheXNet; ưu tiên MODEL_PATH hoặc hybrid_large_v2.pth
    env_model_path = os.environ.get("MODEL_PATH", "").strip()
    if env_model_path:
        model_path = env_model_path if os.path.isabs(env_model_path) else os.path.join(project_root, env_model_path)
    else:
        model_path = os.path.join(project_root, "Trainedmodel", "hybrid_large_v2.pth")

    if not os.path.exists(model_path):
        for alt in [
            os.path.join(project_root, "Trainedmodel", "hybrid_large_v2.pth"),
            os.path.join(project_root, "Trainedmodel", "hybrid_model.pth"),
            r"CheXNet\Trainedmodel\hybrid_model.pth",
            os.path.join(os.path.dirname(project_root), "CheXNet", "Trainedmodel", "hybrid_model.pth"),
            os.path.join(project_root, "Trainedmodel", "hybrid_model_light.pth")
        ]:
            if os.path.exists(alt):
                model_path = alt
                break
    print(f"[BACKEND] Loading checkpoint from: {model_path}", flush=True)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Cannot find model at {model_path}. Please check path!", flush=True)
        return

    try:
        # FIX: Load checkpoint VE CPU truoc, tranh cross-device issues khi
        # load_state_dict nhan tensors tren GPU nhung model dang o CPU.
        # Su dung checkpoint_utils.py de thong nhat logic load toan project.
        ckpt = load_checkpoint_safe(model_path, device=torch.device('cpu'))
        
        model_size = ckpt.get('model_size', 'large')
        img_size = ckpt.get('img_size', 384)
        
        # FIX: Su dung extract_state_dict() tu checkpoint_utils.py de xu ly
        # module. prefix va cac truong hop state_dict key khac nhau.
        cleaned_state_dict = extract_state_dict(ckpt)

        print(f"[BACKEND] Initializing HybridCNNViTModel (Size: {model_size.upper()}, Img: {img_size}px)...", flush=True)
        model = HybridCNNViTModel(num_classes=15, model_size=model_size, img_size=img_size, pretrained=False)

        load_result = model.load_state_dict(cleaned_state_dict, strict=False)
        critical_prefixes = ('classifier.', 'attention_head.', 'fpn_lateral.', 'fpn_merge.', 'vit_norm.')
        critical_missing = [k for k in load_result.missing_keys if k.startswith(critical_prefixes)]
        if load_result.missing_keys:
            print(f"[WARN] {len(load_result.missing_keys)} missing keys: {load_result.missing_keys[:10]}"
                  f"{' ...' if len(load_result.missing_keys) > 10 else ''}", flush=True)
        if load_result.unexpected_keys:
            print(f"[WARN] {len(load_result.unexpected_keys)} unexpected keys: {load_result.unexpected_keys[:10]}"
                  f"{' ...' if len(load_result.unexpected_keys) > 10 else ''}", flush=True)
        if critical_missing:
            print(f"[CRITICAL] classifier/attention_head/fpn MISSING from checkpoint: {critical_missing}. "
                  f"Model running with RANDOM weights for these layers!", flush=True)
        elif not load_result.missing_keys and not load_result.unexpected_keys:
            print(f"[BACKEND] Checkpoint matched 100% - no missing or unexpected keys.", flush=True)

        # FIX: Di chuyen model len GPU SAU khi load_state_dict thanh cong.
        # Truoc day load checkpoint len GPU (map_location=device) roi load vao
        # model dang o CPU -> cross-device assignment co the gay loi am tham.
        print(f"[BACKEND] Moving Model to {device} ({device_name})...", flush=True)
        model = model.to(device)
        model.eval()
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(False)
        for param in model.parameters():
            param.requires_grad_(False)
        
        # FIX: Pipeline tien xu ly THONG NHAT voi predict_single.py va TrainModel.py.
        # Truoc day: ToTensor() -> Resize/Crop tren TENSOR (dung bilinear interpolation
        # cua PyTorch) -> Normalize. Ket qua khac biet dang ke so voi predict_single.py
        # vi predict_single.py dung Resize/Crop tren PIL Image (dung Pillow's BILINEAR,
        # co anti-aliasing) -> ToTensor() -> Normalize.
        # Gio: thong nhat PIL -> Resize -> CenterCrop -> ToTensor -> Normalize.
        transform_inference = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        ckpt_epoch = ckpt.get('epoch')
        ckpt_auroc = ckpt.get('best_auroc')
        current_model_info = {
            "path": model_path,
            "filename": os.path.basename(model_path),
            "model_size": model_size,
            "img_size": img_size,
            "epoch": int(ckpt_epoch) if ckpt_epoch is not None else None,
            "best_auroc": float(ckpt_auroc) if ckpt_auroc is not None else None,
        }

        # Giai phong checkpoint khoi RAM sau khi da load xong
        del ckpt, cleaned_state_dict
        gc.collect()
        
        vram_allocated = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
        print(f"[SUCCESS] CheXNet Model ({current_model_info['filename']}) loaded onto {device} ({device_name}) SUCCESSFULLY! (VRAM: {vram_allocated:.3f} GB)\n", flush=True)
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Error loading model into GPU: {e}", flush=True)
        traceback.print_exc()
    
    # ── Load YOLO Detector (tính từ thư mục CheXNet) ──
    yolo_path = os.path.join(project_root, "Trainedmodel", "yolov11m.pt")
    if not os.path.exists(yolo_path):
        for alt in [
            r"CheXNet\Trainedmodel\yolov11m.pt",
            os.path.join(os.path.dirname(project_root), "CheXNet", "Trainedmodel", "yolov11m.pt")
        ]:
            if os.path.exists(alt):
                yolo_path = alt
                break
    yolo_detector = YOLODetector(model_path=yolo_path, device=device)
    if yolo_detector.available:
        print(f"[BACKEND] YOLO11m detector loaded from: {yolo_path}", flush=True)
    else:
        print(f"[WARN] YOLO11m detector could not be loaded from: {yolo_path}", flush=True)



@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "yolo_loaded": yolo_detector is not None and yolo_detector.available,
        "device": str(device) if device else "unknown",
        "model_file": current_model_info.get("filename"),
        "model_info": current_model_info,
    }

# Nhóm bệnh chưa có ground-truth BBox trong VinDr-CXR -> attention_head yếu, ưu tiên Grad-CAM khi bổ sung khung
GRADCAM_PREFERRED_DISEASES = {'Pneumonia', 'Edema', 'Emphysema', 'Hernia'}


def _project_crop_bbox_to_original(bbox, orig_size, resized_size, crop_size):
    """Quy đổi bbox trong ảnh CenterCrop(crop_size) về hệ tọa độ ảnh gốc (orig_w, orig_h)."""
    orig_w, orig_h = orig_size
    resized_w, resized_h = resized_size
    if orig_w <= 0 or orig_h <= 0 or resized_w <= 0 or resized_h <= 0:
        return None
    crop_left = int(round((resized_w - crop_size) / 2.0))
    crop_top = int(round((resized_h - crop_size) / 2.0))
    ox1 = max(0, min(orig_w, int(round((bbox['x1'] + crop_left) * orig_w / resized_w))))
    oy1 = max(0, min(orig_h, int(round((bbox['y1'] + crop_top) * orig_h / resized_h))))
    ox2 = max(0, min(orig_w, int(round((bbox['x2'] + crop_left) * orig_w / resized_w))))
    oy2 = max(0, min(orig_h, int(round((bbox['y2'] + crop_top) * orig_h / resized_h))))
    if ox2 <= ox1 or oy2 <= oy1:
        return None
    return {'x1': ox1, 'y1': oy1, 'x2': ox2, 'y2': oy2}


def _compute_fpn_gradcam_maps(model_obj, inp_tensor, class_indices):
    """Tính bản đồ Grad-CAM trên fpn_merge cho danh sách class_indices trong 1 lượt forward."""
    unique_indices = list(dict.fromkeys(int(i) for i in class_indices))
    fpn_module = getattr(model_obj, 'fpn_merge', None)
    if not unique_indices or fpn_module is None:
        return {}

    feats = {}

    def _capture_fpn(_mod, _inp, out):
        feats['fpn'] = out

    handle = fpn_module.register_forward_hook(_capture_fpn)
    gradcam_maps = {}
    try:
        with torch.enable_grad():
            x_in = inp_tensor.detach().requires_grad_(True)
            out = model_obj(x_in)
            logits = out[0] if isinstance(out, tuple) else out
            fpn = feats.get('fpn')
            if fpn is None or not fpn.requires_grad:
                return {}
            for pos, cls_idx in enumerate(unique_indices):
                grad = torch.autograd.grad(
                    logits[0, cls_idx],
                    fpn,
                    retain_graph=(pos < len(unique_indices) - 1),
                    allow_unused=True,
                )[0]
                if grad is None:
                    continue
                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * fpn).sum(dim=1, keepdim=True))[0, 0].detach().cpu().numpy()
                cmin, cmax = float(cam.min()), float(cam.max())
                if cmax - cmin > 1e-8:
                    cam = (cam - cmin) / (cmax - cmin + 1e-8)
                else:
                    cam = np.zeros_like(cam, dtype=np.float32)
                gradcam_maps[cls_idx] = cam.astype(np.float32)
    except Exception as exc:
        print(f"[WARN] Grad-CAM localization fallback failed: {exc}", flush=True)
    finally:
        handle.remove()
    return gradcam_maps


def _select_hybrid_localization(disease_name, att_2d, att_filtered_2d, lung_mask, img_size, gradcam_2d=None):
    """Chọn bản đồ vị trí và danh sách bbox tốt nhất từ mô hình Hybrid khi YOLO chưa vẽ khung."""
    import cv2

    if disease_name in GRADCAM_PREFERRED_DISEASES and gradcam_2d is not None and float(gradcam_2d.max()) > 1e-4:
        gcam_vis = gradcam_2d
        if disease_name not in NON_LUNG_DISEASES and lung_mask is not None and float(lung_mask.mean()) > 0.05:
            mask_resized = cv2.resize(lung_mask, (gradcam_2d.shape[1], gradcam_2d.shape[0]), interpolation=cv2.INTER_NEAREST)
            masked_gcam = gradcam_2d * mask_resized
            if float(masked_gcam.max()) >= 0.25:
                masked_boxes = cam_to_bboxes(masked_gcam, img_size, threshold_method='otsu')
                if masked_boxes:
                    return masked_gcam, masked_boxes, 'hybrid-gradcam'
        gcam_boxes = cam_to_bboxes(gcam_vis, img_size, threshold_method='otsu')
        if gcam_boxes:
            return gcam_vis, gcam_boxes, 'hybrid-gradcam'

    att_boxes = cam_to_bboxes(att_filtered_2d, img_size, threshold_method='otsu')
    if att_boxes:
        return att_filtered_2d, att_boxes, 'hybrid-attention'

    if float(att_2d.max()) >= 0.15:
        raw_boxes = cam_to_bboxes(att_2d, img_size, threshold_method='otsu')
        if raw_boxes:
            return att_2d, raw_boxes, 'hybrid-attention'

    if gradcam_2d is not None and float(gradcam_2d.max()) > 1e-4:
        gcam_boxes = cam_to_bboxes(gradcam_2d, img_size, threshold_method='otsu')
        return gradcam_2d, gcam_boxes, 'hybrid-gradcam'

    return att_filtered_2d, [], 'hybrid-attention'


def _collect_hybrid_fallback_boxes(
    disease_name, probability, bboxes, source, orig_size, resized_size, crop_size, max_boxes=2, min_conf=0.2
):
    """Chuyển đổi tối đa max_boxes khung Hybrid sang tọa độ ảnh gốc cho bệnh YOLO chưa vẽ."""
    if not bboxes:
        return []
    orig_area = max(1, orig_size[0] * orig_size[1])
    candidates = [b for b in bboxes if float(b.get('confidence', 0)) >= min_conf] or bboxes[:1]
    collected = []
    for bbox in candidates:
        projected = _project_crop_bbox_to_original(bbox, orig_size, resized_size, crop_size)
        if not projected:
            continue
        box_area = (projected['x2'] - projected['x1']) * (projected['y2'] - projected['y1'])
        if box_area < 0.002 * orig_area:
            continue
        collected.append({
            'disease': disease_name,
            'confidence': float(bbox.get('confidence', 0.0)),
            'probability': float(probability),
            'source': source,
            'bbox': projected,
        })
        if len(collected) >= max_boxes:
            break
    return collected


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Mô hình chưa được tải thành công trên Backend."}
        
    try:
        import cv2
        import base64
        
        # Doc anh tu request
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # FIX: Tien xu ly THONG NHAT voi predict_single.py.
        # Dung transform_inference (PIL -> Resize -> CenterCrop -> ToTensor -> Normalize)
        # thay vi ToTensor -> Resize -> Crop -> Normalize (thu tu cu bi sai).
        inputs = transform_inference(image).unsqueeze(0).to(device)
        
        # Anh goc numpy de overlay heatmap (sau khi crop giong pipeline)
        img_size = current_model_info.get("img_size") or 384
        orig_w, orig_h = image.size
        resized_vis = transforms.Resize(int(img_size * 1.14))(image)
        resized_size = resized_vis.size
        original_vis = transforms.CenterCrop(img_size)(resized_vis)
        original_np = np.array(original_vis)
        
        # Chạy dự đoán chuẩn như predict_single.py (Không dùng Data Augmentation lúc test)
        pooled_np = None
        with torch.inference_mode():
            logits, att_aug, pooled_t = model(inputs, return_embedding=True)
            pooled_np = pooled_t.cpu().numpy()
            
            probs = torch.sigmoid(logits[0].float()).cpu().numpy()
            att_maps = att_aug[0].cpu().numpy()
            
            # Khởi tạo uncertainty giả (để không lỗi đoạn code sau)
            std_probs = np.zeros_like(probs)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Lấy mask phổi để lọc attention
        lung_mask = get_lung_mask(original_np, method='classical')
        att_maps_filtered = apply_lung_mask(att_maps, lung_mask, CLASS_NAMES)
        
        # ── YOLO Detection ──
        yolo_detections_raw = []
        yolo_detections_filtered = []
        if yolo_detector and yolo_detector.available:
            # Chạy YOLO trên ảnh gốc (full resolution, chưa crop 384)
            yolo_input = np.array(image)  # PIL → numpy RGB
            yolo_detections_raw = yolo_detector.predict(yolo_input, conf=0.15, imgsz=1024)
            
            # Tạo dict xác suất classifier để Cascade filter
            classifier_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
            yolo_detections_filtered = yolo_detector.cascade_filter(yolo_detections_raw, classifier_probs)
            
            print(f"[YOLO] Raw: {len(yolo_detections_raw)} boxes → Cascade: {len(yolo_detections_filtered)} boxes", flush=True)
        
        # Thu thập toàn bộ 15 xác suất
        all_results = []
        for i, prob in enumerate(probs):
            all_results.append({
                "idx": i,
                "disease": CLASS_NAMES[i],
                "probability": float(prob)
            })
            
        # Sắp xếp toàn bộ 15 bệnh (bao gồm cả No Finding) để trả về cho Frontend hiển thị chart
        all_results.sort(key=lambda x: x["probability"], reverse=True)
        
        # Lọc danh sách bệnh tổn thương (bỏ No Finding) ĐỂ VẼ HEATMAP
        pathology_results = [r for r in all_results if r["disease"] != 'No Finding']
        
        # Sử dụng optimal_thresholds giống hệt predict_single.py (thay vì CLINICAL_THRESHOLDS)
        positive_targets = []
        for r in pathology_results:
            d_idx = r["idx"]
            thr = optimal_thresholds[d_idx] if 'optimal_thresholds' in globals() and optimal_thresholds is not None else 0.5
            if r["probability"] >= thr:
                positive_targets.append(r)

        # Giới hạn số bệnh vẽ JPEG heatmap ở 6 như trước, nhưng vẫn trích khung Hybrid
        # cho MỌI bệnh vượt ngưỡng mà YOLO chưa vẽ.
        top_targets = positive_targets[:6]

        yolo_by_disease = {}
        if yolo_detections_filtered and yolo_detector:
            for det in yolo_detections_filtered:
                key = det.get('display_name')
                if key:
                    yolo_by_disease.setdefault(key, []).append(det)

        # Với mọi bệnh vượt ngưỡng chưa có YOLO box (kể cả bệnh xếp sau top-6 như Pneumonia),
        # chỉ chạy thêm 1 lượt Grad-CAM nếu thực sự cần thiết.
        gradcam_needed_indices = []
        for item in positive_targets:
            disease_name = item["disease"]
            d_idx = item["idx"]
            if yolo_by_disease.get(disease_name):
                continue
            if disease_name in GRADCAM_PREFERRED_DISEASES:
                gradcam_needed_indices.append(d_idx)
                continue
            has_att_box = bool(cam_to_bboxes(att_maps_filtered[d_idx], img_size, threshold_method='otsu'))
            if not has_att_box and float(att_maps[d_idx].max()) >= 0.15:
                has_att_box = bool(cam_to_bboxes(att_maps[d_idx], img_size, threshold_method='otsu'))
            if not has_att_box:
                gradcam_needed_indices.append(d_idx)

        gradcam_maps = _compute_fpn_gradcam_maps(model, inputs, gradcam_needed_indices) if gradcam_needed_indices else {}
        if torch.cuda.is_available() and gradcam_needed_indices:
            torch.cuda.empty_cache()

        hybrid_boxes = []
        hybrid_loc_by_idx = {}
        for item in positive_targets:
            disease_name = item["disease"]
            d_idx = item["idx"]
            if yolo_by_disease.get(disease_name):
                continue
            vis_map_2d, bboxes, loc_source = _select_hybrid_localization(
                disease_name, att_maps[d_idx], att_maps_filtered[d_idx],
                lung_mask, img_size, gradcam_maps.get(d_idx),
            )
            hybrid_loc_by_idx[d_idx] = (vis_map_2d, bboxes, loc_source)
            hybrid_boxes.extend(
                _collect_hybrid_fallback_boxes(
                    disease_name, float(item["probability"]), bboxes, loc_source,
                    (orig_w, orig_h), resized_size, img_size,
                )
            )
            
        # Generate Heatmaps & Hotspot/Fallback Boxes cho top-6 bệnh được chọn
        predictions = []
        for item in top_targets:
            d_idx = item["idx"]
            disease_name = item["disease"]
            att_2d = att_maps[d_idx]
            att_filtered_2d = att_maps_filtered[d_idx]
            matching_yolo = []
            if yolo_by_disease.get(disease_name) and yolo_detector:
                matching_yolo = yolo_detector.scale_detections(
                    yolo_by_disease[disease_name], (orig_w, orig_h), (img_size, img_size)
                )
            
            # Tính toán các metric phân tích mới
            quality = compute_attention_quality(att_2d)
            # YÊU CẦU: KHÔNG TÁC ĐỘNG VÀO KẾT QUẢ DỰ ĐOÁN CỦA MODEL
            calibrated_conf = float(item["probability"])
            uncertainty = float(std_probs[d_idx])

            if matching_yolo:
                vis_map_2d = att_filtered_2d
                bboxes = cam_to_bboxes(att_filtered_2d, img_size, threshold_method='otsu')
                area_pct = compute_lesion_area_ratio(att_filtered_2d, lung_mask, 0.3)
                location = get_lesion_location(att_filtered_2d, lung_mask, img_size)
            else:
                vis_map_2d, bboxes, _ = hybrid_loc_by_idx[d_idx]
                effective_mask = lung_mask if (vis_map_2d is att_filtered_2d and disease_name not in NON_LUNG_DISEASES) else np.ones_like(lung_mask)
                area_pct = compute_lesion_area_ratio(vis_map_2d, effective_mask, 0.3)
                location = get_lesion_location(vis_map_2d, effective_mask, img_size)
            
            # Resize heatmap lên kích thước ảnh crop hiển thị
            att_resized = cv2.resize(vis_map_2d, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            
            # Normalize 0-255
            att_norm = np.uint8(255 * (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8))
            
            # Apply ColorMap JET
            heatmap = cv2.applyColorMap(att_norm, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # ══════════════════════════════════════════════════════════
            # PHƯƠNG ÁN A: DISPLAY FUSION
            # Heatmap chỉ hiển thị BÊN TRONG YOLO bbox (nếu có).
            # Nếu không có YOLO detection → fallback heatmap toàn ảnh
            # + viền đứt nét từ Hybrid (Attention / Grad-CAM).
            # ══════════════════════════════════════════════════════════
            
            if matching_yolo:
                # ── CÓ YOLO BBOX: Heatmap CHỈ trong bbox ──
                contour_vis = original_np.copy()
                
                for det in matching_yolo:
                    bx1 = max(0, det['bbox']['x1'])
                    by1 = max(0, det['bbox']['y1'])
                    bx2 = min(img_size, det['bbox']['x2'])
                    by2 = min(img_size, det['bbox']['y2'])
                    
                    # Blend heatmap CHỈ bên trong bbox
                    roi_heatmap = heatmap[by1:by2, bx1:bx2]
                    roi_original = original_np[by1:by2, bx1:bx2]
                    roi_blended = np.uint8(roi_heatmap * 0.4 + roi_original * 0.6)
                    contour_vis[by1:by2, bx1:bx2] = roi_blended
                    
                    # Vẽ YOLO bbox viền liền (tin cậy cao)
                    cv2.rectangle(contour_vis, (bx1, by1), (bx2, by2),
                                  (0, 255, 128), 2, lineType=cv2.LINE_AA)
                    
                    # Label trên bbox
                    label = det.get('display_name', '')
                    score = det.get('cascade_score', det.get('confidence', 0))
                    label_text = f"{label} {score:.0%}"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(contour_vis, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 4, max(th + 6, by1)), (0, 255, 128), -1)
                    cv2.putText(contour_vis, label_text, (bx1 + 2, max(th + 2, by1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                # ── KHÔNG CÓ YOLO BBOX: Fallback heatmap toàn ảnh + viền đứt nét Hybrid ──
                contour_vis = np.uint8(heatmap * 0.4 + original_np * 0.6)
                for bbox in bboxes:
                    draw_dashed_rectangle(
                        contour_vis,
                        (bbox['x1'], bbox['y1']),
                        (bbox['x2'], bbox['y2']),
                        (255, 165, 0),
                        thickness=2,
                    )
            
            # Encode Overlay thành Base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(contour_vis, cv2.COLOR_RGB2BGR))
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            predictions.append({
                "disease": item["disease"],
                "probability": item["probability"],
                "calibrated_confidence": calibrated_conf,
                "uncertainty": uncertainty,
                "attention_quality": quality,
                "lesion_area_percent": float(area_pct),
                "lesion_location": location,
                "bounding_boxes": bboxes,
                "heatmap": f"data:image/jpeg;base64,{b64_str}"
            })
            
        # Kiểm tra sự đồng xuất hiện bệnh
        co_occurrences = check_co_occurrence(probs, threshold=0.4)
        
        # Sinh báo cáo y khoa tự động
        clinical_report = report_gen.generate(predictions, co_occurrences)
        
        # Tìm ca bệnh tương tự (CBIR) — nếu index có sẵn
        similar_cases = []
        if cbir_finder.available and pooled_np is not None:
            similar_cases = cbir_finder.search(pooled_np, k=5)
        
        # Chuẩn bị YOLO detections cho response
        yolo_response = []
        for det in yolo_detections_filtered:
            yolo_response.append({
                "disease": det.get('display_name', det['vindr_class_name']),
                "vindr_class": det['vindr_class_name'],
                "confidence": det['confidence'],
                "cascade_score": det.get('cascade_score', det['confidence']),
                "bbox": det['bbox'],
            })
            
        return {
            "success": True,
            "filename": file.filename,
            "predictions": predictions,
            "all_probabilities": all_results,
            "yolo_detections": yolo_response,
            "hybrid_boxes": hybrid_boxes,
            "co_occurrences": co_occurrences,
            "clinical_report": clinical_report,
            "similar_cases": similar_cases
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Lỗi trong quá trình dự đoán: {str(e)}"}

@app.post("/api/add_reference_case")
async def add_reference_case(file: UploadFile = File(...), patient_id: str = Form("Local_Case"), diseases: str = Form("")):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # FIX: Dung transform_inference thong nhat voi /predict endpoint.
        # Pipeline: PIL -> Resize -> CenterCrop -> ToTensor -> Normalize.
        input_tensor = transform_inference(image).unsqueeze(0).to(device)
        
        # Inference để lấy vector đặc trưng (Embedding)
        with torch.no_grad():
            _, _, pooled = model(input_tensor, return_embedding=True)
            
        # Parse danh sách bệnh
        dis_list = [d.strip() for d in diseases.split(',') if d.strip()]
        
        metadata = {
            'filename': patient_id,
            'diseases': dis_list
        }
        
        # Thêm vào cơ sở dữ liệu FAISS
        success = cbir_finder.add_case(pooled.cpu().numpy(), metadata)
        
        if success:
            return {"success": True, "message": "Đã lưu ca bệnh vào cơ sở dữ liệu tham chiếu."}
        else:
            return {"success": False, "error": "CBIR chưa sẵn sàng hoặc có lỗi nội bộ."}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Lỗi xử lý: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    # Chạy server ở port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)