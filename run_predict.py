import os
import sys
import gc
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Đảm bảo in ký tự Unicode/emoji không bị lỗi cp1252 trên Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thêm đường dẫn project tính từ thư mục gốc CheXNet
project_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(project_dir, 'Backend')
models_dir = os.path.join(project_dir, 'Models')
for p in [project_dir, backend_dir, models_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from predict_single import load_model, predict, CLASS_NAMES
from yolo_detector import YOLODetector

def run_hybrid_yolo_inference():
    print("=" * 65)
    print("  HỆ THỐNG SUY LUẬN ĐA MÔ HÌNH: CHEXNET HYBRID + YOLO11M")
    print("=" * 65)

    # 1. Đường dẫn các mô hình (tính từ thư mục gốc CheXNet)
    hybrid_path = os.path.join(project_dir, "Trainedmodel", "hybrid_model.pth")
    yolo_path = os.path.join(project_dir, "Trainedmodel", "yolov11m.pt")
    
    if not os.path.exists(hybrid_path):
        for alt in [r"CheXNet\Trainedmodel\hybrid_model.pth", os.path.join(os.path.dirname(project_dir), "CheXNet", "Trainedmodel", "hybrid_model.pth")]:
            if os.path.exists(alt):
                hybrid_path = alt
                break

    if not os.path.exists(yolo_path):
        for alt in [r"CheXNet\Trainedmodel\yolov11m.pt", os.path.join(os.path.dirname(project_dir), "CheXNet", "Trainedmodel", "yolov11m.pt")]:
            if os.path.exists(alt):
                yolo_path = alt
                break

    # 2. Xác định danh sách ảnh thử nghiệm trong Results/EX
    ex_dir = os.path.join(project_dir, "Results", "EX")
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if os.path.isabs(arg_path):
            image_paths = [arg_path]
        elif os.path.exists(os.path.join(project_dir, arg_path)):
            image_paths = [os.path.join(project_dir, arg_path)]
        else:
            image_paths = [arg_path]
    else:
        valid_exts = ('.png', '.jpg', '.jpeg')
        image_paths = [
            os.path.join(ex_dir, f) for f in os.listdir(ex_dir)
            if f.lower().endswith(valid_exts) and not f.endswith(('_predict.png', '_bbox.png', '_fusion.png'))
        ]
        if not image_paths:
            image_paths = [os.path.join(ex_dir, "Cardiomegally.png")]

    # 3. Nạp mô hình 1: CheXNet Hybrid CNN-ViT (Tối ưu VRAM: nạp CPU trước rồi sang GPU)
    print(f"\n[1/2] Đang nạp mô hình CheXNet Hybrid từ: {hybrid_path}...")
    model, device = load_model(hybrid_path, model_size="large", img_size=384)
    vram_chexnet = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0

    # 4. Nạp mô hình 2: YOLO11m Object Detector
    print(f"\n[2/2] Đang nạp mô hình YOLO11m từ: {yolo_path}...")
    yolo_detector = YOLODetector(model_path=yolo_path, device=device)
    vram_total = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
    print(f"✅ Đã nạp xong cả 2 mô hình! Tổng VRAM sử dụng: {vram_total:.2f} GB (CheXNet: {vram_chexnet:.2f} GB, YOLO: {vram_total - vram_chexnet:.2f} GB)")

    # 5. Đọc per-class optimal thresholds nếu có
    thresholds_file = os.path.join(project_dir, 'Results', 'optimal_thresholds.npy')
    optimal_thresholds = np.load(thresholds_file) if os.path.isfile(thresholds_file) else None

    # Lặp qua từng ảnh thử nghiệm
    for img_idx, image_path in enumerate(image_paths, 1):
        if not os.path.exists(image_path):
            print(f"❌ Không tìm thấy ảnh: {image_path}")
            continue

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n{'='*65}")
        print(f"  [Ảnh {img_idx}/{len(image_paths)}] ĐANG XỬ LÝ: {os.path.basename(image_path)}")
        print(f"{'='*65}")

        # 6. Chạy suy luận CheXNet
        print(f"🔍 Đang chạy suy luận phân loại CheXNet...")
        probs, positives, att_maps, original_np = predict(
            model, device, image_path, img_size=384, threshold=0.5,
            optimal_thresholds=optimal_thresholds
        )

        # 7. Chạy suy luận YOLO11m
        print("🔍 Đang chạy suy luận định vị tổn thương YOLO11m...")
        raw_image = Image.open(image_path).convert('RGB')
        yolo_raw = yolo_detector.predict(np.array(raw_image), conf=0.15, imgsz=1024)

        # 8. Áp dụng Bộ lọc liên hợp 2 giai đoạn (Cascade 2-Stage Filter)
        classifier_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
        yolo_cascade = yolo_detector.cascade_filter(yolo_raw, classifier_probs)

        # 9. LỌC TẤT CẢ CÁC BỆNH CÓ BOUNDING BOX TỪ YOLO (ĐIỀU KIỆN: CÓ BBOX & ĐẠT NGƯỠNG CASCADE)
        DISEASE_NAMES_VI = {
            'Atelectasis': 'Xẹp phổi', 'Cardiomegaly': 'Phì đại tim', 'Effusion': 'Tràn dịch màng phổi',
            'Infiltration': 'Thâm nhiễm phổi', 'Mass': 'Khối u phổi', 'Nodule': 'Nốt mờ phổi',
            'Pneumonia': 'Viêm phổi', 'Pneumothorax': 'Tràn khí màng phổi', 'Consolidation': 'Đông đặc phổi',
            'Edema': 'Phù phổi', 'Emphysema': 'Khí phế thũng', 'Fibrosis': 'Xơ hóa phổi',
            'Pleural_Thickening': 'Dày màng phổi', 'Hernia': 'Thoát vị hoành',
        }
        DISEASE_COLORS = {
            'Cardiomegaly': (0, 255, 128),      # Xanh lá ngọc
            'Effusion': (0, 190, 255),          # Xanh da trời
            'Pneumothorax': (255, 60, 60),      # Đỏ san hô
            'Atelectasis': (255, 180, 0),       # Vàng cam
            'Infiltration': (200, 100, 255),    # Tím nhạt
            'Consolidation': (255, 110, 160),   # Hồng sen
            'Mass': (255, 50, 100),             # Đỏ hồng
            'Nodule': (255, 140, 40),           # Cam đậm
            'Pneumonia': (255, 80, 80),         # Đỏ tươi
            'Edema': (80, 220, 255),            # Xanh lơ
            'Emphysema': (180, 220, 50),        # Xanh chanh
            'Fibrosis': (160, 160, 160),        # Xám bạc
            'Pleural_Thickening': (100, 180, 180),
            'Hernia': (220, 120, 50),
        }

        # Gom nhóm các bounding box theo từng bệnh lý
        boxes_by_disease = {}
        for det in yolo_cascade:
            d_name = det.get('display_name')
            if d_name and d_name in CLASS_NAMES and d_name != 'No Finding':
                boxes_by_disease.setdefault(d_name, []).append(det)

        # Danh sách các bệnh lý thỏa mãn điều kiện CÓ BBOX
        detected_diseases_with_boxes = []
        for d_name, b_list in boxes_by_disease.items():
            d_idx = CLASS_NAMES.index(d_name)
            p_cls = float(probs[d_idx])
            best_box = max(b_list, key=lambda b: b.get('cascade_score', b['confidence']))
            detected_diseases_with_boxes.append({
                'disease': d_name,
                'disease_vi': DISEASE_NAMES_VI.get(d_name, d_name),
                'idx': d_idx,
                'p_cls': p_cls,
                'yolo_conf': best_box['confidence'],
                'cascade_score': best_box.get('cascade_score', best_box['confidence']),
                'boxes': b_list,
                'best_box': best_box,
                'color': DISEASE_COLORS.get(d_name, (0, 255, 128))
            })

        # Sắp xếp các bệnh có Bounding Box theo điểm liên hợp Cascade giảm dần
        detected_diseases_with_boxes.sort(key=lambda x: x['cascade_score'], reverse=True)

        # In báo cáo chi tiết các bệnh có Bbox ra Console
        print("\n" + "═" * 68)
        print("   🏆 CÁC BỆNH LÝ CÓ ĐIỂM TIN CẬY CAO NHẤT KÈM BOUNDING BOX")
        print("═" * 68)
        print(f"  • Tệp ảnh phân tích     : {os.path.basename(image_path)}")
        print(f"  • Số bệnh có Bounding Box: {len(detected_diseases_with_boxes)}")
        print("─" * 68)

        if detected_diseases_with_boxes:
            for rank, item in enumerate(detected_diseases_with_boxes, 1):
                d = item['disease']
                d_vi = item['disease_vi']
                box_coord = item['best_box']['bbox']
                print(f"  [{rank}] {d.upper()} ({d_vi}):")
                print(f"      - Xác suất CheXNet       : {item['p_cls']:.2%}")
                print(f"      - Độ tin cậy YOLO11m     : {item['yolo_conf']:.2%}")
                print(f"      - Điểm liên hợp Cascade : {item['cascade_score']:.2%}")
                print(f"      - Bounding Box [x1,y1,x2,y2]: [{box_coord['x1']}, {box_coord['y1']}, {box_coord['x2']}, {box_coord['y2']}]")
        else:
            print("  ⚠️ Không phát hiện bệnh lý nào có Bounding Box đạt ngưỡng cascade.")
        print(f"  • Tổng VRAM thực tế     : {vram_total:.2f} GB")
        print("═" * 68)

        # 10. Trực quan hóa Display Fusion & Xuất các ảnh riêng biệt
        orig_w, orig_h = raw_image.size
        img_size = 384

        # ── A. LƯU ẢNH BBOX RIÊNG BIỆT (CHỨA TẤT CẢ CÁC BỆNH CÓ BBOX) ──
        raw_np_bbox = np.array(raw_image).copy()
        for item in detected_diseases_with_boxes:
            d_color = item['color']
            for b_info in item['boxes']:
                b = b_info['bbox']
                bx1, by1, bx2, by2 = max(0, b['x1']), max(0, b['y1']), min(orig_w, b['x2']), min(orig_h, b['y2'])
                score = b_info.get('cascade_score', b_info['confidence'])
                label_box = f"{item['disease']} {score:.0%}"
                
                # Vẽ box
                cv2.rectangle(raw_np_bbox, (bx1, by1), (bx2, by2), d_color, 3, lineType=cv2.LINE_AA)
                
                # Vẽ background nhãn
                (tw, th), _ = cv2.getTextSize(label_box, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(raw_np_bbox, (bx1, max(0, by1 - th - 10)), (bx1 + tw + 8, by1), d_color, -1)
                cv2.putText(raw_np_bbox, label_box, (bx1 + 4, max(20, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

        save_bbox_path = os.path.join(ex_dir, f'{base_name}_bbox.png')
        Image.fromarray(raw_np_bbox).save(save_bbox_path)
        print(f"  📸 [Ảnh BBox độc lập]       : {save_bbox_path}")

        # ── B. LƯU ẢNH DISPLAY FUSION RIÊNG BIỆT (LỒNG GHÉP HEATMAP ⊂ TỪNG BBOX) ──
        raw_np_fusion = np.array(raw_image).copy()
        for item in detected_diseases_with_boxes:
            d_color = item['color']
            att_2d = att_maps[item['idx']]
            att_full = cv2.resize(att_2d, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            att_norm_full = np.uint8(255 * (att_full - att_full.min()) / (att_full.max() - att_full.min() + 1e-8))
            heatmap_full = cv2.applyColorMap(att_norm_full, cv2.COLORMAP_JET)
            heatmap_full = cv2.cvtColor(heatmap_full, cv2.COLOR_BGR2RGB)

            for b_info in item['boxes']:
                b = b_info['bbox']
                bx1, by1, bx2, by2 = max(0, b['x1']), max(0, b['y1']), min(orig_w, b['x2']), min(orig_h, b['y2'])
                roi_hm = heatmap_full[by1:by2, bx1:bx2]
                roi_orig = raw_np_fusion[by1:by2, bx1:bx2]
                raw_np_fusion[by1:by2, bx1:bx2] = np.uint8(roi_hm * 0.45 + roi_orig * 0.55)
                
                cv2.rectangle(raw_np_fusion, (bx1, by1), (bx2, by2), d_color, 3, lineType=cv2.LINE_AA)
                score = b_info.get('cascade_score', b_info['confidence'])
                label_box = f"{item['disease']} {score:.0%}"
                (tw, th), _ = cv2.getTextSize(label_box, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(raw_np_fusion, (bx1, max(0, by1 - th - 10)), (bx1 + tw + 8, by1), d_color, -1)
                cv2.putText(raw_np_fusion, label_box, (bx1 + 4, max(20, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

        save_fusion_path = os.path.join(ex_dir, f'{base_name}_fusion.png')
        Image.fromarray(raw_np_fusion).save(save_fusion_path)
        print(f"  📸 [Ảnh Display Fusion độc lập]: {save_fusion_path}")

        # ── C. LƯU BẢNG SO SÁNH 3 KHUNG HÌNH (SUMMARY FIGURE) ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # Panel 1: Ảnh gốc
        axes[0].imshow(original_np)
        axes[0].set_title("1. Ảnh X-Quang Gốc", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Display Fusion trên kích thước 384x384
        fusion_vis_384 = original_np.copy()
        for item in detected_diseases_with_boxes:
            d_color = item['color']
            att_2d = att_maps[item['idx']]
            att_resized = cv2.resize(att_2d, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            att_norm = np.uint8(255 * (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8))
            heatmap_384 = cv2.applyColorMap(att_norm, cv2.COLORMAP_JET)
            heatmap_384 = cv2.cvtColor(heatmap_384, cv2.COLOR_BGR2RGB)

            scaled_boxes = yolo_detector.scale_detections(item['boxes'], (orig_w, orig_h), (img_size, img_size))
            for sb_info in scaled_boxes:
                sb = sb_info['bbox']
                bx1, by1 = max(0, sb['x1']), max(0, sb['y1'])
                bx2, by2 = min(img_size, sb['x2']), min(img_size, sb['y2'])
                roi_hm = heatmap_384[by1:by2, bx1:bx2]
                roi_orig = fusion_vis_384[by1:by2, bx1:bx2]
                fusion_vis_384[by1:by2, bx1:bx2] = np.uint8(roi_hm * 0.45 + roi_orig * 0.55)
                cv2.rectangle(fusion_vis_384, (bx1, by1), (bx2, by2), d_color, 2, lineType=cv2.LINE_AA)
                score = sb_info.get('cascade_score', sb_info['confidence'])
                label_txt = f"{item['disease']} {score:.0%}"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(fusion_vis_384, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 4, by1), d_color, -1)
                cv2.putText(fusion_vis_384, label_txt, (bx1 + 2, max(12, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        axes[1].imshow(fusion_vis_384)
        detected_names_str = ", ".join([d['disease'] for d in detected_diseases_with_boxes])
        axes[1].set_title(f"2. Display Fusion ({len(detected_diseases_with_boxes)} bệnh có BBox)", fontsize=12, fontweight='bold', color='#009944')
        axes[1].axis('off')

        # Panel 3: Phổ xác suất làm nổi bật các bệnh có Bounding Box
        disease_order = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if CLASS_NAMES[i] != 'No Finding']
        probs_order = [probs[CLASS_NAMES.index(d)] for d in disease_order]
        sorted_pairs = sorted(zip(disease_order, probs_order), key=lambda x: x[1])
        d_names = [x[0] for x in sorted_pairs]
        d_probs = [x[1] for x in sorted_pairs]

        # Màu sắc: bệnh có BBox được tô màu riêng nổi bật, bệnh không có BBox tô màu xám nhạt
        detected_names_set = {item['disease'] for item in detected_diseases_with_boxes}
        bar_colors = []
        for d in d_names:
            if d in detected_names_set:
                # Chuyển màu RGB sang chuẩn hex hoặc rgb float
                match_item = next(it for it in detected_diseases_with_boxes if it['disease'] == d)
                c_rgb = match_item['color']
                bar_colors.append(f'#{c_rgb[0]:02x}{c_rgb[1]:02x}{c_rgb[2]:02x}')
            else:
                bar_colors.append('#b0c4de')

        axes[2].barh(d_names, d_probs, color=bar_colors, edgecolor='white', height=0.6)
        axes[2].set_xlim(0, 1.0)
        axes[2].set_title(f"3. Phổ Xác Suất (Đậm màu = Có BBox)", fontsize=12, fontweight='bold')
        axes[2].grid(axis='x', linestyle='--', alpha=0.5)

        # Thêm chú thích điểm cascade lên thanh bar của bệnh có Bbox
        for i, (name, prob) in enumerate(zip(d_names, d_probs)):
            if name in detected_names_set:
                match_item = next(it for it in detected_diseases_with_boxes if it['disease'] == name)
                casc = match_item['cascade_score']
                axes[2].text(prob + 0.02, i, f"Cascade {casc:.0%}", va='center', fontsize=9, fontweight='bold', color='#1a5276')

        plt.tight_layout()
        save_summary_path = os.path.join(ex_dir, f'{base_name}_predict.png')
        plt.savefig(save_summary_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"  📸 [Ảnh tổng hợp 3 khung]  : {save_summary_path}")

    print("\n✅ Hoàn tất toàn bộ suy luận kiểm thử độc lập!")

if __name__ == '__main__':
    run_hybrid_yolo_inference()
