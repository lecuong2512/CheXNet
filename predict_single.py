# predict_single.py
# Dự đoán và visualize attention map cho một ảnh X-quang bất kỳ
import os, sys
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# ── Đường dẫn tới thư mục chứa Model.py ──────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models')
if os.path.isdir(MODEL_DIR) and MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from Model import HybridCNNViTModel
from checkpoint_utils import load_checkpoint_safe, extract_state_dict

# ── Tên 15 nhãn bệnh ─────────────────────────────────────────────────────────
CLASS_NAMES = [
    'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

MAX_DISEASES_SHOWN = 6  # Giới hạn số bệnh hiển thị heatmap

# ═════════════════════════════════════════════════════════════════════════════
def load_model(model_path: str, model_size: str = 'base', img_size: int = 384):
    """Load model từ checkpoint, tự xử lý DataParallel prefix."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = load_checkpoint_safe(model_path, device)
    state_dict = extract_state_dict(ckpt)

    model = HybridCNNViTModel(num_classes=15, model_size=model_size, img_size=img_size)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    best_auroc = ckpt.get('best_auroc', 'N/A')
    trained_epoch = ckpt.get('epoch', 'N/A')
    print(f"✅ Loaded model | epoch={trained_epoch} | best_auroc={best_auroc}")
    return model, device

# ─────────────────────────────────────────────────────────────────────────────
def predict(model, device, image_path: str, img_size: int = 384,
            threshold: float = 0.5, optimal_thresholds: np.ndarray = None):
    """
    Chạy inference một ảnh.
    Trả về:
        probs        : np.ndarray (15,)           – xác suất từng lớp
        positives    : list[str]                   – tên các lớp vượt threshold
        att_maps     : np.ndarray (15, H_feat, W_feat) – 15 attention map thô (chưa resize)
        original_np  : np.ndarray (img_size, img_size, 3) – ảnh gốc đã resize
    """
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attention_map = model(input_tensor)

    probs = torch.sigmoid(logits)[0].cpu().numpy()
    # Dùng per-class threshold nếu có, không thì dùng threshold cố định
    if optimal_thresholds is not None:
        positives = [CLASS_NAMES[i] for i, p in enumerate(probs)
                     if p >= optimal_thresholds[i]]
    else:
        positives = [CLASS_NAMES[i] for i, p in enumerate(probs) if p >= threshold]

    # Attention map: [1, 15, H_feat, W_feat] → [15, H_feat, W_feat]
    # KHÔNG resize ở đây — mỗi kênh sẽ được resize riêng trong visualize()
    att_maps = attention_map[0].cpu().numpy()

    original_np = np.array(original_image.resize((img_size, img_size)))
    return probs, positives, att_maps, original_np

# ─────────────────────────────────────────────────────────────────────────────
def _make_heatmap_overlay(att_map_2d, original_np, img_size):
    """Resize 1 attention map [H_feat, W_feat] → overlay heatmap trên ảnh gốc."""
    att_resized = cv2.resize(att_map_2d, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    att_norm = np.uint8(255 * (att_resized - att_resized.min()) /
                        (att_resized.max() - att_resized.min() + 1e-8))
    heatmap = cv2.applyColorMap(att_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(heatmap * 0.4 + original_np * 0.6)
    return overlay


def visualize(probs, positives, att_maps, original_np,
              image_path: str, save_path: str = None, threshold: float = 0.5,
              img_size: int = 384, optimal_thresholds: np.ndarray = None):
    """
    Vẽ per-disease heatmap overlays:
      Panel 1: Ảnh gốc
      Panel 2..N: Heatmap overlay cho từng bệnh dương tính (tối đa MAX_DISEASES_SHOWN)
      Panel cuối: Bar chart xác suất 15 lớp
    """
    # Xác định bệnh dương tính (loại No Finding), sắp xếp theo xác suất giảm dần
    if optimal_thresholds is not None:
        detected = [(i, CLASS_NAMES[i], probs[i]) for i in range(len(CLASS_NAMES))
                    if CLASS_NAMES[i] != 'No Finding' and probs[i] >= optimal_thresholds[i]]
    else:
        detected = [(i, CLASS_NAMES[i], probs[i]) for i in range(len(CLASS_NAMES))
                    if CLASS_NAMES[i] != 'No Finding' and probs[i] >= threshold]
    detected.sort(key=lambda x: x[2], reverse=True)
    detected = detected[:MAX_DISEASES_SHOWN]

    # Nếu không phát hiện bệnh nào, hiển thị bệnh có xác suất cao nhất
    if not detected:
        candidates = [(i, CLASS_NAMES[i], probs[i]) for i in range(len(CLASS_NAMES))
                      if CLASS_NAMES[i] != 'No Finding']
        best = max(candidates, key=lambda x: x[2])
        detected = [best]

    n_panels = 1 + len(detected) + 1  # ảnh gốc + heatmaps + bar chart
    fig = plt.figure(figsize=(5 * n_panels, 5))
    gs = fig.add_gridspec(1, n_panels)

    # Panel 1 – Ảnh gốc
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(original_np)
    ax0.set_title('Ảnh Gốc', fontsize=12, fontweight='bold')
    ax0.axis('off')

    # Panels 2..N – Per-disease heatmap overlays
    for j, (disease_idx, disease_name, prob) in enumerate(detected):
        ax = fig.add_subplot(gs[j + 1])
        overlay = _make_heatmap_overlay(att_maps[disease_idx], original_np, img_size)
        ax.imshow(overlay)
        thr_for_color = optimal_thresholds[disease_idx] if optimal_thresholds is not None else threshold
        title_color = 'red' if prob >= thr_for_color else 'orange'
        ax.set_title(f'{disease_name}\n(p={prob:.3f})', fontsize=11,
                     fontweight='bold', color=title_color)
        ax.axis('off')

    # Panel cuối – Bar chart xác suất
    ax_bar = fig.add_subplot(gs[-1])
    if optimal_thresholds is not None:
        colors = ['tomato' if p >= optimal_thresholds[i] else 'steelblue' for i, p in enumerate(probs)]
    else:
        colors = ['tomato' if p >= threshold else 'steelblue' for p in probs]
    bars = ax_bar.barh(CLASS_NAMES[::-1], probs[::-1], color=colors[::-1],
                       edgecolor='white', linewidth=0.4)
    if optimal_thresholds is None:
        ax_bar.axvline(x=threshold, color='gray', linestyle='--', linewidth=1,
                       label=f'Threshold={threshold}')
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel('Probability')
    ax_bar.set_title('Xác suất từng lớp', fontsize=12, fontweight='bold')
    ax_bar.legend(fontsize=9)

    # Ghi số lên bar
    for bar, prob in zip(bars, probs[::-1]):
        ax_bar.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                    f'{prob:.2f}', va='center', fontsize=8)

    fig.suptitle(f'Kết quả dự đoán: {os.path.basename(image_path)}',
                 fontsize=13, fontweight='bold', y=1.02)

    # Cố định layout tránh bóp méo đồ thị hoặc đè chữ khi lặp loop
    fig.subplots_adjust(wspace=0.15, hspace=0, left=0.03, right=0.97, bottom=0.10, top=0.85)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Đã lưu: {save_path}")
    else:
        plt.show()

    plt.clf()
    plt.close()

# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  Dự đoán đơn ảnh X-quang – Hybrid CNN-ViT")
    print("="*60)

    model_path = input("\nĐường dẫn model (.pth) [Trainedmodel/hybrid_model.pth]: ").strip() \
                 or 'Trainedmodel/hybrid_model.pth'
    model_size = input("Model size (base/large) [base]: ").strip() or 'base'
    img_size   = int(input("Image size (256/384) [384]: ").strip() or '384')
    threshold  = float(input("Threshold dự đoán (0-1) [0.5]: ").strip() or '0.5')

    model, device = load_model(model_path, model_size, img_size)

    # Load optimal thresholds nếu có
    thresholds_file = os.path.join('Results', 'optimal_thresholds.npy')
    optimal_thresholds = None
    if os.path.isfile(thresholds_file):
        optimal_thresholds = np.load(thresholds_file)
        print(f"✅ Đã load per-class optimal thresholds từ {thresholds_file}")
    else:
        print(f"ℹ️  Không tìm thấy {thresholds_file} — dùng threshold cố định {threshold}")

    print("\n📌 Nhập đường dẫn ảnh để dự đoán. Gõ 'q' để thoát.\n")

    while True:
        image_path = input("Đường dẫn ảnh: ").strip()
        if image_path.lower() in ('q', 'quit', 'exit'):
            print("Thoát.")
            break
        if not os.path.isfile(image_path):
            print(f"❌ Không tìm thấy file: {image_path}")
            continue

        print("🔍 Đang xử lý...")
        try:
            probs, positives, att_maps, original_np = predict(
                model, device, image_path, img_size, threshold,
                optimal_thresholds=optimal_thresholds
            )
        except Exception as e:
            print(f"❌ Lỗi khi xử lý ảnh: {e}")
            continue

        # In kết quả ra console
        print("\n┌─────────────────────────────────────────┐")
        print(f"│ File : {os.path.basename(image_path):<34}│")
        print("├──────────────────────┬──────────────────┤")
        print("│ Nhãn                 │ Xác suất         │")
        print("├──────────────────────┼──────────────────┤")
        for name, prob in zip(CLASS_NAMES, probs):
            marker = " ◀ POSITIVE" if prob >= threshold else ""
            print(f"│ {name:<20} │ {prob:.4f}{marker:<10}│")
        print("└──────────────────────┴──────────────────┘")
        print(f"\n✅ Kết quả dương tính: {positives if positives else ['No Finding']}")

        # Lưu ảnh visualize
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join('Results', f'{base_name}_predict.png')
        visualize(probs, positives, att_maps, original_np,
                  image_path, save_path=save_path, threshold=threshold,
                  img_size=img_size)
        print()


if __name__ == '__main__':
    main()