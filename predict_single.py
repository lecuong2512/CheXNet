# predict_single.py
# Dự đoán và visualize attention map cho một ảnh X-quang bất kỳ
import os, sys
import torch
import torch.serialization  # Đã thêm vào đây để sửa lỗi UnboundLocalError
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict

# ── Đường dẫn tới thư mục chứa Model.py ──────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models')
if os.path.isdir(MODEL_DIR) and MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from Model import HybridCNNViTModel

# ── Tên 15 nhãn bệnh ─────────────────────────────────────────────────────────
CLASS_NAMES = [
    'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# ═════════════════════════════════════════════════════════════════════════════
def load_model(model_path: str, model_size: str = 'base', img_size: int = 384):
    """Load model từ checkpoint, tự xử lý DataParallel prefix."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        import numpy as _np
        # Gọi trực tiếp thông qua torch.serialization đã import ở đầu file
        torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)

    state_dict = ckpt.get('state_dict', ckpt)
    if any(k.startswith('module.') for k in state_dict):
        state_dict = OrderedDict(
            (k[7:] if k.startswith('module.') else k, v)
            for k, v in state_dict.items()
        )

    model = HybridCNNViTModel(num_classes=15, model_size=model_size, img_size=img_size)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    best_auroc = ckpt.get('best_auroc', 'N/A')
    trained_epoch = ckpt.get('epoch', 'N/A')
    print(f"✅ Loaded model | epoch={trained_epoch} | best_auroc={best_auroc}")
    return model, device

# ─────────────────────────────────────────────────────────────────────────────
def predict(model, device, image_path: str, img_size: int = 384,
            threshold: float = 0.5):
    """
    Chạy inference một ảnh.
    Trả về:
        probs        : np.ndarray (15,)  – xác suất từng lớp
        positives    : list[str]         – tên các lớp vượt threshold
        att_map      : np.ndarray (H, W) – attention map đã resize
        original_np  : np.ndarray (H, W, 3) – ảnh gốc đã resize
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attention_map = model(input_tensor)

    probs = torch.sigmoid(logits)[0].cpu().numpy()
    positives = [CLASS_NAMES[i] for i, p in enumerate(probs) if p >= threshold]

    # Xử lý attention map
    att_np = attention_map.squeeze().cpu().numpy()
    att_resized = cv2.resize(att_np, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    original_np = np.array(original_image.resize((img_size, img_size)))
    return probs, positives, att_resized, original_np

# ─────────────────────────────────────────────────────────────────────────────
def visualize(probs, positives, att_map, original_np,
              image_path: str, save_path: str = None, threshold: float = 0.5):
    """Vẽ 4 panel: ảnh gốc | heatmap | overlay | bar chart xác suất."""

    # Heatmap
    att_norm = np.uint8(255 * (att_map - att_map.min()) /
                        (att_map.max() - att_map.min() + 1e-8))
    heatmap = cv2.applyColorMap(att_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(heatmap * 0.4 + original_np * 0.6)

    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 4)

    # Panel 1 – ảnh gốc
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(original_np)
    ax0.set_title('Ảnh Gốc', fontsize=12, fontweight='bold')
    ax0.axis('off')

    # Panel 2 – heatmap
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(heatmap)
    ax1.set_title('Attention Map', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Panel 3 – overlay
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(overlay)
    label_str = '\n'.join(positives) if positives else 'No Finding'
    ax2.set_title(f'Overlay\n({label_str})', fontsize=11, fontweight='bold', color='darkred')
    ax2.axis('off')

    # Panel 4 – bar chart xác suất
    ax3 = fig.add_subplot(gs[3])
    colors = ['tomato' if p >= threshold else 'steelblue' for p in probs]
    bars = ax3.barh(CLASS_NAMES[::-1], probs[::-1], color=colors[::-1], edgecolor='white', linewidth=0.4)
    ax3.axvline(x=threshold, color='gray', linestyle='--', linewidth=1, label=f'Threshold={threshold}')
    ax3.set_xlim(0, 1)
    ax3.set_xlabel('Probability')
    ax3.set_title('Xác suất từng lớp', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)

    # Ghi số lên bar
    for bar, prob in zip(bars, probs[::-1]):
        ax3.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                 f'{prob:.2f}', va='center', fontsize=8)

    fig.suptitle(f'Kết quả dự đoán: {os.path.basename(image_path)}',
                 fontsize=13, fontweight='bold', y=1.02)
    
    # Cố định layout tránh bóp méo đồ thị hoặc đè chữ khi lặp loop
    fig.subplots_adjust(wspace=0.15, hspace=0, left=0.05, right=0.95, bottom=0.15, top=0.85)

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
            probs, positives, att_map, original_np = predict(
                model, device, image_path, img_size, threshold
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
        visualize(probs, positives, att_map, original_np,
                  image_path, save_path=save_path, threshold=threshold)
        print()


if __name__ == '__main__':
    main()