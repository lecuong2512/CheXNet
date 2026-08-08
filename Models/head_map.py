# head_map.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os

from Model import HybridCNNViTModel

DISEASE_COLUMNS = [
    'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


def _make_heatmap_overlay(att_map_2d, original_np, img_size):
    """Resize 1 attention map [H_feat, W_feat] lên kích thước ảnh, tô màu JET, chồng lên ảnh gốc."""
    att_map_resized = cv2.resize(att_map_2d, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    att_map_norm = np.uint8(255 * (att_map_resized - att_map_resized.min()) /
                             (att_map_resized.max() - att_map_resized.min() + 1e-8))
    heatmap = cv2.applyColorMap(att_map_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlaid = np.uint8(heatmap * 0.4 + original_np * 0.6)
    return heatmap, overlaid, att_map_resized


def generate_endogenous_attention_map(model_path, image_path, model_size='base', img_size=384,
                                       save_path=None, prob_threshold=0.5, max_diseases_shown=6):
    """
    Trích xuất Attention Map từ Segmentation Head của mô hình Hybrid.

    Mô hình sinh ra MỘT attention map RIÊNG cho MỖI bệnh (15 map, [1, 15, H, W]),
    không còn 1 map dùng chung cho mọi bệnh như trước. Hàm này hiển thị heatmap
    cho các bệnh mà mô hình dự đoán DƯƠNG TÍNH (xác suất >= prob_threshold) -
    đây chính là các vùng cần bác sĩ chú ý kiểm tra lại, mỗi bệnh có vùng khoanh
    riêng thay vì 1 vùng chung chung không rõ ứng với bệnh nào.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Load model {model_size.upper()} từ {model_path}...")
    
    model = HybridCNNViTModel(num_classes=len(DISEASE_COLUMNS), model_size=model_size, img_size=img_size)
    from checkpoint_utils import load_checkpoint_safe, extract_state_dict
    ckpt = load_checkpoint_safe(model_path, device)
    state_dict = extract_state_dict(ckpt)
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load và tiền xử lý ảnh
    original_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Forward qua mô hình
    with torch.no_grad():
        logits, attention_map = model(input_tensor)  # attention_map: [1, 15, H_feat, W_feat]
    
    # Lấy xác suất dự đoán (Sigmoid) cho từng bệnh
    probs = torch.sigmoid(logits)[0].cpu().numpy()  # [15]
    att_map_all = attention_map[0].cpu().numpy()    # [15, H_feat, W_feat]

    original_np = np.array(original_image.resize((img_size, img_size)))

    # Chọn các bệnh dương tính để hiển thị (sắp xếp theo xác suất giảm dần),
    # giới hạn số lượng hiển thị để hình không quá rối khi nhiều bệnh cùng dương tính
    positive_indices = [i for i in range(len(DISEASE_COLUMNS))
                         if DISEASE_COLUMNS[i] != 'No Finding' and probs[i] >= prob_threshold]
    positive_indices.sort(key=lambda i: probs[i], reverse=True)
    shown_indices = positive_indices[:max_diseases_shown]

    if not shown_indices:
        print(f"ℹ️  Không có bệnh nào vượt ngưỡng xác suất {prob_threshold} - "
              f"hiển thị heatmap của bệnh có xác suất cao nhất để tham khảo.")
        candidate = [i for i in range(len(DISEASE_COLUMNS)) if DISEASE_COLUMNS[i] != 'No Finding']
        shown_indices = [max(candidate, key=lambda i: probs[i])]

    n_panels = 1 + len(shown_indices)  # 1 ảnh gốc + 1 panel/bệnh (mỗi panel là overlay)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    axes[0].imshow(original_np)
    axes[0].set_title('Ảnh Gốc')
    axes[0].axis('off')

    att_maps_resized = {}
    for panel_idx, disease_idx in enumerate(shown_indices, start=1):
        disease_name = DISEASE_COLUMNS[disease_idx]
        _, overlaid, att_resized = _make_heatmap_overlay(att_map_all[disease_idx], original_np, img_size)
        att_maps_resized[disease_name] = att_resized
        axes[panel_idx].imshow(overlaid)
        axes[panel_idx].set_title(f'{disease_name}\n(p={probs[disease_idx]:.2f})')
        axes[panel_idx].axis('off')

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu bản đồ đặc trưng tại: {save_path}")
    else:
        plt.show()
    
    plt.close()

    # In tóm tắt xác suất toàn bộ 15 bệnh để bác sĩ tham khảo nhanh
    print("\n📋 Xác suất dự đoán từng bệnh:")
    for i in sorted(range(len(DISEASE_COLUMNS)), key=lambda i: probs[i], reverse=True):
        marker = "🔴" if probs[i] >= prob_threshold and DISEASE_COLUMNS[i] != 'No Finding' else "  "
        print(f"  {marker} {DISEASE_COLUMNS[i]:<22}: {probs[i]:.3f}")

    return probs, att_maps_resized
