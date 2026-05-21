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

def generate_endogenous_attention_map(model_path, image_path, model_size='base', img_size=384, save_path=None):
    """
    Trích xuất trực tiếp Attention Map từ Segmentation Head của mô hình Hybrid
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Load model {model_size.upper()} từ {model_path}...")
    
    model = HybridCNNViTModel(num_classes=15, model_size=model_size, img_size=img_size)
    try:
        import torch.serialization, numpy as _np
        torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    # Xử lý tiền tố module. nếu train bằng DataParallel
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        state_dict = OrderedDict((k[7:], v) if k.startswith('module.') else (k, v) for k, v in state_dict.items())
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load và tiền xử lý ảnh
    original_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Forward qua mô hình
    with torch.no_grad():
        logits, attention_map = model(input_tensor) # attention_map shape: [1, 1, H_feat, W_feat]
    
    # Lấy xác suất dự đoán (Sigmoid)
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    
    # Xử lý Attention Map
    att_map_np = attention_map.squeeze().cpu().numpy() # Bỏ batch và channel dims
    
    # Resize map lên kích thước ảnh gốc bằng phép nội suy
    att_map_resized = cv2.resize(att_map_np, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    
    # Chuẩn hóa về dải [0, 255] để lên màu
    att_map_norm = np.uint8(255 * (att_map_resized - att_map_resized.min()) / (att_map_resized.max() - att_map_resized.min() + 1e-8))
    
    # Phủ màu JET
    heatmap = cv2.applyColorMap(att_map_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Chồng ảnh
    original_np = np.array(original_image.resize((img_size, img_size)))
    overlaid = np.uint8(heatmap * 0.4 + original_np * 0.6)
    
    # Vẽ đồ thị
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_np)
    axes[0].set_title('Ảnh Gốc')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Attention Map (Nội sinh)')
    axes[1].axis('off')
    
    axes[2].imshow(overlaid)
    axes[2].set_title('Kết quả Masking')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu bản đồ đặc trưng tại: {save_path}")
    else:
        plt.show()
    
    plt.close()
    return probs, att_map_resized
