# Model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HybridCNNViTModel(nn.Module):
    """
    Hybrid ConvNeXtV2 + SwinV2 for Multi-label Classification with Residual Masking
    """
    def __init__(self, num_classes=15, model_size='base', img_size=384, dropout_rate=0.3):
        super(HybridCNNViTModel, self).__init__()
        
        # Chọn cấu hình cặp Backbone
        # Chọn cấu hình cặp Backbone
        if model_size == 'large':
            cnn_name = f'convnextv2_large.fcmae_ft_in22k_in1k_{img_size}'
            swin_name = 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'
            cnn_dim, swin_dim = 1536, 1536
        else: # base
            cnn_name = f'convnextv2_base.fcmae_ft_in22k_in1k_{img_size}'
            swin_name = 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k'
            cnn_dim, swin_dim = 1024, 1024

        # 1. Local Feature Extractor (CNN)
        self.cnn = timm.create_model(cnn_name, pretrained=True, features_only=True)
        
        # Projector để đồng bộ số kênh giữa CNN và ViT (nếu cần)
        self.channel_proj = nn.Conv2d(cnn_dim, swin_dim, kernel_size=1) if cnn_dim != swin_dim else nn.Identity()

        # 2. Global Context (ViT - Lấy stage cuối của SwinV2)
        swin_full = timm.create_model(swin_name, pretrained=True, num_classes=0, global_pool='')
        self.vit_blocks = swin_full.layers[-1] 
        if hasattr(self.vit_blocks, 'downsample') and self.vit_blocks.downsample is not None:
            self.vit_blocks.downsample = nn.Identity()
        self.vit_norm = swin_full.norm

        # 3. Attention Map Generator (Segmentation Head)
        self.attention_head = nn.Sequential(
            nn.Conv2d(swin_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(swin_dim, num_classes)
        )

        # Gate alpha: học được, khởi tạo ở đây để được đăng ký đúng với model.parameters()
        self.gate_alpha = nn.Parameter(torch.tensor([0.5]))

        # Gradient Checkpointing để tránh OOM khi train Large-Large
        if model_size == 'large':
            self.cnn.set_grad_checkpointing(True)
            swin_full.set_grad_checkpointing(True)

    def forward(self, x):
        # 1. Trích xuất đặc trưng cục bộ
        cnn_features = self.cnn(x)[-1] 
        cnn_features = self.channel_proj(cnn_features)
        
        # Chuẩn bị shape cho SwinV2
        vit_input = cnn_features.permute(0, 2, 3, 1) 
        
        # 2. Tính toán Attention toàn cục qua ViT
        vit_features = self.vit_blocks(vit_input)
        vit_features = self.vit_norm(vit_features)
        vit_features = vit_features.permute(0, 3, 1, 2)

        # 3. Sinh Attention Map (Mask) [B, 1, H, W]
        attention_map = self.attention_head(vit_features) 

        # ====================================================================
        # Gated Residual Attention với Scaling Factor (alpha)
        # gate_alpha được khởi tạo trong __init__ để optimizer theo dõi đúng
        # ====================================================================

        # Clamp attention để tránh "attention collapse"
        safe_attention = torch.clamp(attention_map, min=0.1, max=1.0)

        highlighted_features = cnn_features * safe_attention
        masked_features = cnn_features + self.gate_alpha * highlighted_features

        # 5. Global Average Pooling & Phân loại
        pooled = F.adaptive_avg_pool2d(masked_features, (1, 1)).flatten(1)
        logits = self.classifier(pooled)

        return logits, attention_map
