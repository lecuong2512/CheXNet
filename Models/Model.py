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
        # Lưu ý: ConvNeXtV2 pretrained weights chỉ tồn tại cho một số resolution
        # cụ thể. Khi img_size không khớp (vd 256), dùng tag gần nhất — timm tự
        # xử lý resize nội bộ (interpolate position embeddings nếu có).
        _CONVNEXTV2_TAGS = {
            'large': {
                384: 'convnextv2_large.fcmae_ft_in22k_in1k_384',
                256: 'convnextv2_large.fcmae_ft_in22k_in1k_384',  # fallback
            },
            'base': {
                384: 'convnextv2_base.fcmae_ft_in22k_in1k_384',
                256: 'convnextv2_base.fcmae_ft_in1k',  # pretrained @224, OK cho 256
            },
        }
        if model_size == 'large':
            cnn_name = _CONVNEXTV2_TAGS['large'].get(img_size, 'convnextv2_large.fcmae_ft_in22k_in1k_384')
            swin_name = 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'
            cnn_dim, swin_dim = 1536, 1536
        else: # base
            cnn_name = _CONVNEXTV2_TAGS['base'].get(img_size, 'convnextv2_base.fcmae_ft_in1k')
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
        # Output num_classes kênh (1 attention map RIÊNG cho mỗi bệnh) thay vì 1
        # map dùng chung. Mỗi kênh học cách khoanh vùng tổn thương đặc trưng cho
        # đúng bệnh đó (vd kênh Cardiomegaly học khoanh vùng tim to, kênh Mass
        # học khoanh vùng khối u...).
        self.num_classes = num_classes
        self.attention_head = nn.Sequential(
            nn.Conv2d(swin_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

        # Chia kênh của cnn_features (swin_dim kênh) thành num_classes NHÓM để
        # mỗi nhóm được gate riêng bởi đúng attention map của bệnh tương ứng.
        # swin_dim không chia hết cho num_classes (vd 1024/15) -> các nhóm đầu
        # nhận floor(swin_dim/num_classes) kênh, nhóm CUỐI nhận phần dư còn lại
        # (không đều, nhưng đơn giản và không làm mất/lặp kênh nào).
        base_group_size = swin_dim // num_classes
        self.channel_group_sizes = [base_group_size] * (num_classes - 1)
        self.channel_group_sizes.append(swin_dim - base_group_size * (num_classes - 1))
        assert sum(self.channel_group_sizes) == swin_dim

        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(swin_dim, num_classes)
        )

        # Gate alpha: học được nhưng bị giới hạn trong [0, max_gate_alpha] qua Sigmoid.
        # Khởi tạo thấp (tương đương alpha~0.3 sau sigmoid) và chặn trần 0.8 để nhánh
        # BCE không thể "ăn gian" bằng cách đẩy gate lên quá cao - nếu gate_alpha lớn
        # và attention_map collapse về ~1.0 khắp nơi, residual gần như nhân đôi toàn bộ
        # feature mà không cần học đúng vùng tổn thương. Giới hạn trần giữ ảnh hưởng
        # của attention luôn ở mức có ý nghĩa nhưng không quá áp đảo.
        self.max_gate_alpha = 0.8
        self.gate_alpha_raw = nn.Parameter(torch.tensor([-0.5]))  # sigmoid(-0.5)*0.8 ≈ 0.30

        # Gradient Checkpointing để tránh OOM khi train Large-Large
        if model_size == 'large':
            self.cnn.set_grad_checkpointing(True)
            # Bật grad checkpointing trên swin_full TRƯỚC khi trích xuất layers,
            # timm SwinV2 propagate flag xuống từng SwinTransformerV2Stage.
            # Verify bằng cách bật trên cả swin_full (propagate) lẫn trực tiếp
            # trên self.vit_blocks nếu có method.
            swin_full.set_grad_checkpointing(True)
            if hasattr(self.vit_blocks, 'set_grad_checkpointing'):
                self.vit_blocks.set_grad_checkpointing(True)
            elif hasattr(self.vit_blocks, 'grad_checkpointing'):
                self.vit_blocks.grad_checkpointing = True

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

        # 3. Sinh Attention Map (Mask) [B, num_classes, H, W] - 1 kênh/bệnh
        attention_map = self.attention_head(vit_features) 

        # ====================================================================
        # Gated Residual Attention theo NHÓM KÊNH (per-class channel grouping)
        # gate_alpha được khởi tạo trong __init__ để optimizer theo dõi đúng
        #
        # Mỗi bệnh có 1 nhóm kênh CNN feature riêng (self.channel_group_sizes)
        # VÀ 1 attention map riêng (attention_map[:, k]). Nhóm kênh thứ k chỉ bị
        # gate bởi đúng attention map của bệnh thứ k - không trộn lẫn thông tin
        # không gian giữa các bệnh khác nhau. Sau đó ghép lại đúng thứ tự kênh
        # ban đầu để vào classifier.
        #
        # KHÔNG còn clamp(min=0.1) như trước: clamp cũ tạo "sàn an toàn" khiến
        # attention_map có thể đổ về ~1.0 ở MỌI pixel mà vẫn không bị phạt gì ở
        # nhánh classification. Bỏ sàn này để nhánh BCE thực sự phụ thuộc vào
        # việc attention phải khoanh đúng vùng.
        # ====================================================================

        gate_alpha = torch.sigmoid(self.gate_alpha_raw) * self.max_gate_alpha
        gated_groups = []
        ch_start = 0
        for k, group_size in enumerate(self.channel_group_sizes):
            ch_end = ch_start + group_size
            group_features = cnn_features[:, ch_start:ch_end, :, :]
            group_attention = attention_map[:, k:k+1, :, :]  # [B,1,H,W], broadcast theo kênh
            highlighted_group = group_features * group_attention
            gated_group = group_features + gate_alpha * highlighted_group
            gated_groups.append(gated_group)
            ch_start = ch_end
        masked_features = torch.cat(gated_groups, dim=1)

        # 5. Global Average Pooling & Phân loại
        pooled = F.adaptive_avg_pool2d(masked_features, (1, 1)).flatten(1)
        logits = self.classifier(pooled)

        return logits, attention_map
