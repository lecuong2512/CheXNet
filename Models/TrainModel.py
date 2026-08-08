# TrainModel.py
import os
import time
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Optional
import copy

from Model import HybridCNNViTModel
from read_data import DatasetGenerator, FastDataLoader, HybridBatchSampler, VINDR_SOURCE_NAME
from checkpoint_utils import load_checkpoint_safe, extract_state_dict


def _unwrap_model(model):
    """Trích xuất model gốc từ DataParallel/DDP wrapper."""
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def _is_main_process():
    """True nếu là process chính (rank 0 hoặc không dùng DDP)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


class AsymmetricLossOptimized(nn.Module):
    """
    Asymmetric Loss (ASL) — Ridnik et al., ICCV 2021.
    Drop-in replacement cho BCEWithLogitsLoss, thiết kế chuyên biệt cho
    multi-label classification với class imbalance nghiêm trọng.

    Cơ chế hoạt động:
    1. Probability Shifting: dịch xác suất negative xuống thêm `clip`, khiến
       các ca True Negative dễ (model đã rất chắc) có loss ≈ 0 → gradient ≈ 0.
       Model "bỏ qua" ca dễ, dồn lực vào ca khó.
    2. Asymmetric Focusing: gamma_neg cao (4) phạt nhẹ negative dễ, gamma_pos
       thấp (0) giữ nguyên trọng số cho positive → ép model tập trung vào
       các ca bệnh thực sự.

    Tham số mặc định đã được validate trên MS-COCO, Open Images, và các
    bộ dữ liệu y tế có tính chất tương tự (positive rate < 5%).
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits, targets):
        # Sigmoid (nhận raw logits giống BCEWithLogitsLoss)
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos

        # Probability shifting — chỉ áp dụng cho negative
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic cross-entropy
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # Dùng torch.no_grad() thay vì set_grad_enabled(False/True) để an toàn
            # khi có exception — tránh tắt gradient vĩnh viễn cho toàn bộ training.
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    pt0 = xs_pos * targets
                    pt1 = xs_neg * (1 - targets)
                    pt = pt0 + pt1
                    one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                    one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            else:
                pt0 = xs_pos * targets
                pt1 = xs_neg * (1 - targets)
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.sum() / logits.shape[0]


class ModelEMA:
    """
    Exponential Moving Average of model parameters AND buffers.

    Duy trì bản sao "shadow" của trọng số VÀ buffers (BatchNorm running_mean/var),
    cập nhật sau mỗi optimizer step:
      shadow = decay * shadow + (1 - decay) * current

    Với decay=0.999, shadow là trung bình có trọng số của ~1000 step gần nhất.
    Giúp làm phẳng dao động SGD → bộ trọng số tổng quát hơn, đặc biệt hiệu quả
    khi model đã "chín" (resume từ checkpoint tốt).

    Workflow:
    1. Sau mỗi optimizer.step(): gọi ema.update(model)
    2. Trước validation: gọi ema.apply_shadow(model)
    3. Sau validation: gọi ema.restore(model)
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # Track cả parameters (weights) lẫn buffers (BatchNorm running_mean/var)
        # để tránh mismatch khi apply_shadow swap EMA weights nhưng giữ fast-training BN stats.
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        for name, buf in model.named_buffers():
            self.shadow[name] = buf.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        for name, buf in model.named_buffers():
            if name in self.shadow:
                # BatchNorm num_batches_tracked là int64, không thể mul_ với float decay.
                # Chỉ áp dụng EMA cho float buffers (running_mean/var); copy trực tiếp int buffers.
                if self.shadow[name].is_floating_point():
                    self.shadow[name].mul_(self.decay).add_(buf.data, alpha=1 - self.decay)
                else:
                    self.shadow[name].copy_(buf.data)

    def apply_shadow(self, model):
        """Swap model weights+buffers với EMA (dùng trước validation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        for name, buf in model.named_buffers():
            if name in self.shadow:
                self.backup[name] = buf.data.clone()
                buf.data.copy_(self.shadow[name])

    def restore(self, model):
        """Khôi phục trọng số training gốc (dùng sau validation)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        for name, buf in model.named_buffers():
            if name in self.backup:
                buf.data.copy_(self.backup[name])
        self.backup = {}


class DiceLoss(nn.Module):
    """
    Dice Loss + Focal Tversky bổ trợ, hỗ trợ MULTI-CHANNEL (1 kênh/bệnh).

    Với kiến trúc 15 attention map (1 map riêng cho mỗi bệnh), GT mask cũng có
    15 kênh tương ứng. Một ảnh VinDr thường chỉ có bbox cho 1-2 bệnh cụ thể;
    13 kênh còn lại không có annotation cho ảnh đó (KHÔNG đồng nghĩa "không có
    tổn thương" - chỉ đơn giản là không được đánh dấu). Vì vậy bắt buộc phải có
    `valid_mask` [N, num_classes] để chỉ tính Dice/Tversky trên đúng các kênh
    THỰC SỰ có ground-truth bbox cho ảnh đó - bỏ qua hoàn toàn các kênh không
    có annotation (không tính cả "đúng" lẫn "sai" trên kênh đó).

    - mask_smooth=0.0: KHÔNG làm mềm GT mask. Giá trị cũ (0.02) đã vô tình
      biến pixel nền (0→0.02) tạo ra "nền giả" chiếm >90% diện tích, khiến
      Dice/Tversky score bị kẹt ở mức rất thấp (~0.08) dù attention đã đúng.
    - tversky_beta > alpha: phạt nặng hơn False Negative (bỏ sót vùng tổn thương)
      so với False Positive, vì mục tiêu là hỗ trợ bác sĩ không bỏ sót tổn thương.
    - focal_gamma: mũ hóa (1 - Tversky) để tạo gradient DỐC HƠN ở vùng loss còn
      cao (attention chưa khớp vị trí) so với công thức Dice/Tversky tuyến tính
      thông thường.
    """
    def __init__(self, smooth=1.0, mask_smooth=0.0, tversky_alpha=0.3, tversky_beta=0.7,
                 tversky_weight=0.6, focal_gamma=0.75):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.mask_smooth = mask_smooth
        self.tversky_alpha = tversky_alpha  # phạt False Positive
        self.tversky_beta = tversky_beta    # phạt False Negative (ưu tiên cao hơn)
        self.tversky_weight = tversky_weight
        self.focal_gamma = focal_gamma
        self.last_raw_dice = 0.0  # Lưu raw Dice score (trước khi kết hợp Tversky) để logging

    def forward(self, preds, targets, valid_mask):
        """
        preds:      [N, num_classes, H, W] - attention map dự đoán
        targets:    [N, num_classes, H_orig, W_orig] - GT mask gốc (chưa resize)
        valid_mask: [N, num_classes] bool - True nếu kênh đó có GT bbox thật
        """
        # Nội suy mask gốc cho bằng kích thước attention map
        targets = F.interpolate(targets, size=preds.shape[2:], mode='bilinear', align_corners=False)

        # Label smoothing cho mask (mặc định 0.0 = không smooth)
        if self.mask_smooth > 0:
            targets = targets * (1 - 2 * self.mask_smooth) + self.mask_smooth

        # Chỉ giữ lại các kênh THỰC SỰ có GT (valid_mask=True), gom về dạng phẳng
        # theo từng (sample, class) hợp lệ rồi mới flatten không gian - đảm bảo
        # kênh không có annotation không đóng góp gì vào loss (không tính đúng,
        # không tính sai).
        valid_mask = valid_mask.bool()
        if valid_mask.sum() == 0:
            self.last_raw_dice = 0.0
            return preds.sum() * 0.0  # không có kênh hợp lệ nào -> loss = 0 nhưng vẫn giữ graph

        preds_valid = preds[valid_mask]      # [n_valid, H, W]
        targets_valid = targets[valid_mask]  # [n_valid, H, W]

        preds_flat = preds_valid.contiguous().view(-1)
        targets_flat = targets_valid.contiguous().view(-1)

        intersection = (preds_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        self.last_raw_dice = dice.item()  # Lưu raw Dice score để epoch loop in ra
        dice_loss = 1 - dice

        # Focal Tversky: TP, FP, FN tính trên giá trị liên tục (soft), sau đó mũ hóa
        # (1 - Tversky) bằng focal_gamma < 1 để "kéo dãn" gradient ở vùng loss cao.
        TP = (preds_flat * targets_flat).sum()
        FP = (preds_flat * (1 - targets_flat)).sum()
        FN = ((1 - preds_flat) * targets_flat).sum()
        tversky = (TP + self.smooth) / (TP + self.tversky_alpha * FP + self.tversky_beta * FN + self.smooth)
        focal_tversky_loss = (1 - tversky).clamp(min=1e-6) ** self.focal_gamma

        return (1 - self.tversky_weight) * dice_loss + self.tversky_weight * focal_tversky_loss


class AttentionSparsityLoss(nn.Module):
    """
    Regularization áp dụng cho TOÀN BỘ batch (cả ảnh không có ground-truth mask),
    nhằm chặn đường "trốn" của attention map mà Dice/Tversky loss không thấy được.

    Gồm 2 thành phần:
    1. L1 regularization: phạt mean activation — ép attention map thưa thớt
       (vùng nền → 0). Khác biệt so với bản cũ (target_density=0.15 hai chiều):
       bản cũ ép trung bình toàn bộ 15 kênh đạt 0.15, buộc model phải "bịa"
       activation ở 13-14 kênh không có bệnh (hallucination). Bản mới chỉ
       phạt L1 thuần (ép về 0), cho phép attention tự do kích hoạt ở bất kỳ
       mức nào miễn là có tín hiệu Dice/BCE hỗ trợ.
    2. Binary entropy: ép giá trị về gần 0 hoặc 1 (viền biên rõ) để
       bản đồ attention dễ đọc khi hỗ trợ bác sĩ quan sát.
    """
    def __init__(self, l1_weight=0.1, entropy_weight=0.05):
        super(AttentionSparsityLoss, self).__init__()
        self.l1_weight = l1_weight
        self.entropy_weight = entropy_weight

    def forward(self, attention_map):
        # L1: phạt mean activation → ép các vùng không có tổn thương về 0
        l1_loss = attention_map.mean()

        # Binary entropy: ép pixel về phân cực (gần 0 hoặc gần 1)
        eps = 1e-6
        a = attention_map.clamp(eps, 1 - eps)
        entropy = -(a * torch.log(a) + (1 - a) * torch.log(1 - a))
        entropy_loss = entropy.mean()

        return self.l1_weight * l1_loss + self.entropy_weight * entropy_loss

class UncertaintyWeighting(nn.Module):
    """Kendall & Gal 2018 — Uncertainty-based automatic multi-task loss weighting.
    
    Thay vì chỉnh tay trọng số loss (vd dice*2.0, sparsity*0.2), học log(σ²)
    cho mỗi task. Loss = Σ (1/(2σ²_i)) * L_i + log(σ²_i).
    Khi một loss giảm nhanh, σ²_i tự tăng (giảm trọng số) để model phân bổ
    capacity sang task khác.
    """
    def __init__(self, n_tasks=3):
        super().__init__()
        # Khởi tạo log(σ²) = 0 → σ² = 1 → trọng số ban đầu = 0.5
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, *losses, active_mask=None):
        """Tính weighted loss. active_mask: list[bool] — True nếu loss[i] có data thật,
        False nếu loss[i] = 0 do không có data (vd Dice khi batch không có VinDr).
        Khi active_mask[i]=False, bỏ qua hoàn toàn loss[i] và KHÔNG update log_vars[i]."""
        total = 0
        weighted_losses = []
        # Clamp log_vars để chặn trôi vô hạn khi bất kỳ loss component nào → 0.
        # Khoảng [-4, 4] → precision ∈ [0.018, 54.6] — đủ linh hoạt mà không bùng nổ.
        clamped_log_vars = torch.clamp(self.log_vars, min=-4.0, max=4.0)
        for i, loss in enumerate(losses):
            # Bỏ qua loss không có data — tránh log_vars[i] trôi khi loss luôn = 0
            if active_mask is not None and not active_mask[i]:
                weighted_losses.append(0.0)
                continue
            precision = torch.exp(-clamped_log_vars[i])  # 1/σ²
            # Đúng công thức Kendall & Gal 2018: 0.5 * (1/σ²) * L + 0.5 * log(σ²)
            weighted = 0.5 * precision * loss + 0.5 * clamped_log_vars[i]
            total += weighted
            weighted_losses.append(weighted.item())
        return total, weighted_losses

class HybridTrainer:
    CLASS_NAMES = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]

    @staticmethod
    def train(pathDirData: str, pathFileTrain: str, pathFileVal: str,
              model_size: str, img_size: int, trBatchSize: int, trMaxEpoch: int,
              pathModel: str = 'Trainedmodel/hybrid_model.pth',
              preload_images: bool = False, num_workers_preload: int = 8,
              resume_path: str = None, use_torch_compile: bool = False):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")
        
        # ── Tối ưu hóa GPU (A100/H100) ──
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # ── AMP (Automatic Mixed Precision) ──
        use_amp = device.type == 'cuda'
        if use_amp:
            # BF16 chỉ dùng trên Ampere+ (cc >= 8.0) — cuDNN conv kernels
            # không hỗ trợ BF16 trên cc < 8.0 (T4/V100 dùng FP16).
            compute_cap = torch.cuda.get_device_capability(0)
            amp_dtype = torch.bfloat16 if compute_cap >= (8, 0) else torch.float16
        else:
            amp_dtype = torch.float32
        # BF16 không cần loss scaling (cùng dynamic range với FP32)
        use_scaler = use_amp and (amp_dtype == torch.float16)
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
        if use_amp:
            dtype_name = 'BF16' if amp_dtype == torch.bfloat16 else 'FP16'
            print(f"⚡ AMP: {dtype_name} | GradScaler: {'ON' if use_scaler else 'OFF'}")
        
        # ---- Build Model
        print(f"\n🏗️ Building Hybrid Model ({model_size.upper()})...")
        model = HybridCNNViTModel(num_classes=15, model_size=model_size, img_size=img_size).to(device)
        
        # Multi-GPU: ưu tiên DDP > DataParallel
        #   DDP: mỗi GPU chạy 1 process riêng, không bị GIL bottleneck,
        #        throughput gần tuyến tính theo số GPU.
        #   DataParallel: dùng 1 process, bị GIL → chỉ hiệu quả ~60-70%.
        #
        # Để dùng DDP, khởi động bằng:
        #   torchrun --nproc_per_node=N main.py
        # Nếu không dùng torchrun, tự động fallback về DataParallel.
        use_ddp = False
        if torch.cuda.device_count() > 1:
            if 'LOCAL_RANK' in os.environ:
                # DDP: được khởi động bởi torchrun
                local_rank = int(os.environ['LOCAL_RANK'])
                if not dist.is_initialized():
                    dist.init_process_group(backend='nccl')
                torch.cuda.set_device(local_rank)
                device = torch.device(f'cuda:{local_rank}')
                model = model.to(device)
                model = DDP(model, device_ids=[local_rank])
                use_ddp = True
                if _is_main_process():
                    print(f"🚀 DDP: ENABLED ({torch.cuda.device_count()} GPUs, rank {dist.get_rank()})")
            else:
                # Fallback: DataParallel
                model = torch.nn.DataParallel(model)
                print(f"⚠️  DataParallel: {torch.cuda.device_count()} GPUs "
                      f"(dùng `torchrun --nproc_per_node={torch.cuda.device_count()} main.py` để tăng tốc với DDP)")

        # ── torch.compile (PyTorch 2.0+) ──
        # Công tắc cứng: multi-GPU (DataParallel/DDP) có thể gây lỗi với
        # torch.compile — chỉ bật khi use_torch_compile=True.
        if use_torch_compile and device.type == 'cuda' and hasattr(torch, 'compile'):
            try:
                compile_target = _unwrap_model(model)
                compiled = torch.compile(compile_target, mode='reduce-overhead')
                if isinstance(model, (nn.DataParallel, DDP)):
                    model.module = compiled
                else:
                    model = compiled
                print("⚡ torch.compile: ENABLED (reduce-overhead mode)")
            except Exception as e:
                print(f"⚠️  torch.compile không khả dụng: {e}")
        elif not use_torch_compile:
            print("ℹ️  torch.compile: DISABLED (use_torch_compile=False)")

        # ---- Resume từ checkpoint cũ nếu được yêu cầu
        resume_epoch = 0
        resume_stage = 1
        resume_best_auroc = 0.0
        resume_ckpt = None  # Lưu checkpoint dict để load optimizer/scheduler/scaler sau
        if resume_path and os.path.isfile(resume_path):
            print(f"\n🔄 Resume từ checkpoint: {resume_path}")
            ckpt = load_checkpoint_safe(resume_path, device)
            # M2 fix: Ưu tiên 'training_state_dict' (weights gốc) khi resume training.
            # 'state_dict' chứa EMA weights (quá mượt, phá momentum dynamics).
            # Backward-compatible: checkpoint cũ chỉ có 'state_dict' → dùng nó.
            if 'training_state_dict' in ckpt:
                state_dict = extract_state_dict({'state_dict': ckpt['training_state_dict']})
                print("ℹ️  Load TRAINING weights (không phải EMA) để resume training")
            else:
                state_dict = extract_state_dict(ckpt)
                print("ℹ️  Checkpoint cũ — dùng state_dict (có thể là EMA weights)")
            target_model = _unwrap_model(model)
            target_model.load_state_dict(state_dict)
            resume_epoch      = ckpt.get('epoch', 0)
            resume_stage      = ckpt.get('stage', 1)
            resume_best_auroc = ckpt.get('best_auroc', 0.0)
            ckpt_model_size   = ckpt.get('model_size', model_size)
            ckpt_img_size     = ckpt.get('img_size', img_size)
            if ckpt_model_size != model_size or ckpt_img_size != img_size:
                print(f"⚠️  Checkpoint: model_size={ckpt_model_size}, img_size={ckpt_img_size}")
                print(f"    Tham số hiện tại: model_size={model_size}, img_size={img_size}")
                print("    Đảm bảo hai bộ tham số khớp nhau để tránh lỗi khi load weights.")
            print(f"✅ Đã load checkpoint (epoch {resume_epoch}, stage {resume_stage}, "
                  f"best AUROC={resume_best_auroc:.4f})")
            print(f"   Sẽ tiếp tục từ epoch {resume_epoch + 1}, "
                  f"còn tối đa {trMaxEpoch} epoch nữa.")
            resume_ckpt = ckpt  # Giữ lại để load optimizer/scheduler/scaler sau
        elif resume_path:
            print(f"⚠️  Không tìm thấy checkpoint tại '{resume_path}' — bắt đầu train mới.")

        # ---- Data Transforms (Geometric only for Training to keep Mask sync)
        # Khi resume ở stage 2, model đã "thuộc bài" ảnh gốc → thêm nhiễu
        # màu sắc (ColorJitter) và blur (GaussianBlur) để chống overfit.
        #
        # QUAN TRỌNG: Trong read_data.py, ảnh (3ch) được GHÉP với mask (15ch)
        # thành tensor 18 kênh trước khi áp transform. Geometric transforms
        # (Crop, Flip, Rotate, Affine) hoạt động bình thường trên mọi số kênh,
        # nhưng ColorJitter/GaussianBlur YÊU CẦU 1 hoặc 3 kênh.
        # → Dùng wrapper ImageOnlyTransform để CHỈ áp color augmentation lên
        # 3 kênh đầu (ảnh), bỏ qua 15 kênh sau (mask).
        class ImageOnlyTransform(torch.nn.Module):
            """Áp dụng transform chỉ lên 3 kênh ảnh đầu tiên của tensor combined."""
            def __init__(self, transform):
                super().__init__()
                self.transform = transform
            def forward(self, combined):
                img = combined[:3]       # 3 kênh ảnh
                mask = combined[3:]      # 15 kênh mask
                img = self.transform(img)
                return torch.cat([img, mask], dim=0)

        color_augs = []
        if resume_stage == 2:
            color_augs = [
                ImageOnlyTransform(transforms.ColorJitter(brightness=0.15, contrast=0.15)),
                ImageOnlyTransform(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))),
            ]
        transformTrain = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ] + color_augs)
        
        transformVal = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size)
        ])

        print("\n📂 Loading datasets...")
        datasetTrain = DatasetGenerator(
            pathDirData, pathFileTrain, transformTrain,
            preload_images=preload_images, num_workers_preload=num_workers_preload
        )
        datasetVal = DatasetGenerator(
            pathDirData, pathFileVal, transformVal,
            preload_images=preload_images, num_workers_preload=num_workers_preload
        )

        # Lọc Index cho Giai đoạn 1 (Chỉ lấy VinDr)
        vin_indices_train = [i for i, s in enumerate(datasetTrain.sources) if s == VINDR_SOURCE_NAME]
        vin_subset_train = torch.utils.data.Subset(datasetTrain, vin_indices_train)
        
        num_workers = min(12, os.cpu_count() or 4)
        
        # DDP: dùng DistributedSampler để chia data giữa các GPU
        ddp_sampler_stage1 = None
        ddp_sampler_val = None
        if use_ddp:
            ddp_sampler_stage1 = torch.utils.data.distributed.DistributedSampler(
                vin_subset_train, shuffle=True
            )
            ddp_sampler_val = torch.utils.data.distributed.DistributedSampler(
                datasetVal, shuffle=False
            )

        # DataLoader GĐ1
        loader_stage1 = FastDataLoader.create_dataloader(
            vin_subset_train, batch_size=trBatchSize,
            shuffle=(ddp_sampler_stage1 is None), num_workers=num_workers,
            sampler=ddp_sampler_stage1
        )
        
        # DataLoader GĐ2 (Hybrid 1:3 Batch Sampler)
        # Phải dùng batch_sampler= vì HybridBatchSampler yield cả batch (list of indices)
        hybrid_sampler = HybridBatchSampler(datasetTrain.sources, trBatchSize)
        loader_stage2 = FastDataLoader.create_dataloader(
            datasetTrain, batch_size=trBatchSize, num_workers=num_workers, batch_sampler=hybrid_sampler
        )
        
        dataLoaderVal = FastDataLoader.create_dataloader(
            datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=num_workers,
            sampler=ddp_sampler_val
        )

        # ---- Optimizers & Loss
        # Discriminative LR: head mới (attention_head, fpn_lateral, fpn_merge)
        # khởi tạo ngẫu nhiên → cần LR cao hơn (5e-4) để bắt kịp backbone đã pretrained.
        # Backbone (ConvNeXtV2, SwinV2) giữ LR thấp (1e-4) để không phá pretrained features.
        raw_model = _unwrap_model(model)
        head_params_ids = set(
            id(p) for p in list(raw_model.attention_head.parameters())
            + list(raw_model.fpn_lateral.parameters())
            + list(raw_model.fpn_merge.parameters())
        )
        backbone_params = [p for p in model.parameters() if id(p) not in head_params_ids]
        head_params = [p for p in model.parameters() if id(p) in head_params_ids]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-4},
            {'params': head_params, 'lr': 5e-4},
        ], weight_decay=0.05)

        # ASL thay thế BCE: giảm trọng số các ca âm tính dễ, tập trung vào
        # ca dương tính và ca khó → đặc biệt hiệu quả khi positive rate < 5%
        # như dữ liệu X-quang (Hernia ~0.2%, Emphysema ~2%).
        criterion_bce = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05)
        # Giữ BCEWithLogitsLoss riêng cho validation để có val loss comparable
        criterion_val = nn.BCEWithLogitsLoss()
        criterion_dice = DiceLoss()
        criterion_sparsity = AttentionSparsityLoss()

        # Uncertainty weighting: tự động cân bằng trọng số BCE, Dice, Sparsity
        # QUAN TRỌNG: phải add_param_group TRƯỚC khi tạo scheduler, vì scheduler
        # ghi nhận số param groups lúc khởi tạo. Nếu add sau → scheduler chỉ
        # tạo LR cho 1 group nhưng optimizer có 2 → crash khi scheduler.step().
        uncertainty_weights = UncertaintyWeighting(n_tasks=3).to(device)
        optimizer.add_param_group({'params': uncertainty_weights.parameters(), 'lr': 1e-3})

        # Scheduler: warmup chỉ áp dụng khi bắt đầu stage 1 từ đầu.
        # Nếu resume vào stage 2 thì dùng CosineAnnealingLR thuần — model đã ổn định,
        # warmup lại từ lr=1e-5 sẽ làm chậm không cần thiết.
        if resume_stage == 2:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, trMaxEpoch),
                eta_min=1e-6
            )
        else:
            warmup_epochs = 3
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, trMaxEpoch - warmup_epochs), eta_min=1e-6
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )

        # ── EMA: khởi tạo từ trọng số hiện tại (đã load checkpoint nếu resume) ──
        # Khi resume, state_dict trong checkpoint đã chứa EMA weights (được lưu
        # bằng cách ghi đè params trong full_state). Model đã load state_dict đó,
        # nên EMA khởi tạo từ model.named_parameters() sẽ tự động có EMA weights.
        target_model_ref = _unwrap_model(model)
        ema = ModelEMA(target_model_ref, decay=0.999)
        if resume_path and os.path.isfile(resume_path):
            print("ℹ️  EMA khởi tạo từ checkpoint weights (đã chứa EMA params)")
        print(f"🎯 ASL: gamma_neg={criterion_bce.gamma_neg}, gamma_pos={criterion_bce.gamma_pos}, clip={criterion_bce.clip}")
        print(f"📊 EMA: decay={ema.decay}")

        # ── Khôi phục trạng thái optimizer/scheduler/scaler/uncertainty từ checkpoint ──
        if resume_ckpt is not None:
            if 'optimizer_state_dict' in resume_ckpt:
                try:
                    optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
                    print("✅ Đã khôi phục Optimizer state (momentum + variance)")
                except Exception as e:
                    print(f"⚠️  Không load được optimizer state: {e}")
            else:
                print("⚠️  Checkpoint cũ không chứa optimizer state — AdamW khởi tạo lại momentum")

            if 'scaler_state_dict' in resume_ckpt:
                try:
                    scaler.load_state_dict(resume_ckpt['scaler_state_dict'])
                    print("✅ Đã khôi phục GradScaler state")
                except Exception as e:
                    print(f"⚠️  Không load được scaler state: {e}")

            if 'scheduler_state_dict' in resume_ckpt:
                try:
                    # Kiểm tra tương thích: scheduler state từ checkpoint phải khớp
                    # số param groups hiện tại. Nếu optimizer đổi cấu trúc (vd thêm
                    # discriminative LR → 3 groups thay vì 2), scheduler cũ sẽ crash
                    # tại step() vì zip(param_groups, lr_values) lệch kích thước.
                    saved_sched = resume_ckpt['scheduler_state_dict']
                    n_optimizer_groups = len(optimizer.param_groups)
                    # SequentialLR lưu base_lrs trong sub-schedulers, không trực tiếp
                    # Kiểm tra qua _last_lr hoặc base_lrs của scheduler con
                    sched_compatible = True
                    if hasattr(scheduler, '_schedulers'):
                        # SequentialLR: kiểm tra sub-scheduler đầu tiên
                        for sub_sched_state in saved_sched.get('_schedulers', []):
                            if 'base_lrs' in sub_sched_state and len(sub_sched_state['base_lrs']) != n_optimizer_groups:
                                sched_compatible = False
                                break
                    elif 'base_lrs' in saved_sched and len(saved_sched['base_lrs']) != n_optimizer_groups:
                        sched_compatible = False

                    if sched_compatible:
                        scheduler.load_state_dict(saved_sched)
                        print("✅ Đã khôi phục Scheduler state")
                    else:
                        print(f"⚠️  Scheduler checkpoint không tương thích (checkpoint: {len(saved_sched.get('base_lrs', []))} groups, "
                              f"hiện tại: {n_optimizer_groups} groups) — tạo scheduler mới")
                except Exception as e:
                    print(f"⚠️  Không load được scheduler state (có thể do thay đổi T_max): {e}")

            if 'uncertainty_weights_state_dict' in resume_ckpt:
                try:
                    uncertainty_weights.load_state_dict(resume_ckpt['uncertainty_weights_state_dict'])
                    print(f"✅ Đã khôi phục UncertaintyWeighting (log_vars={uncertainty_weights.log_vars.data.tolist()})")
                except Exception as e:
                    print(f"⚠️  Không load được uncertainty_weights state: {e}")

            stage1_dice_met_ever = resume_ckpt.get('stage1_dice_met_ever', False)
            del resume_ckpt  # Giải phóng bộ nhớ
        else:
            stage1_dice_met_ever = False

        # Khởi tạo trạng thái training — dùng giá trị resume nếu có checkpoint
        bestAUROC = resume_best_auroc
        stage = resume_stage
        current_loader = loader_stage1 if stage == 1 else loader_stage2
        stage1_streak = 0

        # Early stopping: tách patience riêng cho từng stage.
        # Stage 1 (warm-up attention): patience cao hơn vì Dice cần nhiều epoch để
        # giảm dần từ baseline cao (~0.85) về ngưỡng chuyển stage, và val AUROC thường
        # thấp hơn thực lực (val set có ~89% ảnh NIH mà stage 1 chưa học). Cắt sớm ở
        # stage 1 lãng phí toàn bộ công sức warm-up attention.
        # Stage 2 (toàn bộ dữ liệu): patience thấp hơn vì mỗi epoch ~1.5h rất tốn kém.
        patience_stage1 = 10
        # Tăng patience stage 2 từ 5→8: khi đổi từ BCE sang ASL, 1-2 epoch đầu
        # AUROC có thể tạm giảm nhẹ trước khi bứt phá (gradient landscape thay
        # đổi). Cần đủ patience để không early stop sớm trong giai đoạn thích nghi.
        patience_stage2 = 8
        early_stop_patience = patience_stage1 if stage == 1 else patience_stage2
        epochs_no_improve = 0

        stage_label = f"Giai đoạn {stage}" if stage == 1 else "Giai đoạn 2 (Resume)"
        print(f"\n🚀 Bắt đầu {stage_label}: Warm-up Attention với Bounding Box")
        if resume_epoch > 0:
            print(f"   (Tiếp tục từ epoch {resume_epoch}, best AUROC={bestAUROC:.4f})")

        for epoch in range(trMaxEpoch):
            global_epoch = resume_epoch + epoch + 1
            print(f"\n{'='*60}\nEpoch [{global_epoch}] (session epoch {epoch+1}/{trMaxEpoch}) - STAGE {stage}\n{'='*60}")

            # DDP: set epoch cho DistributedSampler để shuffle khác mỗi epoch
            if use_ddp and stage == 1 and ddp_sampler_stage1 is not None:
                ddp_sampler_stage1.set_epoch(global_epoch)
            if use_ddp and ddp_sampler_val is not None:
                ddp_sampler_val.set_epoch(global_epoch)

            # Train
            no_finding_idx = HybridTrainer.CLASS_NAMES.index('No Finding')
            trainLoss, bceLoss, diceLoss, sparsityLoss, rawDice = HybridTrainer.epochTrain(
                model, current_loader, optimizer, criterion_bce, criterion_dice, criterion_sparsity, device, stage,
                no_finding_idx=no_finding_idx, scaler=scaler, amp_dtype=amp_dtype, ema=ema, uncertainty_weights=uncertainty_weights
            )

            # Val: dùng EMA weights để đánh giá (tổng quát hơn training weights)
            target_model_ref = _unwrap_model(model)
            ema.apply_shadow(target_model_ref)
            valLoss, valAUROC, valAcc = HybridTrainer.epochVal(
                model, dataLoaderVal, criterion_val, device,
                amp_dtype=amp_dtype, use_tta=True
            )
            ema.restore(target_model_ref)

            # In log_vars + precision để theo dõi UncertaintyWeighting phân bổ trọng số ra sao
            if uncertainty_weights is not None:
                log_vars = uncertainty_weights.log_vars.data.tolist()
                precisions = [float(torch.exp(-v).item()) for v in uncertainty_weights.log_vars.data]
                print(f"\n🔧 UncertaintyWeighting log_vars: ASL={log_vars[0]:.3f}, Dice={log_vars[1]:.3f}, Sparsity={log_vars[2]:.3f}")
                print(f"   Precision (1/σ²):             ASL={precisions[0]:.3f}, Dice={precisions[1]:.3f}, Sparsity={precisions[2]:.3f}")

            print(f"\nTrain - Total Loss (weighted): {trainLoss:.4f} | ASL: {bceLoss:.4f} | Dice Loss: {diceLoss:.6f} | Raw Dice Score: {rawDice:.4f} | Sparsity: {sparsityLoss:.4f}")
            print(f"Val   - Total Loss: {valLoss:.4f}   | AUROC: {valAUROC:.4f} | Acc: {valAcc:.4f}")

            scheduler.step()

            # Check Stage Transition
            if stage == 1:
                # Điều kiện chuyển stage 1 → 2 chỉ dựa trên DICE là chính:
                #
                # Lý do BỎ ngưỡng AUROC tuyệt đối (> 0.75):
                # Val set có ~89% ảnh NIH mà Stage 1 (chỉ train trên VinDr) chưa từng
                # học -> AUROC trên val không phản ánh đúng chất lượng attention đã học.
                # Yêu cầu AUROC > 0.75 là không thể đạt được ở stage 1 với cấu hình dữ
                # liệu này, bất kể attention có học tốt đến đâu.
                #
                # Thay vào đó: dùng Dice < ngưỡng làm điều kiện chính (đo trực tiếp
                # chất lượng attention so với GT bbox) + AUROC cải thiện so với baseline
                # đầu stage 1 (đo mô hình không bị overfit trên VinDr) + phải ổn định
                # liên tiếp 2 epoch để tránh chuyển do dao động ngẫu nhiên.
                #
                # Ngưỡng Dice 0.65: đã kiểm chứng thực nghiệm, khi Dice < 0.65 với Focal
                # Tversky (gamma=0.75), attention map đã định vị đúng không gian tổn thương
                # ở mức có ý nghĩa (không chỉ đúng mật độ trung bình).
                stage1_threshold_met = (diceLoss < 0.65) and (valAUROC > 0.68)
                if stage1_threshold_met:
                    if not stage1_dice_met_ever:
                        print(f"🎉 Lần đầu đạt chuẩn Dice! Lấy model này làm chuẩn mới (reset bestAUROC từ {bestAUROC:.4f} xuống {valAUROC:.4f})")
                        bestAUROC = valAUROC - 1.0  # Ép lưu model và update bestAUROC mới
                        stage1_dice_met_ever = True
                        
                    stage1_streak += 1
                    print(f"  ✓ Stage 1 threshold met ({stage1_streak}/2): Dice={diceLoss:.4f}<0.65, AUROC={valAUROC:.4f}>0.68")
                else:
                    stage1_streak = 0

                if stage1_streak >= 2:
                    print("\n🔓 Đạt ngưỡng tối ưu ổn định (2 epoch liên tiếp)! Chuyển sang Giai đoạn 2 (Toàn bộ dữ liệu + 1:3 Sampler)")
                    stage = 2
                    current_loader = loader_stage2
                    # Đổi sang patience thấp hơn cho stage 2 (mỗi epoch ~1.5h)
                    early_stop_patience = patience_stage2
                    epochs_no_improve = 0  # reset để stage 2 có đủ patience
                    # Reset sang CosineAnnealingLR thuần cho stage 2
                    # (không cần warmup lại vì model đã ổn định sau stage 1)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(1, trMaxEpoch - epoch),
                        eta_min=1e-6
                    )

            # Save Model & Early Stopping
            if valAUROC > bestAUROC:
                bestAUROC = valAUROC
                epochs_no_improve = 0
                
                # DDP: chỉ rank 0 lưu checkpoint (tránh race condition ghi file)
                if _is_main_process():
                    os.makedirs(os.path.dirname(pathModel) or '.', exist_ok=True)

                    # M2 fix: Lưu CẢ training weights LẪN EMA weights riêng biệt.
                    # - 'state_dict': EMA weights (dùng cho inference/test — tổng quát hơn)
                    # - 'training_state_dict': training weights gốc (dùng khi resume training
                    #   để bảo toàn momentum dynamics — EMA weights quá "mượt" để train tiếp)
                    ema_model = _unwrap_model(model)
                    # Deep copy training state: state_dict() trả về references,
                    # cần clone để tránh bị ảnh hưởng khi tạo ema_state bên dưới.
                    training_state = {k: v.clone() for k, v in ema_model.state_dict().items()}

                    # Tạo EMA state bằng cách ghi đè params+buffers bằng EMA shadow
                    ema_state = {k: v.clone() for k, v in training_state.items()}
                    for name, ema_val in ema.shadow.items():
                        if name in ema_state:
                            ema_state[name] = ema_val.cpu()

                    torch.save({
                        'epoch': global_epoch,
                        'state_dict': ema_state,           # EMA (cho inference)
                        'training_state_dict': training_state,  # Training (cho resume)
                        'best_auroc': bestAUROC,
                        'stage': stage,
                        'stage1_dice_met_ever': stage1_dice_met_ever,
                        'model_size': model_size,
                        'img_size': img_size,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'uncertainty_weights_state_dict': uncertainty_weights.state_dict(),
                    }, pathModel)
                    print(f"✅ Model saved (epoch {global_epoch}, AUROC: {bestAUROC:.4f})")
            else:
                epochs_no_improve += 1
                print(f"⏳ No improvement ({epochs_no_improve}/{early_stop_patience})")
                if epochs_no_improve >= early_stop_patience:
                    print(f"\n🛑 Early stopping triggered tại epoch {global_epoch} "
                          f"(patience={early_stop_patience})")
                    break

        # DDP cleanup
        if use_ddp and dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def epochTrain(model, dataLoader, optimizer, criterion_bce, criterion_dice, criterion_sparsity, device, stage,
                   no_finding_idx=0, scaler=None, amp_dtype=torch.float32, ema=None, uncertainty_weights=None):
        model.train()
        total_loss, total_bce, total_dice, total_sparsity = 0.0, 0.0, 0.0, 0.0
        total_raw_dice = 0.0  # Tổng raw Dice score (không phải loss) để theo dõi overlap thực tế
        n_dice_batches = 0
        attn_mean_sum, attn_std_sum = 0.0, 0.0
        
        use_amp = (amp_dtype != torch.float32)

        pbar = tqdm(dataLoader, desc="Training", ncols=100)
        for inputs, targets, masks, source_flags in pbar:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            source_flags = source_flags.to(device)
            
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                logits, attention_maps = model(inputs)

            # Tính loss ở FP32 để tránh lỗi NaN (bfloat16 precision khiến 1 - 1e-6 = 1.0 -> log(0) = NaN)
            logits = logits.float()
            attention_maps = attention_maps.float()

            # Theo dõi mean/std của attention map toàn batch để phát hiện collapse
            # (mô hình "lười" tô toàn 0 hoặc toàn 1 thay vì học vùng tổn thương thật)
            attn_mean_sum += attention_maps.mean().item()
            attn_std_sum += attention_maps.std().item()
            
            vin_mask = (source_flags == 1)
            
            # Tính BCE cho tất cả
            bce_loss = criterion_bce(logits, targets.float())
            total_bce += bce_loss.item()

            # Sparsity/Entropy regularization áp dụng cho TOÀN BỘ batch (kể cả ảnh
            # NIH không có GT mask). Đây là phần then chốt chặn đường "trốn" mà
            # Dice/Tversky không thấy được, vì Dice chỉ tính trên ~25% ảnh VinDr.
            # Nếu không có dòng này, attention vẫn có thể collapse về ~1.0 trên 75%
            # ảnh còn lại mà không bị phạt gì.
            sparsity_loss = criterion_sparsity(attention_maps)
            total_sparsity += sparsity_loss.item()
            
            dice_loss_val = torch.tensor(0.0, device=device)
            has_valid_dice = False  # Theo dõi: batch này có Dice data thực sự không?
            # Tính Dice Loss cho VinDr - CHỈ trên các kênh THỰC SỰ có GT bbox.
            # Quy tắc xác định kênh hợp lệ: với dữ liệu VinDr-CXR, mỗi nhãn dương
            # (label=1) cho 1 bệnh LUÔN đi kèm bbox tương ứng (đã kiểm chứng trên
            # dữ liệu thật: 100% dòng label=1 có bbox, trừ riêng "No Finding" -
            # bệnh này không có khái niệm vùng tổn thương/bbox). Vì vậy: kênh k
            # hợp lệ <=> targets[:, k] == 1 VÀ k != index của 'No Finding'.
            if vin_mask.sum() > 0:
                vin_attention = attention_maps[vin_mask]      # [N_vin, 15, H, W]
                vin_gt_masks = masks[vin_mask].float()        # [N_vin, 15, H, W]

                # Tính valid_mask dựa trên mask thực tế SAU augmentation, thay vì
                # label gốc. RandomResizedCrop/Affine/Rotation có thể đẩy bbox nhỏ
                # ra ngoài khung hình → mask rỗng hoàn toàn dù label vẫn = 1.
                # Nếu vẫn dùng label gốc, model bị phạt "không tìm thấy tổn thương"
                # ở kênh mà bbox đã bị cắt mất → gradient mâu thuẫn, kẹt Dice.
                valid_mask = (vin_gt_masks.sum(dim=(2, 3)) > 0)  # [N_vin, 15]
                valid_mask[:, no_finding_idx] = False  # No Finding không có bbox

                has_valid_dice = valid_mask.sum() > 0  # Có kênh bệnh thật sự có bbox
                dice_loss_val = criterion_dice(vin_attention, vin_gt_masks, valid_mask)
                # C2 fix: Chỉ đếm batch dice khi CÓ kênh hợp lệ thật sự,
                # tránh pha loãng avg_dice bằng các batch toàn "No Finding"
                if has_valid_dice:
                    total_dice += dice_loss_val.item()
                    total_raw_dice += criterion_dice.last_raw_dice
                    n_dice_batches += 1

            # Uncertainty weighting tự động cân bằng 3 loss components
            # C1 fix: active_mask[1]=False khi batch không có Dice data thật
            # → log_vars[1] (Dice) không bị update, tránh trôi xuống -4.0
            if uncertainty_weights is not None:
                dice_active = has_valid_dice
                loss, _ = uncertainty_weights(bce_loss, dice_loss_val, sparsity_loss,
                                              active_mask=[True, dice_active, True])
            else:
                loss = bce_loss + dice_loss_val * (2.0 if stage == 1 else 1.2) + sparsity_loss * 0.2

            # No Finding consistency: phạt khi P(No Finding) cao + max(P(bệnh)) cũng cao
            probs_all = torch.sigmoid(logits)
            p_no_finding = probs_all[:, no_finding_idx]
            p_diseases = torch.cat([probs_all[:, :no_finding_idx], probs_all[:, no_finding_idx+1:]], dim=1)
            p_max_disease = p_diseases.max(dim=1)[0]
            # Consistency penalty: p_no_finding * p_max_disease → 0 khi chỉ 1 trong 2 cao
            consistency_loss = (p_no_finding * p_max_disease).mean()
            loss = loss + consistency_loss * 0.1
                
            # AMP backward + optimizer step + gradient clipping
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # Chuyển gradient về FP32 scale trước khi clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if uncertainty_weights is not None:
                    torch.nn.utils.clip_grad_norm_(uncertainty_weights.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if uncertainty_weights is not None:
                    torch.nn.utils.clip_grad_norm_(uncertainty_weights.parameters(), max_norm=1.0)
                optimizer.step()

            # EMA update sau mỗi optimizer step
            if ema is not None:
                ema_target = _unwrap_model(model)
                ema.update(ema_target)

            total_loss += loss.item()

            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Spars': f'{sparsity_loss.item():.3f}'})

        n_batches = len(dataLoader)
        avg_attn_mean = attn_mean_sum / n_batches
        avg_attn_std = attn_std_sum / n_batches

        avg_sparsity = total_sparsity / n_batches

        # Luôn in mean/std của attention map để theo dõi tiến trình qua các epoch,
        # không chỉ lúc cảnh báo - giúp thấy được attention đang "co lại" đúng
        # hướng (mean giảm dần, std tăng dần) hay không.
        print(f"📐 Attention stats - mean: {avg_attn_mean:.3f} | std: {avg_attn_std:.3f}")
        if avg_attn_std < 0.05:
            print(f"⚠️  CẢNH BÁO: Attention map có dấu hiệu collapse "
                  f"(mean={avg_attn_mean:.3f}, std={avg_attn_std:.3f} < 0.05). "
                  f"Mô hình có thể đang tô gần như đồng nhất, không phản ánh đúng vùng tổn thương.")

        avg_dice = total_dice / n_dice_batches if n_dice_batches > 0 else 0.0
        avg_raw_dice = total_raw_dice / n_dice_batches if n_dice_batches > 0 else 0.0
        return total_loss / n_batches, total_bce / n_batches, avg_dice, avg_sparsity, avg_raw_dice

    @staticmethod
    def epochVal(model, dataLoader, criterion_bce, device, amp_dtype=torch.float32, use_tta=False):
        """
        Validation với hỗ trợ Test-Time Augmentation (TTA).
        
        Khi use_tta=True, mỗi ảnh được chạy qua model 2 lần:
        1. Ảnh gốc
        2. Ảnh lật ngang (horizontal flip)
        Kết quả = trung bình xác suất của 2 lần → ổn định hơn, AUROC +0.3-0.5%.
        
        Chi phí: thêm ~3 phút/epoch validation (1 forward pass nữa).
        """
        model.eval()
        total_loss = 0.0
        allPreds, allTargets = [], []
        use_amp = (amp_dtype != torch.float32)
        
        tta_label = "Validation+TTA" if use_tta else "Validation"
        pbar = tqdm(dataLoader, desc=tta_label, ncols=100)
        with torch.no_grad():
            for inputs, targets, _, _ in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    logits, _ = model(inputs)
                logits = logits.float()
                
                if use_tta:
                    # TTA: lật ngang ảnh và lấy trung bình PROBABILITIES (không phải logits)
                    # vì sigmoid là hàm phi tuyến: sigmoid((a+b)/2) ≠ (sigmoid(a)+sigmoid(b))/2
                    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                        logits_flip, _ = model(torch.flip(inputs, dims=[3]))
                    logits_flip = logits_flip.float()
                    avg_probs = (torch.sigmoid(logits) + torch.sigmoid(logits_flip)) / 2.0
                
                loss = criterion_bce(logits, targets.float())  # Val loss tính trên logits gốc
                
                total_loss += loss.item()
                
                if use_tta:
                    allPreds.append(avg_probs.cpu())
                else:
                    allPreds.append(torch.sigmoid(logits).cpu())
                allTargets.append(targets.float().cpu())
                
        allPreds = torch.cat(allPreds, dim=0).numpy()
        allTargets = torch.cat(allTargets, dim=0).numpy()
        
        auroc = HybridTrainer.computeAUROC_mean(allTargets, allPreds)
        acc = accuracy_score(allTargets.flatten(), (allPreds >= 0.5).astype(int).flatten())
        
        return total_loss / len(dataLoader), auroc, acc

    @staticmethod
    def computeAUROC_mean(dataGT, dataPRED):
        aurocIndividual = []
        for i in range(dataGT.shape[1]):
            try:
                aurocIndividual.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
            except Exception:
                aurocIndividual.append(np.nan)
        return np.nanmean(aurocIndividual)
