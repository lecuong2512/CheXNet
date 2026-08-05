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
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Optional
import copy

from Model import HybridCNNViTModel
from read_data import DatasetGenerator, FastDataLoader, HybridBatchSampler, VINDR_SOURCE_NAME
from checkpoint_utils import load_checkpoint_safe, extract_state_dict


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
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum() / logits.shape[0]


class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    Duy trì bản sao "shadow" của trọng số, cập nhật sau mỗi optimizer step:
      shadow_param = decay * shadow_param + (1 - decay) * model_param

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
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """Swap model weights với EMA weights (dùng trước validation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Khôi phục trọng số training gốc (dùng sau validation)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
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

    - mask_smooth: làm mềm GT mask (0/1 -> eps/1-eps) để gradient không biến mất
      khi attention khớp gần hoàn hảo với bbox. Giá trị nhỏ (0.02) để không đè
      sàn loss lên quá cao khi mask rất thưa (nền chiếm >95% diện tích).
    - tversky_beta > alpha: phạt nặng hơn False Negative (bỏ sót vùng tổn thương)
      so với False Positive, vì mục tiêu là hỗ trợ bác sĩ không bỏ sót tổn thương.
    - focal_gamma: mũ hóa (1 - Tversky) để tạo gradient DỐC HƠN ở vùng loss còn
      cao (attention chưa khớp vị trí) so với công thức Dice/Tversky tuyến tính
      thông thường.
    """
    def __init__(self, smooth=1.0, mask_smooth=0.02, tversky_alpha=0.3, tversky_beta=0.7,
                 tversky_weight=0.6, focal_gamma=0.75):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.mask_smooth = mask_smooth
        self.tversky_alpha = tversky_alpha  # phạt False Positive
        self.tversky_beta = tversky_beta    # phạt False Negative (ưu tiên cao hơn)
        self.tversky_weight = tversky_weight
        self.focal_gamma = focal_gamma

    def forward(self, preds, targets, valid_mask):
        """
        preds:      [N, num_classes, H, W] - attention map dự đoán
        targets:    [N, num_classes, H_orig, W_orig] - GT mask gốc (chưa resize)
        valid_mask: [N, num_classes] bool - True nếu kênh đó có GT bbox thật
        """
        # Nội suy mask gốc cho bằng kích thước attention map
        targets = F.interpolate(targets, size=preds.shape[2:], mode='bilinear', align_corners=False)

        # Label smoothing cho mask: tránh GT cứng 0/1 khiến Dice/Tversky bão hòa về
        # một giá trị cố định khi attention học vẹt theo hình bbox
        targets = targets * (1 - 2 * self.mask_smooth) + self.mask_smooth

        # Chỉ giữ lại các kênh THỰC SỰ có GT (valid_mask=True), gom về dạng phẳng
        # theo từng (sample, class) hợp lệ rồi mới flatten không gian - đảm bảo
        # kênh không có annotation không đóng góp gì vào loss (không tính đúng,
        # không tính sai).
        valid_mask = valid_mask.bool()
        if valid_mask.sum() == 0:
            return preds.sum() * 0.0  # không có kênh hợp lệ nào -> loss = 0 nhưng vẫn giữ graph

        preds_valid = preds[valid_mask]      # [n_valid, H, W]
        targets_valid = targets[valid_mask]  # [n_valid, H, W]

        preds_flat = preds_valid.contiguous().view(-1)
        targets_flat = targets_valid.contiguous().view(-1)

        intersection = (preds_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
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
    1. L1 density (HAI PHÍA): phạt khi mean attention lệch khỏi target_density theo
       cả hai hướng (quá cao LẪN quá thấp). Thiết kế hai phía giải quyết "vicious
       cycle" của bản cũ (một phía, chỉ phạt khi quá cao): khi mean đã xuống dưới
       target, gradient entropy tiếp tục kéo xuống thêm mà không có gì kéo ngược
       lại, dẫn đến mean attention trôi xuống ~0.10 và attention không còn đủ tín
       hiệu để học vị trí tổn thương. Hai phía tạo điểm cân bằng ổn định tại
       target_density (mặc định 0.15, phù hợp diện tích tổn thương thực tế trên
       X-quang ngực, thấp hơn 0.25 cũ vốn quá cao so với annotation VinDr-CXR).
    2. Binary entropy (nhẹ hơn): ép giá trị về gần 0 hoặc 1 (viền biên rõ) để
       bản đồ attention dễ đọc khi hỗ trợ bác sĩ quan sát. Entropy_weight nhỏ hơn
       để không lấn át density loss và không tạo vicious cycle bổ sung.
    """
    def __init__(self, target_density=0.15, density_weight=1.0, entropy_weight=0.05):
        super(AttentionSparsityLoss, self).__init__()
        self.target_density = target_density
        self.density_weight = density_weight
        self.entropy_weight = entropy_weight

    def forward(self, attention_map):
        mean_activation = attention_map.mean()
        # Hai phía: phạt khi mean VƯỢT QUÁ hoặc XUỐNG THẤP HƠN target_density.
        # L1 distance tạo gradient hằng số (không phụ thuộc khoảng cách lệch)
        # -> dễ cân bằng hơn L2 và không bị explode gradient.
        density_loss = torch.abs(mean_activation - self.target_density)

        eps = 1e-6
        a = attention_map.clamp(eps, 1 - eps)
        entropy = -(a * torch.log(a) + (1 - a) * torch.log(1 - a))
        entropy_loss = entropy.mean()

        return self.density_weight * density_loss + self.entropy_weight * entropy_loss

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
              resume_path: str = None):
        
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
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # ── torch.compile (PyTorch 2.0+) ──
        if device.type == 'cuda' and hasattr(torch, 'compile'):
            try:
                compile_target = model.module if isinstance(model, nn.DataParallel) else model
                compiled = torch.compile(compile_target, mode='reduce-overhead')
                if isinstance(model, nn.DataParallel):
                    model.module = compiled
                else:
                    model = compiled
                print("⚡ torch.compile: ENABLED (reduce-overhead mode)")
            except Exception as e:
                print(f"⚠️  torch.compile không khả dụng: {e}")

        # ---- Resume từ checkpoint cũ nếu được yêu cầu
        resume_epoch = 0
        resume_stage = 1
        resume_best_auroc = 0.0
        if resume_path and os.path.isfile(resume_path):
            print(f"\n🔄 Resume từ checkpoint: {resume_path}")
            ckpt = load_checkpoint_safe(resume_path, device)
            state_dict = extract_state_dict(ckpt)
            target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
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
        
        # DataLoader GĐ1
        loader_stage1 = FastDataLoader.create_dataloader(
            vin_subset_train, batch_size=trBatchSize, shuffle=True, num_workers=num_workers
        )
        
        # DataLoader GĐ2 (Hybrid 1:3 Batch Sampler)
        # Phải dùng batch_sampler= vì HybridBatchSampler yield cả batch (list of indices)
        hybrid_sampler = HybridBatchSampler(datasetTrain.sources, trBatchSize)
        loader_stage2 = FastDataLoader.create_dataloader(
            datasetTrain, batch_size=trBatchSize, num_workers=num_workers, batch_sampler=hybrid_sampler
        )
        
        dataLoaderVal = FastDataLoader.create_dataloader(
            datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=num_workers
        )

        # ---- Optimizers & Loss
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

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

        # ASL thay thế BCE: giảm trọng số các ca âm tính dễ, tập trung vào
        # ca dương tính và ca khó → đặc biệt hiệu quả khi positive rate < 5%
        # như dữ liệu X-quang (Hernia ~0.2%, Emphysema ~2%).
        criterion_bce = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05)
        # Giữ BCEWithLogitsLoss riêng cho validation để có val loss comparable
        criterion_val = nn.BCEWithLogitsLoss()
        criterion_dice = DiceLoss()
        criterion_sparsity = AttentionSparsityLoss()

        # ── EMA: khởi tạo từ trọng số hiện tại (đã load checkpoint nếu resume) ──
        # Khi resume, state_dict trong checkpoint đã chứa EMA weights (được lưu
        # bằng cách ghi đè params trong full_state). Model đã load state_dict đó,
        # nên EMA khởi tạo từ model.named_parameters() sẽ tự động có EMA weights.
        target_model_ref = model.module if isinstance(model, torch.nn.DataParallel) else model
        ema = ModelEMA(target_model_ref, decay=0.999)
        if resume_path and os.path.isfile(resume_path):
            print("ℹ️  EMA khởi tạo từ checkpoint weights (đã chứa EMA params)")
        print(f"🎯 ASL: gamma_neg={criterion_bce.gamma_neg}, gamma_pos={criterion_bce.gamma_pos}, clip={criterion_bce.clip}")
        print(f"📊 EMA: decay={ema.decay}")

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

            # Train
            no_finding_idx = HybridTrainer.CLASS_NAMES.index('No Finding')
            trainLoss, bceLoss, diceLoss, sparsityLoss = HybridTrainer.epochTrain(
                model, current_loader, optimizer, criterion_bce, criterion_dice, criterion_sparsity, device, stage,
                no_finding_idx=no_finding_idx, scaler=scaler, amp_dtype=amp_dtype, ema=ema
            )

            # Val: dùng EMA weights để đánh giá (tổng quát hơn training weights)
            target_model_ref = model.module if isinstance(model, torch.nn.DataParallel) else model
            ema.apply_shadow(target_model_ref)
            valLoss, valAUROC, valAcc = HybridTrainer.epochVal(
                model, dataLoaderVal, criterion_val, device,
                amp_dtype=amp_dtype, use_tta=True
            )
            ema.restore(target_model_ref)

            print(f"\nTrain - Total Loss: {trainLoss:.4f} | ASL: {bceLoss:.4f} | Dice: {diceLoss:.6f} | Sparsity: {sparsityLoss:.4f}")
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
                # Ngưỡng Dice 0.60: đã kiểm chứng thực nghiệm, khi Dice < 0.60 với Focal
                # Tversky (gamma=0.75), attention map đã định vị đúng không gian tổn thương
                # ở mức có ý nghĩa (không chỉ đúng mật độ trung bình).
                stage1_threshold_met = (diceLoss < 0.65) and (valAUROC > 0.68)
                if stage1_threshold_met:
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
                os.makedirs(os.path.dirname(pathModel) or '.', exist_ok=True)

                # Lưu state_dict đầy đủ: lấy full model state (bao gồm cả
                # buffers như BatchNorm running_mean/running_var) rồi ghi đè
                # các PARAMETERS bằng trọng số EMA (tổng quát hơn training weights).
                # Cách cũ (chỉ lưu ema.shadow) thiếu buffers → lỗi Missing key
                # khi load lại cho test/inference.
                ema_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                full_state = ema_model.state_dict()  # copy đầy đủ cả params + buffers
                for name, ema_param in ema.shadow.items():
                    if name in full_state:
                        full_state[name] = ema_param.cpu()
                torch.save({
                    'epoch': global_epoch,
                    'state_dict': full_state,
                    'best_auroc': bestAUROC,
                    'stage': stage,
                    'model_size': model_size,
                    'img_size': img_size,
                }, pathModel)
                print(f"✅ Model saved (epoch {global_epoch}, AUROC: {bestAUROC:.4f})")
            else:
                epochs_no_improve += 1
                print(f"⏳ No improvement ({epochs_no_improve}/{early_stop_patience})")
                if epochs_no_improve >= early_stop_patience:
                    print(f"\n🛑 Early stopping triggered tại epoch {global_epoch} "
                          f"(patience={early_stop_patience})")
                    break

    @staticmethod
    def epochTrain(model, dataLoader, optimizer, criterion_bce, criterion_dice, criterion_sparsity, device, stage,
                   no_finding_idx=0, scaler=None, amp_dtype=torch.float32, ema=None):
        model.train()
        total_loss, total_bce, total_dice, total_sparsity = 0.0, 0.0, 0.0, 0.0
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
            
            loss = 0
            # Tính BCE cho tất cả
            bce_loss = criterion_bce(logits, targets.float())
            loss += bce_loss
            total_bce += bce_loss.item()

            # Sparsity/Entropy regularization áp dụng cho TOÀN BỘ batch (kể cả ảnh
            # NIH không có GT mask). Đây là phần then chốt chặn đường "trốn" mà
            # Dice/Tversky không thấy được, vì Dice chỉ tính trên ~25% ảnh VinDr.
            # Nếu không có dòng này, attention vẫn có thể collapse về ~1.0 trên 75%
            # ảnh còn lại mà không bị phạt gì.
            sparsity_loss = criterion_sparsity(attention_maps)
            loss += sparsity_loss * 0.2  # giảm từ 0.3 để tránh sparsity lấn át Dice khi đã đạt density target
            total_sparsity += sparsity_loss.item()
            
            # Tính Dice Loss cho VinDr - CHỈ trên các kênh THỰC SỰ có GT bbox.
            # Quy tắc xác định kênh hợp lệ: với dữ liệu VinDr-CXR, mỗi nhãn dương
            # (label=1) cho 1 bệnh LUÔN đi kèm bbox tương ứng (đã kiểm chứng trên
            # dữ liệu thật: 100% dòng label=1 có bbox, trừ riêng "No Finding" -
            # bệnh này không có khái niệm vùng tổn thương/bbox). Vì vậy: kênh k
            # hợp lệ <=> targets[:, k] == 1 VÀ k != index của 'No Finding'.
            if vin_mask.sum() > 0:
                vin_attention = attention_maps[vin_mask]      # [N_vin, 15, H, W]
                vin_gt_masks = masks[vin_mask].float()        # [N_vin, 15, H_orig, W_orig]
                vin_labels = targets[vin_mask]                # [N_vin, 15]

                valid_mask = (vin_labels > 0.5)
                valid_mask[:, no_finding_idx] = False  # No Finding không có bbox

                dice_loss = criterion_dice(vin_attention, vin_gt_masks, valid_mask)
                # GĐ1: trọng số RẤT cao vì đây là giai đoạn warm-up CHUYÊN BIỆT cho
                # attention (mục tiêu chính, không phải phụ) - toàn bộ batch GĐ1 đều
                # là VinDr có GT nên không lo Dice bị "loãng" giữa các ảnh không có mask.
                # GĐ2: vẫn giữ trọng số cao hơn baseline cũ (1.2) vì 75% ảnh không có GT
                # mask -> nếu giảm giám sát Dice ngay lúc này, attention_head dễ "lười"
                # và bão hòa để chỉ tối ưu BCE, khiến attention không còn phản ánh đúng
                # vùng tổn thương.
                loss += dice_loss * (2.0 if stage == 1 else 1.2)
                total_dice += dice_loss.item()
                n_dice_batches += 1
                
            # AMP backward + optimizer step
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # EMA update sau mỗi optimizer step
            if ema is not None:
                ema_target = model.module if isinstance(model, torch.nn.DataParallel) else model
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
        return total_loss / n_batches, total_bce / n_batches, avg_dice, avg_sparsity

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
                    # TTA: lật ngang ảnh và lấy trung bình logits
                    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                        logits_flip, _ = model(torch.flip(inputs, dims=[3]))
                    logits_flip = logits_flip.float()
                    logits = (logits + logits_flip) / 2.0
                
                loss = criterion_bce(logits, targets.float())
                
                total_loss += loss.item()
                
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
