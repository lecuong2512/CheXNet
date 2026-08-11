# TrainModel_finetune.py
# ============================================================================
# Fine-tune Trainer — Tách biệt hoàn toàn với TrainModel.py gốc
#
# Thay đổi so với HybridTrainer gốc:
# 1. LR backbone: 1e-4 → 3e-5 (đã hội tụ, chỉ tinh chỉnh)
# 2. LR head:     5e-4 → 1e-4 (đã học đủ, ổn định)
# 3. Weight decay: 0.05 → 0.08 (tăng regularization)
# 4. ASL gamma_neg: 4 → 2 (nới lỏng, cải thiện ranking/AUROC)
# 5. EMA decay:    0.999 → 0.9995 (ổn định hơn với LR thấp)
# 6. UW clamp:     [-4,4] → [-3,3] (ngăn Dice bị "bỏ rơi")
# 7. Scheduler:    CosineAnnealing T_max=trMaxEpoch (chu kỳ ngắn)
# 8. Color Augmentation: LUÔN BẬT (fix bug gốc)
# 9. Checkpoint:   Chỉ lưu state_dict + metadata (~1.2GB thay vì ~4.2GB)
# 10. Bỏ qua Stage 1 hoàn toàn, bắt đầu trực tiếp Stage 2
# ============================================================================
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


def _unwrap_model(model):
    """Trích xuất model gốc từ DataParallel wrapper."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


# ============================================================================
# Loss Functions (sao chép từ TrainModel.py, chỉ thay đổi ASL gamma_neg)
# ============================================================================

class AsymmetricLossOptimized(nn.Module):
    """ASL với gamma_neg=2 (nới lỏng hơn so với gốc gamma_neg=4).
    
    Giảm gamma_neg giúp mô hình không bị phạt quá nặng khi đoán sai ca âm tính,
    cho phép xác suất dự đoán phân bố rộng hơn → cải thiện khả năng ranking (AUROC).
    """
    def __init__(self, gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-4, disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
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
    """EMA với decay=0.9995 (ổn định hơn so với gốc 0.999)."""
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
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
                if self.shadow[name].is_floating_point():
                    self.shadow[name].mul_(self.decay).add_(buf.data, alpha=1 - self.decay)
                else:
                    self.shadow[name].copy_(buf.data)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        for name, buf in model.named_buffers():
            if name in self.shadow:
                self.backup[name] = buf.data.clone()
                buf.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        for name, buf in model.named_buffers():
            if name in self.backup:
                buf.data.copy_(self.backup[name])
        self.backup = {}


class DiceLoss(nn.Module):
    """Dice Loss + Focal Tversky (giữ nguyên từ code gốc)."""
    def __init__(self, smooth=1e-4, mask_smooth=0.0, tversky_alpha=0.3, tversky_beta=0.7,
                 tversky_weight=0.6, focal_gamma=0.75):
        super().__init__()
        self.smooth = smooth
        self.mask_smooth = mask_smooth
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_weight = tversky_weight
        self.focal_gamma = focal_gamma
        self.last_raw_dice = 0.0

    def forward(self, preds, targets, valid_mask):
        targets = F.interpolate(targets, size=preds.shape[2:], mode='bilinear', align_corners=False)
        if self.mask_smooth > 0:
            targets = targets * (1 - 2 * self.mask_smooth) + self.mask_smooth
        valid_mask = valid_mask.bool()
        if valid_mask.sum() == 0:
            self.last_raw_dice = 0.0
            return preds.sum() * 0.0
        preds_valid = preds[valid_mask]
        targets_valid = targets[valid_mask]
        preds_flat = preds_valid.contiguous().view(-1)
        targets_flat = targets_valid.contiguous().view(-1)
        intersection = (preds_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        self.last_raw_dice = dice.item()
        dice_loss = 1 - dice
        TP = (preds_flat * targets_flat).sum()
        FP = (preds_flat * (1 - targets_flat)).sum()
        FN = ((1 - preds_flat) * targets_flat).sum()
        tversky = (TP + self.smooth) / (TP + self.tversky_alpha * FP + self.tversky_beta * FN + self.smooth)
        focal_tversky_loss = (1 - tversky).clamp(min=1e-6) ** self.focal_gamma
        return (1 - self.tversky_weight) * dice_loss + self.tversky_weight * focal_tversky_loss


class AttentionSparsityLoss(nn.Module):
    """Sparsity regularization (giữ nguyên từ code gốc)."""
    def __init__(self, l1_weight=0.1, entropy_weight=0.05):
        super().__init__()
        self.l1_weight = l1_weight
        self.entropy_weight = entropy_weight

    def forward(self, attention_map):
        l1_loss = attention_map.mean()
        eps = 1e-4
        a = attention_map.clamp(eps, 1 - eps)
        entropy = -(a * torch.log(a) + (1 - a) * torch.log(1 - a))
        entropy_loss = entropy.mean()
        return self.l1_weight * l1_loss + self.entropy_weight * entropy_loss


class UncertaintyWeighting(nn.Module):
    """UW với clamp [-3, 3] (thu hẹp hơn so với gốc [-4, 4])."""
    def __init__(self, n_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *losses, active_mask=None):
        total = 0
        weighted_losses = []
        # Thu hẹp clamp: [-3, 3] → precision ∈ [0.05, 20.1]
        clamped_log_vars = torch.clamp(self.log_vars, min=-3.0, max=3.0)
        for i, loss in enumerate(losses):
            if active_mask is not None and not active_mask[i]:
                weighted_losses.append(0.0)
                continue
            precision = torch.exp(-clamped_log_vars[i])
            weighted = 0.5 * precision * loss + 0.5 * clamped_log_vars[i]
            total += weighted
            weighted_losses.append(weighted.item())
        return total, weighted_losses


# ============================================================================
# FinetuneTrainer
# ============================================================================

class FinetuneTrainer:
    CLASS_NAMES = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]

    @staticmethod
    def train(pathDirData: str, pathFileTrain: str, pathFileVal: str,
              model_size: str, img_size: int, trBatchSize: int, trMaxEpoch: int,
              pathModel: str, checkpoint_path: str,
              preload_images: bool = False, num_workers_preload: int = 8,
              num_workers_train: int = 4):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")

        # ── GPU Optimizations ──
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # ── AMP ──
        use_amp = device.type == 'cuda'
        if use_amp:
            compute_cap = torch.cuda.get_device_capability(0)
            amp_dtype = torch.bfloat16 if compute_cap >= (8, 0) else torch.float16
        else:
            amp_dtype = torch.float32
        use_scaler = use_amp and (amp_dtype == torch.float16)
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
        if use_amp:
            dtype_name = 'BF16' if amp_dtype == torch.bfloat16 else 'FP16'
            print(f"⚡ AMP: {dtype_name} | GradScaler: {'ON' if use_scaler else 'OFF'}")

        # ── Build Model ──
        print(f"\n🏗️ Building Hybrid Model ({model_size.upper()})...")
        model = HybridCNNViTModel(num_classes=15, model_size=model_size, img_size=img_size).to(device)

        # ── Load checkpoint: CHỈ lấy training_state_dict (trọng số train gốc) ──
        print(f"\n📥 Loading checkpoint: {checkpoint_path}")
        ckpt = load_checkpoint_safe(checkpoint_path, device)
        ckpt_epoch = ckpt.get('epoch', 0)
        ckpt_auroc = ckpt.get('best_auroc', 0.0)

        # Ưu tiên training_state_dict (trọng số train), fallback sang state_dict (EMA)
        if 'training_state_dict' in ckpt:
            state_dict = ckpt['training_state_dict']
            print("   ✅ Loaded training_state_dict (trọng số train gốc, giữ momentum dynamics)")
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print("   ⚠️ Fallback: Loaded state_dict (EMA weights)")
        else:
            state_dict = ckpt
            print("   ⚠️ Fallback: Loaded raw checkpoint")

        # Clean state dict keys (remove 'module.' prefix if saved from DataParallel)
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            cleaned[name] = v
        model.load_state_dict(cleaned, strict=False)
        print(f"   ✅ Model weights loaded (from epoch {ckpt_epoch}, AUROC {ckpt_auroc:.4f})")

        # Multi-GPU fallback (DataParallel only, no DDP for simplicity)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"   ⚡ DataParallel: {torch.cuda.device_count()} GPUs")

        # ── Transforms: LUÔN bật Color Augmentation (fix bug gốc) ──
        class ImageOnlyTransform(torch.nn.Module):
            """Áp dụng transform chỉ lên 3 kênh ảnh đầu tiên."""
            def __init__(self, transform):
                super().__init__()
                self.transform = transform
            def forward(self, combined):
                img = combined[:3]
                mask = combined[3:]
                img = self.transform(img)
                return torch.cat([img, mask], dim=0)

        transformTrain = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # Color Augmentation LUÔN BẬT (đây là fix cho bug trong code gốc)
            ImageOnlyTransform(transforms.ColorJitter(brightness=0.15, contrast=0.15)),
            ImageOnlyTransform(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))),
        ])

        transformVal = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size)
        ])

        # ── Load Datasets ──
        print("\n📂 Loading datasets...")
        datasetTrain = DatasetGenerator(
            pathDirData, pathFileTrain, transformTrain,
            preload_images=preload_images, num_workers_preload=num_workers_preload
        )
        datasetVal = DatasetGenerator(
            pathDirData, pathFileVal, transformVal,
            preload_images=preload_images, num_workers_preload=num_workers_preload
        )

        # Stage 2 ngay lập tức: dùng toàn bộ dữ liệu với 1:3 VinDr oversampling
        hybrid_sampler = HybridBatchSampler(datasetTrain.sources, trBatchSize)
        dataLoaderTrain = FastDataLoader.create_dataloader(
            datasetTrain, batch_size=trBatchSize, num_workers=num_workers_train,
            batch_sampler=hybrid_sampler
        )
        dataLoaderVal = FastDataLoader.create_dataloader(
            datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=num_workers_train
        )

        # ── Optimizer MỚI (không load optimizer cũ) ──
        # Discriminative LR: backbone thấp, head cao hơn
        raw_model = _unwrap_model(model)
        head_params_ids = set(
            id(p) for p in list(raw_model.attention_head.parameters())
            + list(raw_model.fpn_lateral.parameters())
            + list(raw_model.fpn_merge.parameters())
        )
        backbone_params = [p for p in model.parameters() if id(p) not in head_params_ids]
        head_params = [p for p in model.parameters() if id(p) in head_params_ids]

        # FINE-TUNE LR: giảm mạnh so với gốc
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 3e-5},   # gốc: 1e-4
            {'params': head_params, 'lr': 1e-4},        # gốc: 5e-4
        ], weight_decay=0.08)  # gốc: 0.05
        print(f"🔧 Optimizer: AdamW (backbone_lr=3e-5, head_lr=1e-4, weight_decay=0.08)")

        # ── Loss Functions ──
        criterion_bce = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0, clip=0.05)  # gốc: gamma_neg=4
        criterion_val = nn.BCEWithLogitsLoss()
        criterion_dice = DiceLoss()
        criterion_sparsity = AttentionSparsityLoss()

        # ── Uncertainty Weighting ──
        uncertainty_weights = UncertaintyWeighting(n_tasks=2).to(device)
        optimizer.add_param_group({'params': uncertainty_weights.parameters(), 'lr': 1e-3})

        # ── Scheduler: CosineAnnealing ngắn ──
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, trMaxEpoch), eta_min=1e-7
        )

        # ── EMA: decay cao hơn cho ổn định ──
        target_model = _unwrap_model(model)
        ema = ModelEMA(target_model, decay=0.9995)  # gốc: 0.999

        print(f"🎯 ASL: gamma_neg={criterion_bce.gamma_neg} (gốc: 4), gamma_pos={criterion_bce.gamma_pos}, clip={criterion_bce.clip}")
        print(f"📊 EMA: decay={ema.decay} (gốc: 0.999)")
        print(f"⏰ Scheduler: CosineAnnealingLR (T_max={trMaxEpoch})")

        # ── Training Loop ──
        bestAUROC = ckpt_auroc  # Khởi đầu = AUROC checkpoint nguồn
        epochs_no_improve = 0
        early_stop_patience = 5
        stage = 2  # Luôn ở Stage 2

        print(f"\n{'='*60}")
        print(f"🔬 BẮT ĐẦU FINE-TUNE từ epoch {ckpt_epoch}")
        print(f"   Best AUROC baseline: {bestAUROC:.4f}")
        print(f"   Early stopping patience: {early_stop_patience}")
        print(f"{'='*60}")

        for epoch in range(trMaxEpoch):
            global_epoch = ckpt_epoch + epoch + 1
            print(f"\n{'='*60}")
            print(f"Epoch [{global_epoch}] (fine-tune epoch {epoch+1}/{trMaxEpoch}) - STAGE 2 (Fine-tune)")
            print(f"{'='*60}")

            # ── Train ──
            no_finding_idx = FinetuneTrainer.CLASS_NAMES.index('No Finding')
            trainLoss, bceLoss, diceLoss, sparsityLoss, rawDice = FinetuneTrainer.epochTrain(
                model, dataLoaderTrain, optimizer, criterion_bce, criterion_dice,
                criterion_sparsity, device, stage, no_finding_idx=no_finding_idx,
                scaler=scaler, amp_dtype=amp_dtype, ema=ema,
                uncertainty_weights=uncertainty_weights
            )

            # ── Val (dùng EMA weights) ──
            target_model_ref = _unwrap_model(model)
            ema.apply_shadow(target_model_ref)
            valLoss, valAUROC, valAcc = FinetuneTrainer.epochVal(
                model, dataLoaderVal, criterion_val, device,
                amp_dtype=amp_dtype, use_tta=True
            )
            ema.restore(target_model_ref)

            # ── Log ──
            if uncertainty_weights is not None:
                log_vars = uncertainty_weights.log_vars.data.tolist()
                precisions = [float(torch.exp(-v).item()) for v in uncertainty_weights.log_vars.data]
                print(f"\n🔧 UncertaintyWeighting log_vars: ASL={log_vars[0]:.3f}, Dice={log_vars[1]:.3f}")
                print(f"   Precision (1/σ²):             ASL={precisions[0]:.3f}, Dice={precisions[1]:.3f}")

            print(f"\nTrain - Total Loss: {trainLoss:.4f} | ASL: {bceLoss:.4f} | Dice: {diceLoss:.6f} | Raw Dice: {rawDice:.4f} | Sparsity: {sparsityLoss:.4f}")
            print(f"Val   - Total Loss: {valLoss:.4f}   | AUROC: {valAUROC:.4f} | Acc: {valAcc:.4f}")

            # In LR hiện tại
            for i, pg in enumerate(optimizer.param_groups):
                label = ['backbone', 'head', 'uw'][i] if i < 3 else f'group_{i}'
                print(f"   LR {label}: {pg['lr']:.2e}")

            scheduler.step()

            # ── Save & Early Stopping ──
            if valAUROC > bestAUROC:
                bestAUROC = valAUROC
                epochs_no_improve = 0

                os.makedirs(os.path.dirname(pathModel) or '.', exist_ok=True)

                # Lấy EMA state cho inference
                ema_model = _unwrap_model(model)
                training_state = {k: v.clone() for k, v in ema_model.state_dict().items()}
                ema_state = {k: v.clone() for k, v in training_state.items()}
                for name, ema_val in ema.shadow.items():
                    if name in ema_state:
                        ema_state[name] = ema_val.cpu()

                # TIẾT KIỆM ổ cứng: CHỈ lưu weights + metadata, BỎ optimizer/scheduler
                torch.save({
                    'epoch': global_epoch,
                    'state_dict': ema_state,
                    'training_state_dict': training_state,
                    'best_auroc': bestAUROC,
                    'stage': stage,
                    'model_size': model_size,
                    'img_size': img_size,
                    'finetuned': True,
                    'finetune_config': {
                        'lr_backbone': 3e-5,
                        'lr_head': 1e-4,
                        'weight_decay': 0.08,
                        'gamma_neg': 2,
                        'ema_decay': 0.9995,
                        'source_checkpoint_epoch': ckpt_epoch,
                        'source_checkpoint_auroc': ckpt_auroc,
                    }
                }, pathModel)
                # In dung lượng file
                file_size_mb = os.path.getsize(pathModel) / (1024 * 1024)
                print(f"✅ Fine-tuned model saved (epoch {global_epoch}, AUROC: {bestAUROC:.4f}, size: {file_size_mb:.0f}MB)")
            else:
                epochs_no_improve += 1
                print(f"⏳ No improvement ({epochs_no_improve}/{early_stop_patience})")
                if epochs_no_improve >= early_stop_patience:
                    print(f"\n🛑 Early stopping triggered tại epoch {global_epoch} "
                          f"(patience={early_stop_patience})")
                    break

        print(f"\n{'='*60}")
        print(f"✅ FINE-TUNE HOÀN TẤT")
        print(f"   Best AUROC: {bestAUROC:.4f} (baseline: {ckpt_auroc:.4f})")
        delta = bestAUROC - ckpt_auroc
        if delta > 0:
            print(f"   📈 Cải thiện: +{delta:.4f}")
        elif delta == 0:
            print(f"   ➡️  Không thay đổi (giữ nguyên checkpoint gốc)")
        else:
            print(f"   📉 Không cải thiện so với checkpoint gốc")
        print(f"   Model saved: {pathModel}")
        print(f"{'='*60}")

    @staticmethod
    def epochTrain(model, dataLoader, optimizer, criterion_bce, criterion_dice, criterion_sparsity,
                   device, stage, no_finding_idx=0, scaler=None, amp_dtype=torch.float32,
                   ema=None, uncertainty_weights=None):
        model.train()
        total_loss, total_bce, total_dice, total_sparsity = 0.0, 0.0, 0.0, 0.0
        total_raw_dice = 0.0
        n_dice_batches = 0
        attn_mean_sum, attn_std_sum = 0.0, 0.0

        pbar = tqdm(dataLoader, desc='Fine-tuning', leave=True)
        for inputs, targets, masks, source_flags in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            source_flags = source_flags.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits, attention_maps = model(inputs)

            logits = logits.float()
            attention_maps = attention_maps.float()

            # Attention stats
            with torch.no_grad():
                attn_mean_sum += attention_maps.mean().item()
                attn_std_sum += attention_maps.std().item()

            # VinDr mask
            vin_mask = (source_flags == 1)

            # ASL Loss
            bce_loss = criterion_bce(logits, targets.float())
            total_bce += bce_loss.item()

            # Sparsity Loss
            sparsity_loss = criterion_sparsity(attention_maps)
            total_sparsity += sparsity_loss.item()

            # Dice Loss (chỉ trên VinDr có bbox)
            dice_loss_val = torch.tensor(0.0, device=device)
            has_valid_dice = False
            if vin_mask.sum() > 0:
                vin_attention = attention_maps[vin_mask]
                vin_gt_masks = masks[vin_mask].float()
                valid_mask = (vin_gt_masks.sum(dim=(2, 3)) > 0)
                valid_mask[:, no_finding_idx] = False
                has_valid_dice = valid_mask.sum() > 0
                dice_loss_val = criterion_dice(vin_attention, vin_gt_masks, valid_mask)
                if has_valid_dice:
                    total_dice += dice_loss_val.item()
                    total_raw_dice += criterion_dice.last_raw_dice
                    n_dice_batches += 1

            # Uncertainty Weighting
            sparsity_weight = 0.05  # Stage 2 cố định
            if uncertainty_weights is not None:
                dice_active = has_valid_dice
                loss, _ = uncertainty_weights(bce_loss, dice_loss_val,
                                              active_mask=[True, dice_active])
                loss = loss + sparsity_weight * sparsity_loss
            else:
                loss = bce_loss + dice_loss_val * 1.2 + sparsity_loss * sparsity_weight

            # Consistency loss
            probs_all = torch.sigmoid(logits)
            p_no_finding = probs_all[:, no_finding_idx]
            p_diseases = torch.cat([probs_all[:, :no_finding_idx], probs_all[:, no_finding_idx+1:]], dim=1)
            p_max_disease = p_diseases.max(dim=1)[0]
            consistency_loss = (p_no_finding * p_max_disease).mean()
            loss = loss + consistency_loss * 0.1

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
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

            # EMA update
            if ema is not None:
                ema_target = _unwrap_model(model)
                ema.update(ema_target)

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Spars': f'{sparsity_loss.item():.3f}'})

        n_batches = len(dataLoader)
        avg_attn_mean = attn_mean_sum / n_batches
        avg_attn_std = attn_std_sum / n_batches
        print(f"📐 Attention stats - mean: {avg_attn_mean:.3f} | std: {avg_attn_std:.3f}")

        avg_dice = total_dice / n_dice_batches if n_dice_batches > 0 else 0.0
        avg_raw_dice = total_raw_dice / n_dice_batches if n_dice_batches > 0 else 0.0
        return total_loss / n_batches, total_bce / n_batches, avg_dice, total_sparsity / n_batches, avg_raw_dice

    @staticmethod
    def epochVal(model, dataLoader, criterion_val, device, amp_dtype=torch.float32, use_tta=True):
        model.eval()
        lossVal, n = 0.0, 0
        allPreds, allTargets = [], []

        with torch.no_grad():
            pbar = tqdm(dataLoader, desc='Validation+TTA', leave=True)
            for inputs, targets, masks, source_flags in pbar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                    logits, _ = model(inputs)
                logits = logits.float()
                probs = torch.sigmoid(logits)

                if use_tta:
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                        logits_flip, _ = model(torch.flip(inputs, dims=[3]))
                    logits_flip = logits_flip.float()
                    probs = (probs + torch.sigmoid(logits_flip)) / 2.0

                loss = criterion_val(logits, targets.float())
                lossVal += loss.item() * inputs.size(0)
                n += inputs.size(0)

                allPreds.append(probs.cpu().numpy())
                allTargets.append(targets.cpu().numpy())

        allPreds = np.concatenate(allPreds, axis=0)
        allTargets = np.concatenate(allTargets, axis=0)

        # AUROC
        aurocIndividual = []
        for i in range(allTargets.shape[1]):
            try:
                auc = roc_auc_score(allTargets[:, i], allPreds[:, i])
            except ValueError:
                auc = float('nan')
            aurocIndividual.append(auc)
        meanAUROC = np.nanmean(aurocIndividual)

        # Accuracy
        allBinary = (allPreds >= 0.5).astype(int)
        acc = accuracy_score(allTargets.flatten(), allBinary.flatten())

        return lossVal / n, meanAUROC, acc
