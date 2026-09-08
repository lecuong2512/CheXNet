# TrainModel_finetune.py
# ============================================================================
# SOTA Fine-tune Trainer v2 — 3-Phase Strategy
#
# Pha 1 (epoch 1-freeze_epochs): Freeze backbone, chỉ train head
#   - Linear Warmup 2 epoch
#   - Head LR: 2e-4, Label Smoothing 0.02
#
# Pha 2 (epoch freeze_epochs+1 → swa_start-1): Unfreeze toàn bộ
#   - Backbone LR: 5e-6, Head LR: 5e-5
#   - MixUp α=0.2, R-Drop consistency
#   - CosineAnnealingLR
#
# Pha 3 (epoch swa_start → end): SWA + Snapshot Ensemble
#   - SWA LR: 1e-5, lưu snapshot mỗi epoch
#   - Cuối cùng: trung bình weights từ n_snapshots checkpoints tốt nhất
#
# Fix so với v1:
#   ✅ Giữ ASL γ_neg=4 (KHÔNG giảm xuống 2)
#   ✅ Giữ EMA decay=0.999
#   ✅ Giữ Weight Decay=0.05
#   ✅ Giữ UW clamp [-4, 4]
#   ✅ Nạp EMA weights (state_dict) thay vì training_state_dict
#   ✅ Thêm Warmup, MixUp, Label Smoothing, R-Drop, SWA
# ============================================================================
import os
import time
import copy
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Optional, List, Dict
from torch.optim.swa_utils import AveragedModel, SWALR

from Model import HybridCNNViTModel
from read_data import DatasetGenerator, FastDataLoader, HybridBatchSampler, VINDR_SOURCE_NAME
from checkpoint_utils import load_checkpoint_safe, extract_state_dict


def _unwrap_model(model):
    """Trích xuất model gốc từ DataParallel/AveragedModel wrapper."""
    if isinstance(model, nn.DataParallel):
        return model.module
    if isinstance(model, AveragedModel):
        return model.module
    return model


# ============================================================================
# Loss Functions — GIỮ NGUYÊN CONFIG GỐC
# ============================================================================

class AsymmetricLossOptimized(nn.Module):
    """ASL giữ nguyên gamma_neg=4 như training gốc.

    FIX: v1 giảm gamma_neg=2 khiến gradient từ True Negative dễ tràn trở lại
    → model bị kéo về đoán "Không bệnh" → Accuracy tăng nhưng AUROC giảm.
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-4,
                 disable_torch_grad_focal_loss=True, label_smoothing=0.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Label smoothing: 0 → ε/2, 1 → 1-ε/2
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2

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
    """EMA giữ nguyên decay=0.999 như training gốc."""
    def __init__(self, model, decay=0.999):
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

    def reinit_from_model(self, model):
        """Reinitialize EMA shadows from current model (for phase transitions)."""
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        for name, buf in model.named_buffers():
            self.shadow[name] = buf.data.clone()


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
    """UW giữ nguyên clamp [-4, 4] như training gốc."""
    def __init__(self, n_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *losses, active_mask=None):
        total = 0
        weighted_losses = []
        clamped_log_vars = torch.clamp(self.log_vars, min=-4.0, max=4.0)
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
# Freeze / Unfreeze Utilities
# ============================================================================

HEAD_KEYWORDS = ['attention_head', 'fpn_lateral', 'fpn_merge', 'classifier',
                 'raw_alpha', 'channel_proj']


def _is_head_param(name: str) -> bool:
    """Check if a parameter belongs to the head (not backbone)."""
    return any(k in name for k in HEAD_KEYWORDS)


def freeze_backbone(model):
    """Freeze tất cả backbone params, chỉ giữ head trainable."""
    raw = _unwrap_model(model)
    frozen_count = 0
    for name, p in raw.named_parameters():
        if not _is_head_param(name):
            p.requires_grad = False
            frozen_count += 1
    trainable = sum(1 for p in raw.parameters() if p.requires_grad)
    print(f"   ❄️  Frozen: {frozen_count} params | Trainable: {trainable} params (head only)")


def unfreeze_all(model):
    """Unfreeze toàn bộ model."""
    raw = _unwrap_model(model)
    for p in raw.parameters():
        p.requires_grad = True
    total = sum(1 for p in raw.parameters())
    print(f"   🔓 Unfrozen ALL: {total} params trainable")


# ============================================================================
# MixUp Utility
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation: trộn ảnh và nhãn."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Đảm bảo lam >= 0.5 (ảnh gốc chiếm đa số)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# ============================================================================
# SOTAFinetuneTrainer — 3-Phase Strategy
# ============================================================================

class SOTAFinetuneTrainer:
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
              num_workers_train: int = 4,
              # === SOTA Fine-tune params ===
              freeze_epochs: int = 3,
              swa_start: int = 9,
              mixup_alpha: float = 0.2,
              rdrop_alpha: float = 0.5,
              label_smoothing: float = 0.02,
              n_snapshots: int = 3):

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

        # ── Load checkpoint: ƯU TIÊN state_dict (EMA weights) ──
        # FIX: v1 dùng training_state_dict → điểm khởi đầu dao động hơn
        print(f"\n📥 Loading checkpoint: {checkpoint_path}")
        ckpt = load_checkpoint_safe(checkpoint_path, device)
        ckpt_epoch = ckpt.get('epoch', 0)
        ckpt_auroc = ckpt.get('best_auroc', 0.0)

        # FIX: Ưu tiên EMA weights (state_dict) cho fine-tune
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print("   ✅ Loaded state_dict (EMA weights — ổn định, tốt cho fine-tune)")
        elif 'training_state_dict' in ckpt:
            state_dict = ckpt['training_state_dict']
            print("   ⚠️ Fallback: Loaded training_state_dict")
        else:
            state_dict = ckpt
            print("   ⚠️ Fallback: Loaded raw checkpoint")

        # Clean keys
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            cleaned[name] = v
        model.load_state_dict(cleaned, strict=False)
        print(f"   ✅ Model weights loaded (from epoch {ckpt_epoch}, AUROC {ckpt_auroc:.4f})")
        del ckpt  # Free memory

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"   ⚡ DataParallel: {torch.cuda.device_count()} GPUs")

        # ── Transforms ──
        class ImageOnlyTransform(torch.nn.Module):
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

        hybrid_sampler = HybridBatchSampler(datasetTrain.sources, trBatchSize)
        dataLoaderTrain = FastDataLoader.create_dataloader(
            datasetTrain, batch_size=trBatchSize, num_workers=num_workers_train,
            batch_sampler=hybrid_sampler
        )
        dataLoaderVal = FastDataLoader.create_dataloader(
            datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=num_workers_train
        )

        # ── Optimizer (Phase 1: Head-only) ──
        raw_model = _unwrap_model(model)
        head_params = [p for n, p in raw_model.named_parameters() if _is_head_param(n)]
        backbone_params = [p for n, p in raw_model.named_parameters() if not _is_head_param(n)]

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 0.0},  # Frozen — LR=0
            {'params': head_params, 'lr': 2e-4},     # Head LR
        ], weight_decay=0.05)  # FIX: giữ nguyên gốc

        # ── Loss Functions ──
        criterion_bce = AsymmetricLossOptimized(
            gamma_neg=4, gamma_pos=0, clip=0.05,  # FIX: giữ gamma_neg=4
            label_smoothing=label_smoothing
        )
        criterion_val = nn.BCEWithLogitsLoss()
        criterion_dice = DiceLoss()
        criterion_sparsity = AttentionSparsityLoss()

        # ── Uncertainty Weighting ──
        uncertainty_weights = UncertaintyWeighting(n_tasks=2).to(device)
        optimizer.add_param_group({'params': uncertainty_weights.parameters(), 'lr': 1e-3})

        # ── Scheduler: Warmup → Cosine ──
        warmup_epochs = min(2, freeze_epochs)
        cosine_epochs = max(1, swa_start - warmup_epochs)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-6
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        # ── EMA ──
        target_model = _unwrap_model(model)
        ema = ModelEMA(target_model, decay=0.999)  # FIX: giữ 0.999

        # ── SWA (khởi tạo cho Phase 3) ──
        swa_model = AveragedModel(raw_model)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=2)

        # ── Print Config ──
        print(f"\n{'='*70}")
        print(f"🔬 SOTA FINE-TUNE CONFIG — 3-Phase Strategy")
        print(f"{'='*70}")
        print(f"  ASL gamma_neg: 4 (giữ nguyên gốc ✅)")
        print(f"  Weight Decay: 0.05 (giữ nguyên gốc ✅)")
        print(f"  EMA decay: 0.999 (giữ nguyên gốc ✅)")
        print(f"  UW clamp: [-4, 4] (giữ nguyên gốc ✅)")
        print(f"  Label Smoothing: {label_smoothing}")
        print(f"  MixUp alpha: {mixup_alpha} (Pha 2+)")
        print(f"  R-Drop alpha: {rdrop_alpha} (Pha 2+)")
        print(f"  ── Pha 1 (epoch 1-{freeze_epochs}): Freeze backbone, Head LR=2e-4 ──")
        print(f"  ── Pha 2 (epoch {freeze_epochs+1}-{swa_start-1}): Unfreeze, Backbone=5e-6, Head=5e-5 ──")
        print(f"  ── Pha 3 (epoch {swa_start}-{trMaxEpoch}): SWA LR=1e-5, {n_snapshots} snapshots ──")
        print(f"{'='*70}")

        # ── Freeze backbone for Phase 1 ──
        print("\n🧊 Pha 1: Freeze backbone...")
        freeze_backbone(model)

        # ── Training Loop ──
        bestAUROC = ckpt_auroc
        epochs_no_improve = 0
        early_stop_patience = 10
        current_phase = 1
        snapshots = []
        snapshot_aurocs = []

        print(f"\n{'='*60}")
        print(f"🚀 BẮT ĐẦU FINE-TUNE SOTA từ epoch {ckpt_epoch}")
        print(f"   Best AUROC baseline: {bestAUROC:.4f}")
        print(f"{'='*60}")

        for epoch in range(trMaxEpoch):
            global_epoch = ckpt_epoch + epoch + 1
            ft_epoch = epoch + 1

            # ── Phase Transitions ──
            if ft_epoch == freeze_epochs + 1 and current_phase == 1:
                current_phase = 2
                print(f"\n{'='*60}")
                print(f"🔓 CHUYỂN SANG PHA 2: Unfreeze + MixUp + R-Drop")
                print(f"{'='*60}")
                unfreeze_all(model)
                optimizer.param_groups[0]['lr'] = 5e-6   # Backbone
                optimizer.param_groups[1]['lr'] = 5e-5   # Head
                ema.reinit_from_model(_unwrap_model(model))
                print(f"   📊 EMA re-initialized from current weights")

            elif ft_epoch == swa_start and current_phase == 2:
                current_phase = 3
                print(f"\n{'='*60}")
                print(f"📊 CHUYỂN SANG PHA 3: SWA + Snapshot Ensemble")
                print(f"{'='*60}")
                print(f"   SWA LR: 1e-5 | Snapshots: {n_snapshots}")

            phase_label = {1: "HEAD-ONLY", 2: "FULL+MixUp+R-Drop", 3: "SWA"}[current_phase]
            print(f"\n{'='*60}")
            print(f"Epoch [{global_epoch}] (ft {ft_epoch}/{trMaxEpoch}) — PHA {current_phase}: {phase_label}")
            print(f"{'='*60}")

            use_mixup = (current_phase >= 2 and mixup_alpha > 0)
            use_rdrop = (current_phase >= 2 and rdrop_alpha > 0)

            # ── Train ──
            no_finding_idx = SOTAFinetuneTrainer.CLASS_NAMES.index('No Finding')
            trainLoss, bceLoss, diceLoss, sparsityLoss, rawDice = SOTAFinetuneTrainer.epochTrain(
                model, dataLoaderTrain, optimizer, criterion_bce, criterion_dice,
                criterion_sparsity, device, no_finding_idx=no_finding_idx,
                scaler=scaler, amp_dtype=amp_dtype, ema=ema,
                uncertainty_weights=uncertainty_weights,
                use_mixup=use_mixup, mixup_alpha=mixup_alpha,
                use_rdrop=use_rdrop, rdrop_alpha=rdrop_alpha,
            )

            # ── Val (EMA weights) ──
            target_model_ref = _unwrap_model(model)
            ema.apply_shadow(target_model_ref)
            valLoss, valAUROC, valAcc = SOTAFinetuneTrainer.epochVal(
                model, dataLoaderVal, criterion_val, device,
                amp_dtype=amp_dtype, use_tta=True
            )
            ema.restore(target_model_ref)

            # ── SWA update (Phase 3) ──
            if current_phase == 3:
                swa_model.update_parameters(_unwrap_model(model))
                swa_scheduler.step()
                snapshot_state = {k: v.cpu().clone() for k, v in _unwrap_model(model).state_dict().items()}
                snapshots.append(snapshot_state)
                snapshot_aurocs.append(valAUROC)
                print(f"   📸 Snapshot #{len(snapshots)} saved (AUROC: {valAUROC:.4f})")
            else:
                scheduler.step()

            # ── Log ──
            if uncertainty_weights is not None:
                log_vars = uncertainty_weights.log_vars.data.tolist()
                precisions = [float(torch.exp(-v).item()) for v in uncertainty_weights.log_vars.data]
                print(f"\n🔧 UW log_vars: ASL={log_vars[0]:.3f}, Dice={log_vars[1]:.3f}")
                print(f"   Precision: ASL={precisions[0]:.3f}, Dice={precisions[1]:.3f}")

            print(f"\nTrain - Loss: {trainLoss:.4f} | ASL: {bceLoss:.4f} | Dice: {diceLoss:.6f} | "
                  f"Raw Dice: {rawDice:.4f} | Sparsity: {sparsityLoss:.4f}")
            print(f"Val   - Loss: {valLoss:.4f}   | AUROC: {valAUROC:.4f} | Acc: {valAcc:.4f}")
            if use_mixup:
                print(f"   🔀 MixUp: alpha={mixup_alpha}")
            if use_rdrop:
                print(f"   🔄 R-Drop: alpha={rdrop_alpha}")

            for i, pg in enumerate(optimizer.param_groups):
                label = ['backbone', 'head', 'uw'][i] if i < 3 else f'group_{i}'
                print(f"   LR {label}: {pg['lr']:.2e}")

            # ── Save & Early Stopping ──
            if valAUROC > bestAUROC:
                bestAUROC = valAUROC
                epochs_no_improve = 0

                os.makedirs(os.path.dirname(pathModel) or '.', exist_ok=True)

                ema_model = _unwrap_model(model)
                training_state = {k: v.clone() for k, v in ema_model.state_dict().items()}
                ema_state = {k: v.clone() for k, v in training_state.items()}
                for name, ema_val in ema.shadow.items():
                    if name in ema_state:
                        ema_state[name] = ema_val.cpu()

                torch.save({
                    'epoch': global_epoch,
                    'state_dict': ema_state,
                    'training_state_dict': training_state,
                    'best_auroc': bestAUROC,
                    'stage': 2,
                    'model_size': model_size,
                    'img_size': img_size,
                    'finetuned': True,
                    'finetune_version': 'SOTA_v2',
                    'phase': current_phase,
                    'finetune_config': {
                        'strategy': '3-phase SOTA',
                        'freeze_epochs': freeze_epochs,
                        'swa_start': swa_start,
                        'mixup_alpha': mixup_alpha,
                        'rdrop_alpha': rdrop_alpha,
                        'label_smoothing': label_smoothing,
                        'asl_gamma_neg': 4,
                        'ema_decay': 0.999,
                        'weight_decay': 0.05,
                        'source_epoch': ckpt_epoch,
                        'source_auroc': ckpt_auroc,
                    }
                }, pathModel)
                file_size_mb = os.path.getsize(pathModel) / (1024 * 1024)
                print(f"✅ Best model saved (epoch {global_epoch}, AUROC: {bestAUROC:.4f}, "
                      f"phase {current_phase}, {file_size_mb:.0f}MB)")
            else:
                epochs_no_improve += 1
                print(f"⏳ No improvement ({epochs_no_improve}/{early_stop_patience})")
                if epochs_no_improve >= early_stop_patience and current_phase >= 2:
                    print(f"\n🛑 Early stopping at epoch {global_epoch}")
                    break

        # ── Snapshot Ensemble ──
        if len(snapshots) >= 2:
            print(f"\n{'='*60}")
            print(f"📊 SNAPSHOT ENSEMBLE — Trung bình {min(n_snapshots, len(snapshots))} checkpoints")
            print(f"{'='*60}")

            sorted_indices = np.argsort(snapshot_aurocs)[::-1][:n_snapshots]
            selected_snapshots = [snapshots[i] for i in sorted_indices]
            selected_aurocs = [snapshot_aurocs[i] for i in sorted_indices]
            print(f"   Selected snapshots AUROC: {[f'{a:.4f}' for a in selected_aurocs]}")

            avg_state = {}
            for key in selected_snapshots[0]:
                tensors = [s[key].float() for s in selected_snapshots]
                avg_state[key] = torch.stack(tensors).mean(dim=0)

            raw_model_eval = _unwrap_model(model)
            raw_model_eval.load_state_dict(avg_state)

            ensembleLoss, ensembleAUROC, ensembleAcc = SOTAFinetuneTrainer.epochVal(
                model, dataLoaderVal, criterion_val, device,
                amp_dtype=amp_dtype, use_tta=True
            )
            print(f"   🏆 Snapshot Ensemble AUROC: {ensembleAUROC:.4f} | Acc: {ensembleAcc:.4f}")

            if ensembleAUROC > bestAUROC:
                bestAUROC = ensembleAUROC
                ensemble_path = pathModel.replace('.pth', '_ensemble.pth')
                torch.save({
                    'epoch': global_epoch,
                    'state_dict': {k: v.cpu() for k, v in avg_state.items()},
                    'best_auroc': ensembleAUROC,
                    'stage': 2,
                    'model_size': model_size,
                    'img_size': img_size,
                    'finetuned': True,
                    'finetune_version': 'SOTA_v2_ensemble',
                    'n_snapshots': len(selected_snapshots),
                }, ensemble_path)
                file_size_mb = os.path.getsize(ensemble_path) / (1024 * 1024)
                print(f"   ✅ Ensemble model saved: {ensemble_path} ({file_size_mb:.0f}MB)")

        # ── Summary ──
        print(f"\n{'='*60}")
        print(f"✅ SOTA FINE-TUNE HOÀN TẤT")
        print(f"   Best AUROC: {bestAUROC:.4f} (baseline: {ckpt_auroc:.4f})")
        delta = bestAUROC - ckpt_auroc
        if delta > 0:
            print(f"   📈 Cải thiện: +{delta:.4f}")
        elif delta == 0:
            print(f"   ➡️  Không thay đổi")
        else:
            print(f"   📉 Không cải thiện so với baseline")
        print(f"   Snapshots collected: {len(snapshots)}")
        print(f"   Model saved: {pathModel}")
        print(f"{'='*60}")

    @staticmethod
    def epochTrain(model, dataLoader, optimizer, criterion_bce, criterion_dice,
                   criterion_sparsity, device, no_finding_idx=0,
                   scaler=None, amp_dtype=torch.float32, ema=None,
                   uncertainty_weights=None,
                   use_mixup=False, mixup_alpha=0.2,
                   use_rdrop=False, rdrop_alpha=0.5):
        model.train()
        total_loss, total_bce, total_dice, total_sparsity = 0.0, 0.0, 0.0, 0.0
        total_raw_dice = 0.0
        n_dice_batches = 0
        attn_mean_sum, attn_std_sum = 0.0, 0.0
        use_amp_flag = (amp_dtype != torch.float32)

        pbar = tqdm(dataLoader, desc='SOTA Fine-tuning', leave=True)
        for inputs, targets, masks, source_flags in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            source_flags = source_flags.to(device, non_blocking=True)

            # ── MixUp ──
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp_flag):
                logits, attention_maps = model(inputs)

            logits = logits.float()
            attention_maps = attention_maps.float()

            with torch.no_grad():
                attn_mean_sum += attention_maps.mean().item()
                attn_std_sum += attention_maps.std().item()

            vin_mask = (source_flags == 1)

            # ── ASL Loss (with MixUp) ──
            if use_mixup:
                bce_loss = lam * criterion_bce(logits, targets_a.float()) + \
                           (1 - lam) * criterion_bce(logits, targets_b.float())
            else:
                bce_loss = criterion_bce(logits, targets.float())
            total_bce += bce_loss.item()

            # ── R-Drop Consistency ──
            rdrop_loss = torch.tensor(0.0, device=device)
            if use_rdrop:
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp_flag):
                    logits2, _ = model(inputs)
                logits2 = logits2.float()
                p1 = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
                p2 = torch.sigmoid(logits2).clamp(1e-6, 1 - 1e-6)
                rdrop_loss = (F.binary_cross_entropy(p1, p2.detach()) +
                              F.binary_cross_entropy(p2, p1.detach())) * 0.5

            # ── Sparsity Loss ──
            sparsity_loss = criterion_sparsity(attention_maps)
            total_sparsity += sparsity_loss.item()

            # ── Dice Loss (VinDr only) ──
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

            # ── Total Loss ──
            sparsity_weight = 0.05
            if uncertainty_weights is not None:
                dice_active = has_valid_dice
                loss, _ = uncertainty_weights(bce_loss, dice_loss_val,
                                              active_mask=[True, dice_active])
                loss = loss + sparsity_weight * sparsity_loss
            else:
                loss = bce_loss + dice_loss_val * 1.2 + sparsity_loss * sparsity_weight

            # R-Drop
            if use_rdrop:
                loss = loss + rdrop_alpha * rdrop_loss

            # Consistency loss
            probs_all = torch.sigmoid(logits)
            p_no_finding = probs_all[:, no_finding_idx]
            p_diseases = torch.cat([probs_all[:, :no_finding_idx], probs_all[:, no_finding_idx+1:]], dim=1)
            p_max_disease = p_diseases.max(dim=1)[0]
            consistency_loss = (p_no_finding * p_max_disease).mean()
            loss = loss + consistency_loss * 0.1

            # ── Backward ──
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

            # EMA
            if ema is not None:
                ema_target = _unwrap_model(model)
                ema.update(ema_target)

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Spars': f'{sparsity_loss.item():.3f}'})

        n_batches = max(len(dataLoader), 1)
        avg_attn_mean = attn_mean_sum / n_batches
        avg_attn_std = attn_std_sum / n_batches
        print(f"📐 Attention stats - mean: {avg_attn_mean:.3f} | std: {avg_attn_std:.3f}")
        if avg_attn_std < 0.05:
            print(f"⚠️  CẢNH BÁO: Attention collapse detected!")

        avg_dice = total_dice / n_dice_batches if n_dice_batches > 0 else 0.0
        avg_raw_dice = total_raw_dice / n_dice_batches if n_dice_batches > 0 else 0.0
        return total_loss / n_batches, total_bce / n_batches, avg_dice, \
               total_sparsity / n_batches, avg_raw_dice

    @staticmethod
    def epochVal(model, dataLoader, criterion_val, device, amp_dtype=torch.float32, use_tta=True):
        model.eval()
        lossVal, n = 0.0, 0
        allPreds, allTargets = [], []
        use_amp_flag = (amp_dtype != torch.float32)

        with torch.no_grad():
            pbar = tqdm(dataLoader, desc='Validation+TTA', leave=True)
            for inputs, targets, masks, source_flags in pbar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp_flag):
                    logits, _ = model(inputs)
                logits = logits.float()
                probs = torch.sigmoid(logits)

                if use_tta:
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp_flag):
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

        aurocIndividual = []
        for i in range(allTargets.shape[1]):
            try:
                auc = roc_auc_score(allTargets[:, i], allPreds[:, i])
            except ValueError:
                auc = float('nan')
            aurocIndividual.append(auc)
        meanAUROC = np.nanmean(aurocIndividual)

        allBinary = (allPreds >= 0.5).astype(int)
        acc = accuracy_score(allTargets.flatten(), allBinary.flatten())

        return lossVal / max(n, 1), meanAUROC, acc
