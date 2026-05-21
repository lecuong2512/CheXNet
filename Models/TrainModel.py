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

from Model import HybridCNNViTModel
from read_data import DatasetGenerator, FastDataLoader, HybridBatchSampler

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Nội suy mask gốc cho bằng kích thước attention map
        targets = F.interpolate(targets, size=preds.shape[2:], mode='bilinear', align_corners=False)
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

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
              pathModel: str = 'Trainedmodel/hybrid_model.pth'):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")
        
        # ---- Build Model
        print(f"\n🏗️ Building Hybrid Model ({model_size.upper()})...")
        model = HybridCNNViTModel(num_classes=15, model_size=model_size, img_size=img_size).to(device)
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # ---- Data Transforms (Geometric only for Training to keep Mask sync)
        transformTrain = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
        
        transformVal = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size)
        ])

        print("\n📂 Loading datasets...")
        datasetTrain = DatasetGenerator(pathDirData, pathFileTrain, transformTrain)
        datasetVal = DatasetGenerator(pathDirData, pathFileVal, transformVal)

        # Lọc Index cho Giai đoạn 1 (Chỉ lấy VinDr)
        vin_indices_train = [i for i, s in enumerate(datasetTrain.sources) if s == 'VinBigData']
        vin_subset_train = torch.utils.data.Subset(datasetTrain, vin_indices_train)
        
        num_workers = min(8, os.cpu_count() or 4)
        
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

        # CosineAnnealingLR: linh hoạt hơn OneCycleLR cho training 2 giai đoạn
        # vì không phụ thuộc vào total_steps cố định
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=trMaxEpoch,
            eta_min=1e-6
        )

        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_dice = DiceLoss()

        bestAUROC = 0.0
        stage = 1
        current_loader = loader_stage1

        # Early stopping
        early_stop_patience = 10
        epochs_no_improve = 0

        print("\n🚀 Bắt đầu Giai đoạn 1: Warm-up Attention với Bounding Box")

        for epoch in range(trMaxEpoch):
            print(f"\n{'='*60}\nEpoch [{epoch+1}/{trMaxEpoch}] - STAGE {stage}\n{'='*60}")

            # Train
            trainLoss, bceLoss, diceLoss = HybridTrainer.epochTrain(
                model, current_loader, optimizer, criterion_bce, criterion_dice, device, stage
            )

            # Val
            valLoss, valAUROC, valAcc = HybridTrainer.epochVal(model, dataLoaderVal, criterion_bce, device)

            print(f"\nTrain - Total Loss: {trainLoss:.4f} | BCE: {bceLoss:.4f} | Dice: {diceLoss:.4f}")
            print(f"Val   - Total Loss: {valLoss:.4f}   | AUROC: {valAUROC:.4f} | Acc: {valAcc:.4f}")

            scheduler.step()

            # Check Stage Transition
            if stage == 1 and valAUROC > 0.70 and diceLoss < 0.4:
                print("\n🔓 Đạt ngưỡng tối ưu! Chuyển sang Giai đoạn 2 (Toàn bộ dữ liệu + 1:3 Sampler)")
                stage = 2
                current_loader = loader_stage2
                # Reset scheduler cho giai đoạn mới
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=trMaxEpoch - epoch,
                    eta_min=1e-6
                )

            # Save Model & Early Stopping
            if valAUROC > bestAUROC:
                bestAUROC = valAUROC
                epochs_no_improve = 0
                os.makedirs(os.path.dirname(pathModel) or '.', exist_ok=True)

                state_dict_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': state_dict_to_save,
                    'best_auroc': bestAUROC,
                    'stage': stage
                }, pathModel)
                print(f"✅ Model saved (AUROC: {bestAUROC:.4f})")
            else:
                epochs_no_improve += 1
                print(f"⏳ No improvement ({epochs_no_improve}/{early_stop_patience})")
                if epochs_no_improve >= early_stop_patience:
                    print(f"\n🛑 Early stopping triggered sau {epoch+1} epochs (patience={early_stop_patience})")
                    break

    @staticmethod
    def epochTrain(model, dataLoader, optimizer, criterion_bce, criterion_dice, device, stage):
        model.train()
        total_loss, total_bce, total_dice = 0.0, 0.0, 0.0
        
        pbar = tqdm(dataLoader, desc="Training", ncols=100)
        for inputs, targets, masks, sources in pbar:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits, attention_maps = model(inputs)
            
            vin_mask = torch.tensor([s == 'VinBigData' for s in sources]).to(device)
            
            loss = 0
            # Tính BCE cho tất cả
            bce_loss = criterion_bce(logits, targets)
            loss += bce_loss
            total_bce += bce_loss.item()
            
            # Tính Dice Loss cho VinDr
            if vin_mask.sum() > 0:
                vin_attention = attention_maps[vin_mask]
                vin_gt_masks = masks[vin_mask]
                
                dice_loss = criterion_dice(vin_attention, vin_gt_masks)
                loss += dice_loss * (1.5 if stage == 1 else 0.5)
                total_dice += dice_loss.item()
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
        return total_loss / len(dataLoader), total_bce / len(dataLoader), total_dice / len(dataLoader)

    @staticmethod
    def epochVal(model, dataLoader, criterion_bce, device):
        model.eval()
        total_loss = 0.0
        allPreds, allTargets = [], []
        
        pbar = tqdm(dataLoader, desc="Validation", ncols=100)
        with torch.no_grad():
            for inputs, targets, _, _ in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                
                loss = criterion_bce(logits, targets)
                total_loss += loss.item()
                
                allPreds.append(torch.sigmoid(logits).cpu())
                allTargets.append(targets.cpu())
                
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
            except:
                aurocIndividual.append(np.nan)
        return np.nanmean(aurocIndividual)
