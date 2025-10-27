# TrainModel.py
import os
import time
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from Models.Model import ConvNeXtV2Model
    from Models.read_data import DatasetGenerator
except ImportError:
    from Model import ConvNeXtV2Model
    from read_data import DatasetGenerator


class ChexnetTrainer:
    CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia', 'No Finding'
    ]

    @staticmethod
    def train(pathDirData, pathFileTrain, pathFileVal,
              nnIsTrained, nnClassCount,
              trBatchSize, trMaxEpoch,
              transCrop, pathModel='CheXNet/Trainedmodel/chexnetmodel.pth',
              checkpoint=None, start_epoch=0):

        print("="*80)
        print("ChestX-ray14 Multi-Label Classification Training")
        print("Backbone: ConvNeXtV2-Large")
        print("="*80)

        # ---- Device Setup (Multi-GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")
        
        # ---- Tensor Core Optimization
        has_tensor_cores = False
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                compute_cap = torch.cuda.get_device_capability(i)
                print(f"  GPU {i}: {gpu_name}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                print(f"    Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
                
                # Tensor Cores available on: Volta (7.0+), Turing (7.5+), Ampere (8.0+), Ada/Hopper (8.9+)
                if compute_cap[0] >= 7:
                    has_tensor_cores = True
                    print(f"    ✓ Tensor Cores: ENABLED")
            
            if has_tensor_cores:
                print("\n🚀 Tensor Core Optimizations:")
                # Enable TF32 for Ampere+ GPUs (huge speedup)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("  ✓ TF32 enabled for matmul and cuDNN")
                
                # Enable cuDNN autotuner for best performance
                torch.backends.cudnn.benchmark = True
                print("  ✓ cuDNN benchmark autotuner enabled")
                
                # Use channels_last memory format for better Tensor Core utilization
                print("  ✓ Channels-last memory format will be used")
        else:
            print("⚠ No CUDA device found")

        # ---- Model
        model = ConvNeXtV2Model(
            num_classes=nnClassCount,
            pretrained=nnIsTrained,
            dropout_rate=0.2
        ).to(device)
        
        # Convert to channels_last for Tensor Core optimization
        if has_tensor_cores:
            model = model.to(memory_format=torch.channels_last)
            print("  ✓ Model converted to channels_last format")
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"\nUsing {torch.cuda.device_count()} GPUs (DataParallel)")
            model = torch.nn.DataParallel(model)

        # ---- Data Transforms (Larger size for better performance)
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        
        transformTrain = transforms.Compose([
            transforms.RandomResizedCrop(transCrop, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
        
        transformVal = transforms.Compose([
            transforms.Resize(int(transCrop * 1.14)),  # 384 -> 438
            transforms.CenterCrop(transCrop),
            transforms.ToTensor(),
            normalize
        ])

        # ---- Datasets & Loaders
        print("\nLoading datasets...")
        datasetTrain = DatasetGenerator(pathDirData, pathFileTrain, transformTrain)
        datasetVal   = DatasetGenerator(pathDirData, pathFileVal, transformVal)

        num_workers = min(8, os.cpu_count() or 4)
        dataLoaderTrain = DataLoader(
            datasetTrain, 
            batch_size=trBatchSize,
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        dataLoaderVal = DataLoader(
            datasetVal, 
            batch_size=trBatchSize,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        print(f"Train samples: {len(datasetTrain)}")
        print(f"Val samples: {len(datasetVal)}")
        print(f"Batch size: {trBatchSize}")
        print(f"Steps per epoch: {len(dataLoaderTrain)}")

        # ---- Optimizer & Scheduler (Advanced)
        # Use fused optimizer for Tensor Core GPUs (faster)
        if has_tensor_cores and hasattr(torch.optim, 'AdamW'):
            try:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=1e-4,
                    weight_decay=0.05,
                    betas=(0.9, 0.999),
                    fused=True  # Fused AdamW for Ampere+ GPUs
                )
                print("  ✓ Using fused AdamW optimizer")
            except:
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=1e-4,
                    weight_decay=0.05,
                    betas=(0.9, 0.999)
                )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.05,
                betas=(0.9, 0.999)
            )
        
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # ---- Loss (BCE with logits for numerical stability)
        criterion = nn.BCEWithLogitsLoss()

        # ---- Mixed Precision Training
        # Use bfloat16 on Ampere+ for better Tensor Core utilization, else float16
        use_bf16 = has_tensor_cores and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        if use_bf16:
            scaler = torch.amp.GradScaler('cuda', enabled=False)  # No scaling needed for bfloat16
            amp_dtype = torch.bfloat16
            print("  ✓ Using bfloat16 (Ampere+ optimal)")
        else:
            scaler = torch.amp.GradScaler('cuda')
            amp_dtype = torch.float16
            print("  ✓ Using float16 with gradient scaling")
        
        # Store for epoch functions
        ChexnetTrainer._amp_dtype = amp_dtype
        ChexnetTrainer._has_tensor_cores = has_tensor_cores

        # ---- Load checkpoint if exists
        bestLoss = float("inf")
        bestAUROC = 0.0
        
        if checkpoint and os.path.exists(checkpoint):
            print(f"\nLoading checkpoint: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location=device,weights_only=False)
            
            state_dict = ckpt['state_dict']
            target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            
            try:
                target_model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: {e}")
                # Handle DataParallel prefix
                if any(k.startswith('module.') for k in state_dict.keys()):
                    stripped = OrderedDict(
                        (k[7:], v) if k.startswith('module.') else (k, v)
                        for k, v in state_dict.items()
                    )
                    target_model.load_state_dict(stripped)
            
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            
            bestLoss = ckpt.get('best_loss', bestLoss)
            bestAUROC = ckpt.get('best_auroc', bestAUROC)
            start_epoch = ckpt.get('epoch', start_epoch)
            
            print(f"Resumed from epoch {start_epoch}, best_loss={bestLoss:.4f}, best_auroc={bestAUROC:.4f}")

        # ---- Training Loop
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        for epoch in range(start_epoch, trMaxEpoch):
            print(f"\nEpoch [{epoch+1}/{trMaxEpoch}]")
            print("-" * 60)
            
            # Train
            trainLoss, trainAUROC, trainAcc = ChexnetTrainer.epochTrain(
                model, dataLoaderTrain, optimizer, criterion, device, scaler
            )
            
            # Validate
            valLoss, valAUROC, valAcc = ChexnetTrainer.epochVal(
                model, dataLoaderVal, criterion, device
            )
            
            # Update scheduler
            scheduler.step()
            
            # Current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"\n{'Metric':<20} {'Train':<15} {'Val':<15}")
            print("-" * 50)
            print(f"{'Loss':<20} {trainLoss:<15.4f} {valLoss:<15.4f}")
            print(f"{'AUROC':<20} {trainAUROC:<15.4f} {valAUROC:<15.4f}")
            print(f"{'Accuracy':<20} {trainAcc:<15.4f} {valAcc:<15.4f}")
            print(f"{'Learning Rate':<20} {current_lr:<15.6f}")
            
            # VRAM usage
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"GPU {i} VRAM: {allocated:.2f}GB / {reserved:.2f}GB")
            
            # Save best model
            is_best = valAUROC > bestAUROC
            if is_best:
                bestAUROC = valAUROC
                bestLoss = valLoss
                
                os.makedirs(os.path.dirname(pathModel) or '.', exist_ok=True)
                
                state_dict_to_save = (
                    model.module.state_dict() 
                    if isinstance(model, torch.nn.DataParallel) 
                    else model.state_dict()
                )
                
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': state_dict_to_save,
                    'best_loss': bestLoss,
                    'best_auroc': bestAUROC,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, pathModel)
                
                print(f"\n✓ Model saved: {pathModel} (AUROC: {bestAUROC:.4f})")
            else:
                print(f"\n  Best AUROC: {bestAUROC:.4f}")

    @staticmethod
    def epochTrain(model, dataLoader, optimizer, criterion, device, scaler):
        model.train()
        
        totalLoss = 0.0
        allPreds = []
        allTargets = []
        
        # Get amp settings
        amp_dtype = getattr(ChexnetTrainer, '_amp_dtype', torch.float16)
        has_tensor_cores = getattr(ChexnetTrainer, '_has_tensor_cores', False)
        
        pbar = tqdm(dataLoader, desc="Training", ncols=100)
        
        for batch_idx, (input, target) in enumerate(pbar):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Convert to channels_last for Tensor Core optimization
            if has_tensor_cores:
                input = input.to(memory_format=torch.channels_last)
            
            # Mixed precision training with appropriate dtype
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                output = model(input)
                loss = criterion(output, target)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if amp_dtype == torch.bfloat16:
                # No scaling needed for bfloat16
                loss.backward()
                optimizer.step()
            else:
                # Use gradient scaling for float16
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            # Metrics
            totalLoss += loss.item()
            allPreds.append(torch.sigmoid(output).detach().cpu())
            allTargets.append(target.detach().cpu())
            
            # Progress
            avgLoss = totalLoss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avgLoss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Compute metrics
        allPreds = torch.cat(allPreds, dim=0).numpy()
        allTargets = torch.cat(allTargets, dim=0).numpy()
        
        auroc = ChexnetTrainer.computeAUROC_mean(allTargets, allPreds)
        acc = ChexnetTrainer.computeAccuracy(allTargets, allPreds)
        
        return totalLoss / len(dataLoader), auroc, acc

    @staticmethod
    def epochVal(model, dataLoader, criterion, device):
        model.eval()
        
        totalLoss = 0.0
        allPreds = []
        allTargets = []
        
        # Get amp settings
        amp_dtype = getattr(ChexnetTrainer, '_amp_dtype', torch.float16)
        has_tensor_cores = getattr(ChexnetTrainer, '_has_tensor_cores', False)
        
        pbar = tqdm(dataLoader, desc="Validation", ncols=100)
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(pbar):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                # Convert to channels_last for Tensor Core optimization
                if has_tensor_cores:
                    input = input.to(memory_format=torch.channels_last)
                
                # Use same dtype as training
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    output = model(input)
                    loss = criterion(output, target)
                
                totalLoss += loss.item()
                allPreds.append(torch.sigmoid(output).cpu())
                allTargets.append(target.cpu())
                
                avgLoss = totalLoss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avgLoss:.4f}'})
        
        # Compute metrics
        allPreds = torch.cat(allPreds, dim=0).numpy()
        allTargets = torch.cat(allTargets, dim=0).numpy()
        
        auroc = ChexnetTrainer.computeAUROC_mean(allTargets, allPreds)
        acc = ChexnetTrainer.computeAccuracy(allTargets, allPreds)
        
        return totalLoss / len(dataLoader), auroc, acc

    @staticmethod
    def computeAUROC_mean(dataGT, dataPRED):
        """Compute mean AUROC across all classes"""
        classCount = dataGT.shape[1]
        aurocIndividual = []
        
        for i in range(classCount):
            try:
                auroc = roc_auc_score(dataGT[:, i], dataPRED[:, i])
                aurocIndividual.append(auroc)
            except:
                aurocIndividual.append(np.nan)
        
        return np.nanmean(aurocIndividual)

    @staticmethod
    def computeAccuracy(dataGT, dataPRED, threshold=0.5):
        """Compute multi-label accuracy"""
        predBinary = (dataPRED >= threshold).astype(int)
        acc = accuracy_score(dataGT.flatten(), predBinary.flatten())
        return acc

    @staticmethod
    def test(pathDirData, pathFileTest, pathModel,
             nnClassCount, trBatchSize, transCrop,
             device=None):
        
        print("\n" + "="*80)
        print("Testing ChestX-ray14 Model")
        print("="*80)
        
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model = ConvNeXtV2Model(num_classes=nnClassCount, pretrained=False).to(device)
        
        print(f"\nLoading model: {pathModel}")
        ckpt = torch.load(pathModel, map_location=device,weights_only=False)
        state_dict = ckpt['state_dict']
        
        try:
            model.load_state_dict(state_dict)
        except Exception:
            if any(k.startswith('module.') for k in state_dict.keys()):
                stripped = OrderedDict(
                    (k[7:], v) if k.startswith('module.') else (k, v)
                    for k, v in state_dict.items()
                )
                model.load_state_dict(stripped)
        
        model.eval()
        
        # Data transforms
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        transformTest = transforms.Compose([
            transforms.Resize(int(transCrop * 1.14)),
            transforms.CenterCrop(transCrop),
            transforms.ToTensor(),
            normalize
        ])
        
        # Dataset
        datasetTest = DatasetGenerator(pathDirData, pathFileTest, transformTest)
        dataLoaderTest = DataLoader(
            datasetTest,
            batch_size=trBatchSize,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Test samples: {len(datasetTest)}")
        
        # Inference
        allPreds = []
        allTargets = []
        
        print("\nRunning inference...")
        with torch.no_grad():
            for input, target in tqdm(dataLoaderTest, desc="Testing"):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                output = model(input)
                pred = torch.sigmoid(output)
                
                allPreds.append(pred.cpu())
                allTargets.append(target.cpu())
        
        allPreds = torch.cat(allPreds, dim=0).numpy()
        allTargets = torch.cat(allTargets, dim=0).numpy()
        
        # Compute AUROC per class
        aurocIndividual = []
        for i in range(nnClassCount):
            try:
                auroc = roc_auc_score(allTargets[:, i], allPreds[:, i])
                aurocIndividual.append(auroc)
            except:
                aurocIndividual.append(np.nan)
        
        aurocMean = np.nanmean(aurocIndividual)
        
        # Print results
        print("\n" + "="*80)
        print("Test Results")
        print("="*80)
        print(f"\nMean AUROC: {aurocMean:.4f}\n")
        print(f"{'Disease':<25} {'AUROC':<10}")
        print("-" * 40)
        
        for i, name in enumerate(ChexnetTrainer.CLASS_NAMES[:nnClassCount]):
            auroc_val = aurocIndividual[i]
            if not np.isnan(auroc_val):
                print(f"{name:<25} {auroc_val:<10.4f}")
            else:
                print(f"{name:<25} {'N/A':<10}")
        
        print("="*80)
        
        return aurocMean, aurocIndividual, allPreds, allTargets
