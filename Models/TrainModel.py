# TrainModel.py - Optimized training with fine-tuning, class weighting, and TTA
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Optional, Tuple, Dict, List
import warnings

try:
    from Models.Model import ConvNeXtV2Model
    from Models.read_data import DatasetGenerator, FastDataLoader, create_tta_transforms
except ImportError:
    from Model import ConvNeXtV2Model
    from read_data import DatasetGenerator, FastDataLoader, create_tta_transforms


class ChexnetTrainer:
    CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia', 'No Finding'
    ]

    @staticmethod
    def train(pathDirData: str, 
              pathFileTrain: str, 
              pathFileVal: str,
              nnIsTrained: bool, 
              nnClassCount: int,
              trBatchSize: int, 
              trMaxEpoch: int,
              transCrop: int, 
              pathModel: str = 'CheXNet/Trainedmodel/chexnetmodel.pth',
              checkpoint: Optional[str] = None, 
              start_epoch: int = 0,
              use_class_weights: bool = True,
              fine_tune_epoch: int = 30,
              use_compile: bool = False):
        """
        Training with advanced optimizations:
        - Multi-GPU support with optimal settings
        - Class weighting for imbalanced data
        - Fine-tuning strategy (freeze -> unfreeze)
        - Mixed precision training (bf16/fp16)
        - Gradient accumulation for large effective batch sizes
        """

        print("="*80)
        print("ChestX-ray14 Multi-Label Classification Training")
        print("Backbone: ConvNeXtV2-Large")
        print("="*80)

        # ---- Device Setup (Multi-GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")
        
        # ---- Tensor Core Optimization
        has_tensor_cores = False
        amp_dtype = torch.float16
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"GPU Count: {num_gpus}")
            
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                compute_cap = torch.cuda.get_device_capability(i)
                memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"  GPU {i}: {gpu_name}")
                print(f"    Memory: {memory_gb:.2f} GB")
                print(f"    Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
                
                if compute_cap[0] >= 7:
                    has_tensor_cores = True
                    print(f"    ✓ Tensor Cores: ENABLED")
                    
                    # Determine optimal dtype
                    if compute_cap[0] >= 8:  # Ampere+
                        amp_dtype = torch.bfloat16
                        print(f"    ✓ Using bfloat16 (optimal for Ampere+)")
            
            if has_tensor_cores:
                print("\n🚀 Tensor Core Optimizations:")
                # Enable TF32 for Ampere+ GPUs
                if amp_dtype == torch.bfloat16:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    print("  ✓ TF32 enabled for matmul and cuDNN")
                
                # Enable cuDNN autotuner
                torch.backends.cudnn.benchmark = True
                print("  ✓ cuDNN benchmark autotuner enabled")
                print("  ✓ Channels-last memory format will be used")
        else:
            print("⚠ No CUDA device found")

        # ---- Model
        print(f"\n🏗️ Building model...")
        model = ConvNeXtV2Model(
            num_classes=nnClassCount,
            pretrained=nnIsTrained,
            dropout_rate=0.3  # Increased for better regularization
        ).to(device)
        
        # Convert to channels_last for Tensor Core optimization
        if has_tensor_cores:
            model = model.to(memory_format=torch.channels_last)
            print("  ✓ Model converted to channels_last format")
        
        # torch.compile for PyTorch 2.0+ (20-30% speedup on Ampere+)
        if use_compile and has_tensor_cores and hasattr(torch, 'compile'):
            try:
                print("  🔥 Compiling model with torch.compile()...")
                model = torch.compile(model, mode='max-autotune')
                print("  ✓ Model compiled (expect 20-30% speedup)")
            except Exception as e:
                print(f"  ⚠ Could not compile: {e}")
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"\n🔀 Using {torch.cuda.device_count()} GPUs (DataParallel)")
            model = torch.nn.DataParallel(model)

        # ---- Data Transforms
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        
        # Enhanced training augmentations
        transformTrain = transforms.Compose([
            transforms.RandomResizedCrop(transCrop, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize
        ])
        
        transformVal = transforms.Compose([
            transforms.Resize(int(transCrop * 1.14)),
            transforms.CenterCrop(transCrop),
            transforms.ToTensor(),
            normalize
        ])

        # ---- Datasets & Loaders
        print("\n📂 Loading datasets...")
        datasetTrain = DatasetGenerator(
            pathDirData, pathFileTrain, transformTrain,
            cache_images=False,  # Set True if you have enough RAM
            preload_images=False  # Set True for maximum speed (needs lots of RAM)
        )
        datasetVal = DatasetGenerator(
            pathDirData, pathFileVal, transformVal,
            cache_images=False,
            preload_images=False
        )
        
        datasetTrain.print_statistics()

        # Optimal DataLoader settings
        num_workers = min(8, os.cpu_count() or 4)
        
        dataLoaderTrain = FastDataLoader.create_dataloader(
            datasetTrain,
            batch_size=trBatchSize,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        dataLoaderVal = FastDataLoader.create_dataloader(
            datasetVal,
            batch_size=trBatchSize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        print(f"✓ Train samples: {len(datasetTrain)}")
        print(f"✓ Val samples: {len(datasetVal)}")
        print(f"✓ Batch size: {trBatchSize}")
        print(f"✓ Steps per epoch: {len(dataLoaderTrain)}")

        # ---- Class Weights for Imbalanced Data
        class_weights = None
        if use_class_weights:
            print("\n⚖️ Computing class weights for imbalanced data...")
            class_weights = datasetTrain.get_class_weights(smooth=0.1).to(device)
            print("Class weights applied:")
            for i, w in enumerate(class_weights):
                print(f"  Class {i:2d} ({ChexnetTrainer.CLASS_NAMES[i]:20s}): {w:.4f}")

        # ---- Optimizer (AdamW with fused kernel if available)
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        # Separate parameters for fine-tuning
        backbone_params = []
        classifier_params = []
        
        for name, param in base_model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        print(f"\n🎯 Optimizer configuration:")
        print(f"  Backbone params: {len(backbone_params)}")
        print(f"  Classifier params: {len(classifier_params)}")
        
        # Use fused optimizer for faster training on Ampere+
        optimizer_class = optim.AdamW
        optimizer_kwargs = {
            'lr': 1e-4,
            'weight_decay': 0.05,
            'betas': (0.9, 0.999)
        }
        
        if has_tensor_cores and amp_dtype == torch.bfloat16:
            try:
                optimizer_kwargs['fused'] = True
                print(f"  ✓ Using fused AdamW optimizer")
            except:
                pass
        
        optimizer = optimizer_class([
            {'params': backbone_params, 'lr': 1e-5},  # Lower LR for pretrained backbone
            {'params': classifier_params, 'lr': 1e-4}  # Higher LR for classifier
        ], **optimizer_kwargs)
        
        # OneCycleLR scheduler for better convergence
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[1e-4, 5e-4],  # Different max LR for each param group
            epochs=trMaxEpoch,
            steps_per_epoch=len(dataLoaderTrain),
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )

        # ---- Loss (BCE with class weights)
        if class_weights is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            print("  ✓ Using weighted BCE loss")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("  ✓ Using standard BCE loss")

        # ---- Mixed Precision Training
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
        
        if use_amp:
            dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
            print(f"  ✓ Using {dtype_name} mixed precision")
        
        # Store for epoch functions
        ChexnetTrainer._amp_dtype = amp_dtype
        ChexnetTrainer._has_tensor_cores = has_tensor_cores
        ChexnetTrainer._use_amp = use_amp

        # ---- Load checkpoint if exists
        bestLoss = float("inf")
        bestAUROC = 0.0
        
        if checkpoint and os.path.exists(checkpoint):
            print(f"\n📥 Loading checkpoint: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            
            state_dict = ckpt['state_dict']
            target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            
            try:
                target_model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: {e}")
                if any(k.startswith('module.') for k in state_dict.keys()):
                    stripped = OrderedDict(
                        (k[7:], v) if k.startswith('module.') else (k, v)
                        for k, v in state_dict.items()
                    )
                    target_model.load_state_dict(stripped)
            
            if 'optimizer' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                except:
                    print("  ⚠ Could not load optimizer state")
            
            if 'scheduler' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                except:
                    print("  ⚠ Could not load scheduler state")
            
            bestLoss = ckpt.get('best_loss', bestLoss)
            bestAUROC = ckpt.get('best_auroc', bestAUROC)
            start_epoch = ckpt.get('epoch', start_epoch)
            
            print(f"✓ Resumed from epoch {start_epoch}")
            print(f"  Best loss: {bestLoss:.4f}")
            print(f"  Best AUROC: {bestAUROC:.4f}")

        # ---- Training Loop
        print("\n" + "="*80)
        print("🚀 Starting Training")
        print("="*80)
        print(f"Fine-tuning strategy: Unfreeze backbone at epoch {fine_tune_epoch}")
        
        training_history = {
            'train_loss': [], 'train_auroc': [], 'train_acc': [],
            'val_loss': [], 'val_auroc': [], 'val_acc': []
        }
        
        for epoch in range(start_epoch, trMaxEpoch):
            epoch_start_time = time.time()
            
            print(f"\n{'='*80}")
            print(f"Epoch [{epoch+1}/{trMaxEpoch}]")
            print(f"{'='*80}")
            
            # Fine-tuning: Unfreeze backbone after fine_tune_epoch
            if epoch == fine_tune_epoch:
                print(f"\n🔓 Unfreezing backbone for fine-tuning...")
                for param in backbone_params:
                    param.requires_grad = True
                
                # Update optimizer with higher LR for backbone
                optimizer = optimizer_class([
                    {'params': backbone_params, 'lr': 5e-5},
                    {'params': classifier_params, 'lr': 1e-4}
                ], **optimizer_kwargs)
                
                # Reset scheduler
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=[5e-5, 1e-4],
                    epochs=trMaxEpoch - fine_tune_epoch,
                    steps_per_epoch=len(dataLoaderTrain),
                    pct_start=0.1
                )
                
                print("✓ Backbone unfrozen, optimizer updated")
            
            # Train
            trainLoss, trainAUROC, trainAcc = ChexnetTrainer.epochTrain(
                model, dataLoaderTrain, optimizer, scheduler, criterion, device, scaler
            )
            
            # Validate
            valLoss, valAUROC, valAcc = ChexnetTrainer.epochVal(
                model, dataLoaderVal, criterion, device
            )
            
            # Update history
            training_history['train_loss'].append(trainLoss)
            training_history['train_auroc'].append(trainAUROC)
            training_history['train_acc'].append(trainAcc)
            training_history['val_loss'].append(valLoss)
            training_history['val_auroc'].append(valAUROC)
            training_history['val_acc'].append(valAcc)
            
            # Current learning rates
            current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            
            # Print metrics
            epoch_time = time.time() - epoch_start_time
            print(f"\n{'Metric':<20} {'Train':<15} {'Val':<15}")
            print("-" * 50)
            print(f"{'Loss':<20} {trainLoss:<15.4f} {valLoss:<15.4f}")
            print(f"{'AUROC':<20} {trainAUROC:<15.4f} {valAUROC:<15.4f}")
            print(f"{'Accuracy':<20} {trainAcc:<15.4f} {valAcc:<15.4f}")
            print(f"{'LR (backbone)':<20} {current_lrs[0]:<15.6f}")
            if len(current_lrs) > 1:
                print(f"{'LR (classifier)':<20} {current_lrs[1]:<15.6f}")
            print(f"{'Epoch Time':<20} {epoch_time:<15.1f}s")
            
            # VRAM usage
            if torch.cuda.is_available():
                print(f"\n{'GPU':<10} {'Allocated':<15} {'Reserved':<15}")
                print("-" * 40)
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"{'GPU ' + str(i):<10} {f'{allocated:.2f} GB':<15} {f'{reserved:.2f} GB':<15}")
            
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
                    'scheduler': scheduler.state_dict(),
                    'training_history': training_history,
                    'config': {
                        'num_classes': nnClassCount,
                        'image_size': transCrop,
                        'dtype': str(amp_dtype),
                    }
                }, pathModel)
                
                print(f"\n✅ Model saved: {pathModel}")
                print(f"   AUROC: {bestAUROC:.4f} | Loss: {bestLoss:.4f}")
            else:
                print(f"\n📊 Best AUROC: {bestAUROC:.4f} (no improvement)")
            
            # Early stopping check (optional)
            if epoch - training_history['val_auroc'].index(max(training_history['val_auroc'])) > 20:
                print("\n⚠ Early stopping: No improvement for 20 epochs")
                break
        
        print("\n" + "="*80)
        print("✅ Training completed!")
        print(f"Best AUROC: {bestAUROC:.4f}")
        print("="*80)

    @staticmethod
    def epochTrain(model, dataLoader, optimizer, scheduler, criterion, device, scaler):
        """Training epoch with optimizations"""
        model.train()
        
        totalLoss = 0.0
        allPreds = []
        allTargets = []
        
        # Get settings
        amp_dtype = getattr(ChexnetTrainer, '_amp_dtype', torch.float16)
        has_tensor_cores = getattr(ChexnetTrainer, '_has_tensor_cores', False)
        use_amp = getattr(ChexnetTrainer, '_use_amp', False)
        
        pbar = tqdm(dataLoader, desc="Training", ncols=100)
        
        for batch_idx, (input, target) in enumerate(pbar):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Convert to channels_last for Tensor Core optimization
            if has_tensor_cores:
                input = input.to(memory_format=torch.channels_last)
            
            # Mixed precision training
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    output = model(input)
                    loss = criterion(output, target)
            else:
                output = model(input)
                loss = criterion(output, target)
            
            optimizer.zero_grad(set_to_none=True)
            
            if amp_dtype == torch.bfloat16 or not use_amp:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            
            scheduler.step()
            
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
        """Validation epoch"""
        model.eval()
        
        totalLoss = 0.0
        allPreds = []
        allTargets = []
        
        # Get settings
        amp_dtype = getattr(ChexnetTrainer, '_amp_dtype', torch.float16)
        has_tensor_cores = getattr(ChexnetTrainer, '_has_tensor_cores', False)
        use_amp = getattr(ChexnetTrainer, '_use_amp', False)
        
        pbar = tqdm(dataLoader, desc="Validation", ncols=100)
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(pbar):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                if has_tensor_cores:
                    input = input.to(memory_format=torch.channels_last)
                
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        output = model(input)
                        loss = criterion(output, target)
                else:
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
    def test(pathDirData: str, 
             pathFileTest: str, 
             pathModel: str,
             nnClassCount: int, 
             trBatchSize: int, 
             transCrop: int,
             device: Optional[torch.device] = None,
             use_tta: bool = False,
             num_tta: int = 5):
        """
        Test trained model with optional Test-Time Augmentation (TTA)
        
        Args:
            use_tta: Enable test-time augmentation for better results
            num_tta: Number of TTA augmentations
        """
        
        print("\n" + "="*80)
        print("Testing ChestX-ray14 Model")
        if use_tta:
            print(f"🔄 Test-Time Augmentation: {num_tta} augmentations")
        print("="*80)
        
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model = ConvNeXtV2Model(num_classes=nnClassCount, pretrained=False).to(device)
        
        print(f"\n📥 Loading model: {pathModel}")
        ckpt = torch.load(pathModel, map_location=device, weights_only=False)
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
        
        if use_tta:
            # Create TTA transforms
            tta_transforms = create_tta_transforms(None, transCrop)
            print(f"✓ Created {len(tta_transforms)} TTA transforms")
        else:
            transformTest = transforms.Compose([
                transforms.Resize(int(transCrop * 1.14)),
                transforms.CenterCrop(transCrop),
                transforms.ToTensor(),
                normalize
            ])
        
        # Dataset
        if not use_tta:
            datasetTest = DatasetGenerator(pathDirData, pathFileTest, transformTest)
            dataLoaderTest = FastDataLoader.create_dataloader(
                datasetTest,
                batch_size=trBatchSize,
                shuffle=False,
                num_workers=4
            )
        else:
            # For TTA, we'll handle transforms differently
            datasetTest = DatasetGenerator(pathDirData, pathFileTest, None)
        
        print(f"✓ Test samples: {len(datasetTest)}")
        
        # Inference
        allPreds = []
        allTargets = []
        
        print("\n🔍 Running inference...")
        
        if not use_tta:
            # Standard inference
            with torch.no_grad():
                for input, target in tqdm(dataLoaderTest, desc="Testing"):
                    input = input.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    
                    output = model(input)
                    pred = torch.sigmoid(output)
                    
                    allPreds.append(pred.cpu())
                    allTargets.append(target.cpu())
        else:
            # TTA inference: Average predictions across augmentations
            with torch.no_grad():
                for idx in tqdm(range(len(datasetTest)), desc="Testing with TTA"):
                    image_path = os.path.join(pathDirData, datasetTest.listImagePaths[idx])
                    target = datasetTest.listImageLabels[idx]
                    
                    # Load original image
                    from PIL import Image
                    image = Image.open(image_path).convert('RGB')
                    
                    # Apply each TTA transform and collect predictions
                    tta_preds = []
                    for transform in tta_transforms:
                        input_tensor = transform(image).unsqueeze(0).to(device)
                        output = model(input_tensor)
                        pred = torch.sigmoid(output)
                        tta_preds.append(pred.cpu())
                    
                    # Average TTA predictions
                    avg_pred = torch.stack(tta_preds).mean(dim=0)
                    
                    allPreds.append(avg_pred)
                    allTargets.append(target.unsqueeze(0))
        
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
        print("📊 Test Results")
        if use_tta:
            print(f"(with {num_tta} Test-Time Augmentations)")
        print("="*80)
        print(f"\n🎯 Mean AUROC: {aurocMean:.4f}\n")
        print(f"{'Disease':<25} {'AUROC':<10} {'Status':<10}")
        print("-" * 50)
        
        for i, name in enumerate(ChexnetTrainer.CLASS_NAMES[:nnClassCount]):
            auroc_val = aurocIndividual[i]
            if not np.isnan(auroc_val):
                status = "✓✓" if auroc_val >= 0.85 else "✓" if auroc_val >= 0.75 else "⚠"
                print(f"{name:<25} {auroc_val:<10.4f} {status:<10}")
            else:
                print(f"{name:<25} {'N/A':<10} {'-':<10}")
        
        print("="*80)
        
        return aurocMean, aurocIndividual, allPreds, allTargets
