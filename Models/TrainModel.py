# TrainModel.py - Optimized with DDP 
import os
import time
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# === [THÊM] ===
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# === [HẾT THÊM] ===
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

    # === [THÊM] ===
    @staticmethod
    def setup_distributed():
        """Setup distributed training environment"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        else:
            rank = 0
            world_size = 1
            local_rank = 0
        
        if world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank

    @staticmethod
    def cleanup_distributed():
        """Cleanup distributed training"""
        if dist.is_initialized():
            dist.destroy_process_group()
    # === [HẾT THÊM] ===

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
        - Multi-GPU support with DDP (DistributedDataParallel)
        - Class weighting for imbalanced data
        - Fine-tuning strategy (freeze -> unfreeze)
        - Mixed precision training (bf16/fp16)
        """

        # === [SỬA] ===
        # Setup distributed
        rank, world_size, local_rank = ChexnetTrainer.setup_distributed()
        is_main_process = rank == 0
        
        if is_main_process:
            print("="*80)
            print("ChestX-ray14 Multi-Label Classification Training")
            print("Backbone: ConvNeXtV2-Large")
            print("Strategy: DistributedDataParallel + Progressive Fine-tuning")
            print("="*80)

        # ---- Device Setup (DDP)
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        
        if is_main_process:
            print(f"\nDistributed Training Setup:")
            print(f"  World Size: {world_size}")
            print(f"  Rank: {rank}")
            print(f"  Local Rank: {local_rank}")
            print(f"  Device: {device}")
        # === [HẾT SỬA] ===
        
        # ---- Tensor Core Optimization
        has_tensor_cores = False
        amp_dtype = torch.float16
        
        if torch.cuda.is_available():
            compute_cap = torch.cuda.get_device_capability(local_rank)
            
            # === [SỬA] === (Chỉ in từ main process)
            if is_main_process:
                gpu_name = torch.cuda.get_device_name(local_rank)
                memory_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
                
                print(f"\nGPU {local_rank}: {gpu_name}")
                print(f"  Memory: {memory_gb:.2f} GB")
                print(f"  Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
            
            if compute_cap[0] >= 7:
                has_tensor_cores = True
                if compute_cap[0] >= 8:
                    amp_dtype = torch.bfloat16
                    if is_main_process:
                        print(f"  ✓ Tensor Cores: ENABLED (using bfloat16)")
                else:
                    if is_main_process:
                        print(f"  ✓ Tensor Cores: ENABLED (using float16)")
            
            if has_tensor_cores and amp_dtype == torch.bfloat16:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if is_main_process:
                    print("  ✓ TF32 enabled")
            
            torch.backends.cudnn.benchmark = True
            # === [HẾT SỬA] ===

        # ---- Model
        if is_main_process:
            print(f"\n🏗️ Building model...")
        
        model = ConvNeXtV2Model(
            num_classes=nnClassCount,
            pretrained=nnIsTrained,
            dropout_rate=0.3
        ).to(device)
        
        if has_tensor_cores:
            model = model.to(memory_format=torch.channels_last)
            if is_main_process:
                print("  ✓ Model converted to channels_last format")
        
        if use_compile and has_tensor_cores and hasattr(torch, 'compile'):
            try:
                if is_main_process:
                    print("  🔥 Compiling model with torch.compile()...")
                model = torch.compile(model, mode='max-autotune')
                if is_main_process:
                    print("  ✓ Model compiled")
            except Exception as e:
                if is_main_process:
                    print(f"  ⚠ Could not compile: {e}")
        
        # === [SỬA] ===
        # Wrap with DDP (thay vì DataParallel)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            if is_main_process:
                print(f"\n🔀 Using DistributedDataParallel with {world_size} GPUs")
        # === [XÓA] ===
        # if torch.cuda.device_count() > 1:
        #     print(f"\n🔀 Using {torch.cuda.device_count()} GPUs (DataParallel)")
        #     model = torch.nn.DataParallel(model)
        # === [HẾT SỬA] ===

        # ---- Data Transforms (Giữ nguyên)
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        
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

        # ---- Datasets
        if is_main_process:
            print("\n📂 Loading datasets...")
        
        datasetTrain = DatasetGenerator(
            pathDirData, pathFileTrain, transformTrain,
            cache_images=False,
            preload_images=False
        )
        datasetVal = DatasetGenerator(
            pathDirData, pathFileVal, transformVal,
            cache_images=False,
            preload_images=False
        )
        
        if is_main_process:
            datasetTrain.print_statistics()

        # === [THÊM] ===
        # Distributed samplers
        train_sampler = DistributedSampler(
            datasetTrain, 
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        ) if world_size > 1 else None
        
        val_sampler = DistributedSampler(
            datasetVal,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        ) if world_size > 1 else None
        # === [HẾT THÊM] ===

        # ---- DataLoaders
        # === [SỬA] ===
        # num_workers = min(8, os.cpu_count() or 4) # (Logic của Đoạn 2)
        # Chia num_workers cho số lượng GPU
        num_workers = min(8, os.cpu_count() or 4) // max(1, world_size)
        
        # Sử dụng DataLoader chuẩn (tương thích DDP) thay vì FastDataLoader
        dataLoaderTrain = DataLoader(
            datasetTrain,
            batch_size=trBatchSize,
            sampler=train_sampler, # <== Thêm sampler
            shuffle=(train_sampler is None), # <== Sửa shuffle
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        )
        
        dataLoaderVal = DataLoader(
            datasetVal,
            batch_size=trBatchSize,
            sampler=val_sampler, # <== Thêm sampler
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        )

        if is_main_process:
            print(f"✓ Train samples: {len(datasetTrain)}")
            print(f"✓ Val samples: {len(datasetVal)}")
            print(f"✓ Batch size per GPU: {trBatchSize}")
            print(f"✓ Effective batch size: {trBatchSize * world_size}")
            print(f"✓ Steps per epoch: {len(dataLoaderTrain)}")
        # === [HẾT SỬA] ===

        # ---- Class Weights
        class_weights = None
        if use_class_weights:
            if is_main_process:
                print("\n⚖️ Computing class weights for imbalanced data...")
            class_weights = datasetTrain.get_class_weights(smooth=0.1).to(device)
            if is_main_process:
                print("Class weights applied:")
                for i, w in enumerate(class_weights[:5]): # In 5 class đầu
                    print(f"  Class {i:2d} ({ChexnetTrainer.CLASS_NAMES[i]:20s}): {w:.4f}")

        # ---- Optimizer
        # === [SỬA] === (Lấy base_model từ DDP wrapper nếu có)
        base_model = model.module if isinstance(model, DDP) else model
        
        backbone_params = []
        classifier_params = []
        
        for name, param in base_model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        if is_main_process:
            print(f"\n🎯 Optimizer configuration (Progressive Fine-tuning):")
            print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}")
            print(f"  Classifier params: {sum(p.numel() for p in classifier_params):,}")

        # Giai đoạn 1: Đóng băng backbone
        if is_main_process:
            print(f"  Phase 1 (Epoch 0-{fine_tune_epoch}): Train classifier only (backbone frozen)")
        for param in backbone_params:
            param.requires_grad = False
        for param in classifier_params:
            param.requires_grad = True
            
        optimizer_class = optim.AdamW
        optimizer_kwargs = {
            'lr': 1e-4, # Sẽ được ghi đè bởi scheduler
            'weight_decay': 0.05,
            'betas': (0.9, 0.999)
        }
        
        if has_tensor_cores and amp_dtype == torch.bfloat16:
            try:
                optimizer_kwargs['fused'] = True
                if is_main_process:
                    print(f"  ✓ Using fused AdamW optimizer")
            except:
                pass
        
        # Optimizer ban đầu chỉ cho classifier
        optimizer = optimizer_class(classifier_params, **optimizer_kwargs)
        
        # Scheduler cho Giai đoạn 1
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-3, # Max LR cho classifier
            epochs=fine_tune_epoch,
            steps_per_epoch=len(dataLoaderTrain),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        # === [HẾT SỬA] ===

        # ---- Loss
        if class_weights is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            if is_main_process: print("  ✓ Using weighted BCE loss")
        else:
            criterion = nn.BCEWithLogitsLoss()
            if is_main_process: print("  ✓ Using standard BCE loss")

        # ---- Mixed Precision
        use_amp = torch.cuda.is_available()
        # === [SỬA] === (Sử dụng API mới của AMP)
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))
        
        if use_amp and is_main_process:
            dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
            print(f"  ✓ Using {dtype_name} mixed precision")
        
        ChexnetTrainer._amp_dtype = amp_dtype
        ChexnetTrainer._has_tensor_cores = has_tensor_cores
        ChexnetTrainer._use_amp = use_amp

        # ---- Load checkpoint
        bestLoss = float("inf")
        bestAUROC = 0.0
        
        if checkpoint and os.path.exists(checkpoint):
            if is_main_process:
                print(f"\n📥 Loading checkpoint: {checkpoint}")
            
            # === [THÊM] === (Cần map_location cho DDP)
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            ckpt = torch.load(checkpoint, map_location=map_location, weights_only=False)
            # === [HẾT THÊM] ===
            
            state_dict = ckpt['state_dict']
            # === [SỬA] === (Sử dụng base_model đã định nghĩa ở trên)
            target_model = base_model 
            
            try:
                target_model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                if is_main_process: print(f"Warning: {e}")
                if any(k.startswith('module.') for k in state_dict.keys()):
                    stripped = OrderedDict(
                        (k[7:], v) if k.startswith('module.') else (k, v)
                        for k, v in state_dict.items()
                    )
                    target_model.load_state_dict(stripped, strict=False)
            
            # Chỉ load optimizer state nếu vẫn đang ở Giai đoạn 1
            if 'optimizer' in ckpt and start_epoch < fine_tune_epoch:
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                except:
                    if is_main_process: print("  ⚠ Could not load optimizer state")
            
            # (Scheduler sẽ tự động điều chỉnh nếu start_epoch > 0)
            
            bestLoss = ckpt.get('best_loss', bestLoss)
            bestAUROC = ckpt.get('best_auroc', bestAUROC)
            start_epoch = ckpt.get('epoch', start_epoch)
            
            if is_main_process:
                print(f"✓ Resumed from epoch {start_epoch}")
                print(f"  Best AUROC: {bestAUROC:.4f}")

        # ---- Training Loop
        if is_main_process:
            print("\n" + "="*80)
            print("🚀 Starting Training")
            print("="*80)
            print(f"Phase 1: Epochs 0-{fine_tune_epoch} (Classifier only)")
            print(f"Phase 2: Epochs {fine_tune_epoch}-{trMaxEpoch} (Full fine-tuning)")
        
        training_history = {
            'train_loss': [], 'train_auroc': [], 'train_acc': [],
            'val_loss': [], 'val_auroc': [], 'val_acc': []
        }
        
        for epoch in range(start_epoch, trMaxEpoch):
            # === [THÊM] === (Rất quan trọng cho DDP)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            # === [HẾT THÊM] ===
            
            epoch_start_time = time.time()
            
            # === [SỬA] === (Chỉ in từ main process)
            if is_main_process:
                print(f"\n{'='*80}")
                print(f"Epoch [{epoch+1}/{trMaxEpoch}]", end="")
                if epoch < fine_tune_epoch:
                    print(f" - Phase 1: Classifier Only")
                else:
                    print(f" - Phase 2: Full Fine-tuning")
                print(f"{'='*80}")
            
            # Giai đoạn 2: Mở đóng băng backbone
            if epoch == fine_tune_epoch:
                if is_main_process:
                    print(f"\n🔓 Unfreezing backbone for fine-tuning...")
                
                # Mở đóng băng
                for param in base_model.parameters():
                    param.requires_grad = True
                
                # Lấy lại danh sách param (bây giờ backbone có requires_grad=True)
                backbone_params_optim = [p for n, p in base_model.named_parameters() 
                                         if 'classifier' not in n and p.requires_grad]
                classifier_params_optim = [p for n, p in base_model.named_parameters() 
                                           if 'classifier' in n and p.requires_grad]
                
                # Tạo optimizer MỚI
                optimizer = optimizer_class([
                    {'params': backbone_params_optim, 'lr': 5e-5}, # LR thấp cho backbone
                    {'params': classifier_params_optim, 'lr': 1e-4} # LR cao hơn cho classifier
                ], **optimizer_kwargs)
                
                # Tạo scheduler MỚI cho Giai đoạn 2
                remaining_epochs = trMaxEpoch - fine_tune_epoch
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=[5e-5, 1e-4], # LR tối đa cho từng nhóm
                    epochs=remaining_epochs,
                    steps_per_epoch=len(dataLoaderTrain),
                    pct_start=0.1
                )
                
                if is_main_process:
                    print("  ✓ Backbone unfrozen, optimizer & scheduler reset")
            
            # Train
            # === [SỬA] === (Truyền thêm cờ DDP)
            trainLoss, trainAUROC, trainAcc = ChexnetTrainer.epochTrain(
                model, dataLoaderTrain, optimizer, scheduler, criterion, 
                device, scaler, is_main_process, world_size
            )
            
            # Validate
            valLoss, valAUROC, valAcc = ChexnetTrainer.epochVal(
                model, dataLoaderVal, criterion, device, 
                is_main_process, world_size
            )
            # === [HẾT SỬA] ===
            
            # === [SỬA] === (Chỉ thực hiện trên main process)
            if is_main_process:
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
                
                if len(current_lrs) > 1:
                    print(f"{'LR (backbone)':<20} {current_lrs[0]:<15.6f}")
                    print(f"{'LR (classifier)':<20} {current_lrs[1]:<15.6f}")
                else:
                    print(f"{'LR':<20} {current_lrs[0]:<15.6f}")
                
                print(f"{'Epoch Time':<20} {epoch_time:<15.1f}s")
                
                # VRAM usage (chỉ cho GPU này)
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
                    reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
                    print(f"{'GPU Memory':<20} {allocated:.2f} GB / {reserved:.2f} GB")
                
                # Save best model
                is_best = valAUROC > bestAUROC
                if is_best:
                    bestAUROC = valAUROC
                    bestLoss = valLoss
                    
                    os.makedirs(os.path.dirname(pathModel) or '.', exist_ok=True)
                    
                    # Lấy state_dict từ model.module khi dùng DDP
                    state_dict_to_save = (
                        model.module.state_dict() 
                        if isinstance(model, DDP) 
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
            # === [HẾT SỬA] ===
            
            # (Early stopping có thể thêm ở đây nếu muốn)

        if is_main_process:
            print("\n" + "="*80)
            print("✅ Training completed!")
            print(f"Best AUROC: {bestAUROC:.4f}")
            print("="*80)
        
        # === [THÊM] ===
        ChexnetTrainer.cleanup_distributed()
        # === [HẾT THÊM] ===

    @staticmethod
    # === [SỬA] === (Thêm is_main_process, world_size)
    def epochTrain(model, dataLoader, optimizer, scheduler, criterion, 
                   device, scaler, is_main_process, world_size):
        model.train()
        
        totalLoss = 0.0
        allPreds = []
        allTargets = []
        
        amp_dtype = getattr(ChexnetTrainer, '_amp_dtype', torch.float16)
        has_tensor_cores = getattr(ChexnetTrainer, '_has_tensor_cores', False)
        use_amp = getattr(ChexnetTrainer, '_use_amp', False)
        
        # === [SỬA] === (Chỉ main process mới có tqdm)
        if is_main_process:
            pbar = tqdm(dataLoader, desc="Training", ncols=100)
        else:
            pbar = dataLoader
        
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
            
            totalLoss += loss.item()
            allPreds.append(torch.sigmoid(output).detach().cpu())
            allTargets.append(target.detach().cpu())
            
            # === [SỬA] === (Cập nhật tqdm trên main process)
            if is_main_process and isinstance(pbar, tqdm):
                avgLoss = totalLoss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{avgLoss:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        # === [THÊM] === (Thu thập metrics từ tất cả các process)
        allPreds = torch.cat(allPreds, dim=0).numpy()
        allTargets = torch.cat(allTargets, dim=0).numpy()
        
        if world_size > 1:
            pred_list = [None] * world_size
            target_list = [None] * world_size
            
            dist.all_gather_object(pred_list, allPreds)
            dist.all_gather_object(target_list, allTargets)
            
            if is_main_process:
                allPreds = np.concatenate(pred_list, axis=0)
                allTargets = np.concatenate(target_list, axis=0)
        
        # Chỉ main process tính metrics
        if is_main_process:
            auroc = ChexnetTrainer.computeAUROC_mean(allTargets, allPreds)
            acc = ChexnetTrainer.computeAccuracy(allTargets, allPreds)
        else:
            auroc = acc = 0.0 # Các process khác không cần tính
        
        return totalLoss / len(dataLoader), auroc, acc
        # === [HẾT THÊM] ===

    @staticmethod
    # === [SỬA] === (Thêm is_main_process, world_size)
    def epochVal(model, dataLoader, criterion, device, 
                 is_main_process, world_size):
        model.eval()
        
        totalLoss = 0.0
        allPreds = []
        allTargets = []
        
        amp_dtype = getattr(ChexnetTrainer, '_amp_dtype', torch.float16)
        has_tensor_cores = getattr(ChexnetTrainer, '_has_tensor_cores', False)
        use_amp = getattr(ChexnetTrainer, '_use_amp', False)
        
        # === [SỬA] ===
        if is_main_process:
            pbar = tqdm(dataLoader, desc="Validation", ncols=100)
        else:
            pbar = dataLoader
        
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
                
                if is_main_process and isinstance(pbar, tqdm):
                    avgLoss = totalLoss / (batch_idx + 1)
                    pbar.set_postfix({'loss': f'{avgLoss:.4f}'})

        # === [THÊM] === (Thu thập metrics từ tất cả các process)
        allPreds = torch.cat(allPreds, dim=0).numpy()
        allTargets = torch.cat(allTargets, dim=0).numpy()
        
        if world_size > 1:
            pred_list = [None] * world_size
            target_list = [None] * world_size
            
            dist.all_gather_object(pred_list, allPreds)
            dist.all_gather_object(target_list, allTargets)
            
            if is_main_process:
                allPreds = np.concatenate(pred_list, axis=0)
                allTargets = np.concatenate(target_list, axis=0)
        
        if is_main_process:
            auroc = ChexnetTrainer.computeAUROC_mean(allTargets, allPreds)
            acc = ChexnetTrainer.computeAccuracy(allTargets, allPreds)
        else:
            auroc = acc = 0.0
        
        return totalLoss / len(dataLoader), auroc, acc
        # === [HẾT THÊM] ===

    @staticmethod
    def computeAUROC_mean(dataGT, dataPRED):
        # (Giữ nguyên)
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
        # (Giữ nguyên)
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
        
        # (Hàm test thường không chạy DDP, nó chạy trên 1 GPU sau khi train)
        # (Giữ nguyên hàm test của Đoạn 2)
        
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
        
        # Xử lý state_dict nếu nó được lưu từ DDP (có prefix 'module.')
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
            # Sử dụng DataLoader chuẩn
            dataLoaderTest = DataLoader(
                datasetTest,
                batch_size=trBatchSize,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        else:
            datasetTest = DatasetGenerator(pathDirData, pathFileTest, None)
        
        print(f"✓ Test samples: {len(datasetTest)}")
        
        # Inference
        allPreds = []
        allTargets = []
        
        print("\n🔍 Running inference...")
        
        if not use_tta:
            with torch.no_grad():
                for input, target in tqdm(dataLoaderTest, desc="Testing"):
                    input = input.to(device, non_blocking=True)
                    output = model(input)
                    pred = torch.sigmoid(output)
                    
                    allPreds.append(pred.cpu())
                    allTargets.append(target.cpu())
        else:
            from PIL import Image
            with torch.no_grad():
                for idx in tqdm(range(len(datasetTest)), desc="Testing with TTA"):
                    image_path = os.path.join(pathDirData, datasetTest.listImagePaths[idx])
                    target = datasetTest.listImageLabels[idx]
                    
                    image = Image.open(image_path).convert('RGB')
                    
                    tta_preds = []
                    for transform in tta_transforms:
                        input_tensor = transform(image).unsqueeze(0).to(device)
                        output = model(input_tensor)
                        pred = torch.sigmoid(output)
                        tta_preds.append(pred.cpu())
                    
                    avg_pred = torch.stack(tta_preds).mean(dim=0)
                    allPreds.append(avg_pred)
                    allTargets.append(target.unsqueeze(0))
        
        allPreds = torch.cat(allPreds, dim=0).numpy()
        allTargets = torch.cat(allTargets, dim=0).numpy()
        
        # Compute AUROC
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
