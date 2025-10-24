import os
import time
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

try:
    from Models.Model import MultiModelArchitecture, DenseNet121, ConvNeXtV2Large
    from Models.read_data import DatasetGenerator
except ImportError:
    from Model import MultiModelArchitecture, DenseNet121, ConvNeXtV2Large
    from read_data import DatasetGenerator


class ChexnetTrainer:
    
    @staticmethod
    def detect_gpus():
        """Detect available GPUs and return device information"""
        if not torch.cuda.is_available():
            print("❌ No CUDA GPUs detected. Using CPU.")
            return torch.device('cpu'), 0, []
        
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        
        print(f"✅ Detected {gpu_count} CUDA GPU(s):")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_info.append({
                'id': i,
                'name': gpu_name,
                'memory_gb': gpu_memory
            })
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        device = torch.device('cuda:0')
        return device, gpu_count, gpu_info

    @staticmethod
    def train(pathDirData, pathFileTrain, pathFileVal,
              nnIsTrained, nnClassCount,
              trBatchSize, trMaxEpoch,
              transCrop, pathModel='Trainedmodel/chexnetmodel.pth',
              checkpoint=None, model_type='densenet121',
              custom_lr=None, custom_batch_size=None):
        """
        Enhanced training with multi-GPU support and real-time monitoring.
        
        Args:
            model_type: Choose from:
                DenseNet: 'densenet121', 'densenet169', 'densenet201'
                ConvNeXtV2: 'convnextv2_base', 'convnextv2_large', 'convnextv2_huge'
                EfficientNetV2: 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l'
                Swin: 'swin_tiny', 'swin_small', 'swin_base'
            custom_lr: Override recommended learning rate
            custom_batch_size: Override recommended batch size
        """
        
        print("\n" + "="*80)
        print("🚀 INITIALIZING TRAINING")
        print("="*80)
        
        # Detect GPUs
        device, gpu_count, gpu_info = ChexnetTrainer.detect_gpus()
        
        # Create model
        print(f"\n📦 Loading model: {model_type}")
        model = MultiModelArchitecture(model_type, nnClassCount, nnIsTrained).to(device)
        
        # Multi-GPU setup
        if gpu_count > 1:
            print(f"🔄 Using DataParallel across {gpu_count} GPUs")
            model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))
        
        # Get recommended hyperparameters
        if custom_lr is None:
            lr = MultiModelArchitecture.get_recommended_lr(model_type)
            print(f"📊 Using recommended learning rate: {lr}")
        else:
            lr = custom_lr
            print(f"📊 Using custom learning rate: {lr}")
        
        if custom_batch_size is not None:
            trBatchSize = custom_batch_size
            print(f"📦 Using custom batch size: {trBatchSize}")
        elif gpu_info:
            recommended_bs = MultiModelArchitecture.get_recommended_batch_size(
                model_type, gpu_info[0]['memory_gb']
            )
            if trBatchSize != recommended_bs:
                print(f"💡 Recommended batch size for your GPU: {recommended_bs}")
                print(f"📦 Using specified batch size: {trBatchSize}")
        
        # Data transforms
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        transformTrain = transforms.Compose([
            transforms.RandomResizedCrop(transCrop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transformVal = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(transCrop),
            transforms.ToTensor(),
            normalize
        ])
        
        # Datasets & Loaders
        print("\n📚 Loading datasets...")
        datasetTrain = DatasetGenerator(pathDirData, pathFileTrain, transformTrain)
        datasetVal = DatasetGenerator(pathDirData, pathFileVal, transformVal)
        
        print(f"   Training samples: {len(datasetTrain)}")
        print(f"   Validation samples: {len(datasetVal)}")
        
        use_cuda = torch.cuda.is_available()
        num_workers = min(8, os.cpu_count() or 2) if use_cuda else 0
        
        dataLoaderTrain = DataLoader(
            datasetTrain, batch_size=trBatchSize,
            shuffle=True, num_workers=num_workers, 
            pin_memory=use_cuda, persistent_workers=use_cuda
        )
        dataLoaderVal = DataLoader(
            datasetVal, batch_size=trBatchSize,
            shuffle=False, num_workers=num_workers, 
            pin_memory=use_cuda, persistent_workers=use_cuda
        )
        
        # Optimizer & Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
        
        # Loss
        criterion = nn.BCELoss()
        
        # Load checkpoint if provided
        start_epoch = 0
        bestLoss = float("inf")
        if checkpoint:
            print(f"\n📂 Loading checkpoint: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location=device)
            state_dict = ckpt['state_dict']
            target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            
            try:
                target_model.load_state_dict(state_dict)
            except Exception:
                if any(k.startswith('module.') for k in state_dict.keys()):
                    stripped = OrderedDict((k[7:], v) if k.startswith('module.') else (k, v)
                                           for k, v in state_dict.items())
                    target_model.load_state_dict(stripped)
                else:
                    raise
            
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0)
            bestLoss = ckpt.get('best_loss', bestLoss)
            print(f"   Resumed from epoch {start_epoch}, best loss: {bestLoss:.4f}")
        
        # Training loop
        print("\n" + "="*80)
        print(f"🏋️ STARTING TRAINING (Epochs {start_epoch+1} to {trMaxEpoch})")
        print("="*80 + "\n")
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, trMaxEpoch):
            epoch_start = time.time()
            
            # Train
            trainLoss, train_time = ChexnetTrainer.epochTrain(
                model, dataLoaderTrain, optimizer, criterion, device, epoch+1
            )
            
            # Validate
            valLoss, val_time = ChexnetTrainer.epochVal(
                model, dataLoaderVal, criterion, device, epoch+1
            )
            
            # Update scheduler
            scheduler.step(valLoss)
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - training_start_time
            
            # Print epoch summary
            print(f"\n{'='*80}")
            print(f"📊 EPOCH {epoch+1}/{trMaxEpoch} SUMMARY")
            print(f"{'='*80}")
            print(f"   Train Loss:      {trainLoss:.4f}  (⏱️  {train_time:.1f}s)")
            print(f"   Val Loss:        {valLoss:.4f}  (⏱️  {val_time:.1f}s)")
            print(f"   Learning Rate:   {current_lr:.2e}")
            print(f"   Epoch Time:      {epoch_time:.1f}s")
            print(f"   Total Time:      {ChexnetTrainer._format_time(total_time)}")
            
            # Save best model
            if valLoss < bestLoss:
                bestLoss = valLoss
                os.makedirs(os.path.dirname(pathModel) or '.', exist_ok=True)
                
                state_dict_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': state_dict_to_save,
                    'best_loss': bestLoss,
                    'optimizer': optimizer.state_dict(),
                    'model_type': model_type
                }, pathModel)
                
                print(f"   ✅ BEST MODEL SAVED → {pathModel}")
            else:
                print(f"   💾 No improvement (best: {bestLoss:.4f})")
            
            print(f"{'='*80}\n")
        
        print("✅ Training completed!")
        print(f"🏆 Best validation loss: {bestLoss:.4f}")
        print(f"⏱️  Total training time: {ChexnetTrainer._format_time(time.time() - training_start_time)}")

    @staticmethod
    def epochTrain(model, dataLoader, optimizer, criterion, device, epoch_num):
        """Training epoch with real-time loss display"""
        model.train()
        totalLoss = 0.0
        count = 0
        
        start_time = time.time()
        
        pbar = tqdm(dataLoader, desc=f"🔥 Training Epoch {epoch_num}", 
                   unit="batch", dynamic_ncols=True)
        
        for batch_idx, (input, target) in enumerate(pbar):
            non_blocking = torch.cuda.is_available()
            input = input.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            
            output = model(input)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            totalLoss += loss.item()
            count += 1
            
            # Real-time loss display
            avg_loss = totalLoss / count
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })
        
        elapsed_time = time.time() - start_time
        return totalLoss / count if count > 0 else float("inf"), elapsed_time

    @staticmethod
    def epochVal(model, dataLoader, criterion, device, epoch_num):
        """Validation epoch with real-time loss display"""
        model.eval()
        totalLoss = 0.0
        count = 0
        
        start_time = time.time()
        
        pbar = tqdm(dataLoader, desc=f"✅ Validation Epoch {epoch_num}", 
                   unit="batch", dynamic_ncols=True)
        
        with torch.no_grad():
            for input, target in pbar:
                non_blocking = torch.cuda.is_available()
                input = input.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                
                output = model(input)
                loss = criterion(output, target)
                
                totalLoss += loss.item()
                count += 1
                
                # Real-time loss display
                avg_loss = totalLoss / count
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}'
                })
        
        elapsed_time = time.time() - start_time
        return totalLoss / count if count > 0 else float("inf"), elapsed_time

    @staticmethod
    def computeAUROC(dataGT, dataPRED, classCount):
        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except:
                outAUROC.append(float("nan"))
        return outAUROC

    @staticmethod
    def test(pathDirData, pathFileTest, pathModel,
             nnClassCount, trBatchSize, transCrop,
             device=None, model_type=None):
        """Enhanced testing with multi-model support"""
        
        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                       'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                       'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                       'Pleural_Thickening', 'Hernia', 'No Finding']
        
        print("\n" + "="*80)
        print("🔍 STARTING TESTING")
        print("="*80)
        
        # Detect device
        if device is None:
            device, gpu_count, gpu_info = ChexnetTrainer.detect_gpus()
        
        # Load checkpoint
        print(f"\n📂 Loading checkpoint: {pathModel}")
        ckpt = torch.load(pathModel, map_location=device)
        
        if model_type is None:
            model_type = ckpt.get('model_type', 'densenet121')
            print(f"   Auto-detected model type: {model_type}")
        
        # Create model
        print(f"\n📦 Creating {model_type} model...")
        model = MultiModelArchitecture(model_type, nnClassCount, isTrained=False).to(device)
        
        # Load weights
        state_dict = ckpt['state_dict']
        try:
            model.load_state_dict(state_dict)
        except Exception:
            if any(k.startswith('module.') for k in state_dict.keys()):
                stripped = OrderedDict((k[7:], v) if k.startswith('module.') else (k, v)
                                       for k, v in state_dict.items())
                model.load_state_dict(stripped)
            else:
                raise
        
        model.eval()
        
        # Data transforms
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        transformTest = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(transCrop),
            transforms.ToTensor(),
            normalize
        ])
        
        # Dataset
        print(f"\n📚 Loading test dataset...")
        datasetTest = DatasetGenerator(pathDirData, pathFileTest, transformTest)
        print(f"   Test samples: {len(datasetTest)}")
        
        use_cuda = torch.cuda.is_available()
        num_workers = min(8, os.cpu_count() or 2) if use_cuda else 0
        
        dataLoaderTest = DataLoader(
            datasetTest, batch_size=trBatchSize,
            shuffle=False, num_workers=num_workers, 
            pin_memory=use_cuda
        )
        
        # Testing
        print("\n🧪 Running inference...")
        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)
        
        with torch.no_grad():
            for input, target in tqdm(dataLoaderTest, desc="Testing", unit="batch"):
                non_blocking = torch.cuda.is_available()
                input = input.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                out = model(input)
                outGT = torch.cat((outGT, target), 0)
                outPRED = torch.cat((outPRED, out), 0)
        
        # Compute AUROC
        print("\n📊 Computing AUROC scores...")
        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.nanmean(aurocIndividual)
        
        print("\n" + "="*80)
        print("📊 TEST RESULTS")
        print("="*80)
        print(f"\n🏆 Mean AUROC: {aurocMean:.4f}\n")
        print("Per-class AUROC:")
        print("-" * 40)
        for i, name in enumerate(CLASS_NAMES[:nnClassCount]):
            score = aurocIndividual[i]
            emoji = "✅" if score > 0.7 else "⚠️" if score > 0.6 else "❌"
            print(f"  {emoji} {name:20s}: {score:.4f}")
        print("="*80 + "\n")
        
        return aurocMean, aurocIndividual
    
    @staticmethod
    def _format_time(seconds):
        """Format seconds to human readable time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
