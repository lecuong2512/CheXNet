# Models/TrainModel.py (updated with auto GPU detection)
import os
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

try:
    from Models.Model import SwinTransformer 
    from Models.read_data import DatasetGenerator
except ImportError:
    from Model import SwinTransformer
    from read_data import DatasetGenerator

class ChexnetTrainer:

    @staticmethod
    def train(pathDirData, pathFileTrain, pathFileVal,
              nnIsTrained, nnClassCount,
              trBatchSize, trMaxEpoch,
              transCrop, pathModel='CheXNet/Trainedmodel/chexnetmodel.pth',
              checkpoint=None,
              model_variant='swin_large_patch4_window7_224.ms_in22k_ft_in1k'):

        # ---- Device & GPU Detection
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"\n{'='*80}")
            print(f"GPU Detection: Found {num_gpus} GPU(s)")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"{'='*80}\n")
            device = torch.device('cuda')
        else:
            print("\n" + "="*80)
            print("WARNING: No GPU detected! Training will use CPU (very slow)")
            print("="*80 + "\n")
            device = torch.device('cpu')
            num_gpus = 0
        
        # ---- Load Model
        model = SwinTransformer(nnClassCount, nnIsTrained, model_variant=model_variant).to(device)
        
        # ---- Auto DataParallel for Multi-GPU
        if num_gpus > 1:
            print(f"Using DataParallel across {num_gpus} GPUs")
            model = torch.nn.DataParallel(model)
            # Tăng batch size tự động theo số GPU (tùy chọn)
            effective_batch_size = trBatchSize * num_gpus
            print(f"Effective batch size: {trBatchSize} x {num_gpus} = {effective_batch_size}")
        elif num_gpus == 1:
            print("Using single GPU")
        
        # ---- Data transforms
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

        # ---- Datasets & Loaders
        datasetTrain = DatasetGenerator(pathDirData, pathFileTrain, transformTrain)
        datasetVal   = DatasetGenerator(pathDirData, pathFileVal, transformVal)

        # Tự động điều chỉnh num_workers dựa trên GPU
        num_workers = 4 * num_gpus if num_gpus > 0 else 0
        num_workers = min(num_workers, os.cpu_count() or 4)  # Giới hạn bởi số CPU cores
        
        print(f"DataLoader workers: {num_workers}")
        
        dataLoaderTrain = DataLoader(
            datasetTrain, 
            batch_size=trBatchSize,
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=(num_gpus > 0),
            persistent_workers=(num_workers > 0)
        )
        dataLoaderVal = DataLoader(
            datasetVal, 
            batch_size=trBatchSize,
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=(num_gpus > 0),
            persistent_workers=(num_workers > 0)
        )

        # ---- Optimizer & Scheduler
        print(f"Using AdamW optimizer with lr=5e-5 and weight_decay=1e-5 for {model_variant}")
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

        # ---- Loss
        criterion = nn.BCEWithLogitsLoss()

        # ---- Load checkpoint
        bestLoss = float("inf")
        bestAUROC = 0.0
        startEpoch = 0
        if checkpoint and os.path.isfile(checkpoint):
            ckpt = torch.load(checkpoint, map_location=device)
            state_dict = ckpt.get('state_dict', ckpt)
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
            bestLoss = ckpt.get('best_loss', bestLoss)
            bestAUROC = ckpt.get('best_auroc', bestAUROC)
            startEpoch = ckpt.get('epoch', 0)
            print(f"Loaded checkpoint '{checkpoint}' (epoch={startEpoch}, best_loss={bestLoss:.4f}, best_auroc={bestAUROC:.4f})")

        # ---- Training loop
        print("\n" + "="*80)
        print(f"Starting training {model_variant} from epoch {startEpoch+1} to {trMaxEpoch}")
        print("="*80 + "\n")
        
        for epoch in range(startEpoch, trMaxEpoch):
            trainLoss = ChexnetTrainer.epochTrain(model, dataLoaderTrain, optimizer, criterion, device)
            valLoss, valAUROC = ChexnetTrainer.epochVal(model, dataLoaderVal, criterion, device, nnClassCount)
            scheduler.step(valLoss)
            current_lr = optimizer.param_groups[0]['lr']

            saved = False
            save_reason = [] 
            
            if valLoss < bestLoss:
                bestLoss = valLoss
                saved = True
                save_reason.append("best_loss")
                
            if valAUROC > bestAUROC:
                bestAUROC = valAUROC
                saved = True
                save_reason.append("best_auroc")
            
            if saved:
                os.makedirs(os.path.dirname(pathModel) or '.', exist_ok=True)
                state_dict_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': state_dict_to_save,
                    'best_loss': bestLoss,
                    'best_auroc': bestAUROC,
                    'optimizer': optimizer.state_dict(),
                    'model_variant': model_variant
                }, pathModel)
                reason_str = "+".join(save_reason)
                print(f"[Epoch {epoch+1:3d}/{trMaxEpoch}] [SAVED-{reason_str}] train_loss={trainLoss:.4f} val_loss={valLoss:.4f} val_auroc={valAUROC:.4f} lr={current_lr:.6f}")
            else:
                print(f"[Epoch {epoch+1:3d}/{trMaxEpoch}] [----------] train_loss={trainLoss:.4f} val_loss={valLoss:.4f} val_auroc={valAUROC:.4f} lr={current_lr:.6f}")
            
            # In GPU memory usage sau mỗi epoch
            if torch.cuda.is_available():
                for i in range(num_gpus):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"  GPU {i} Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        print("\n" + "="*80)
        print(f"Training completed! Best val_loss={bestLoss:.4f}, Best val_auroc={bestAUROC:.4f}")
        print("="*80 + "\n")

    @staticmethod
    def epochTrain(model, dataLoader, optimizer, criterion, device):
        model.train()
        totalLoss, count = 0.0, 0
        pbar = tqdm(dataLoader, desc="Training", leave=False)
        for input, target in pbar:
            non_blocking = (device.type == 'cuda')
            input = input.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            
            output = model(input)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()
            count += 1
            avgLoss = totalLoss / count
            pbar.set_postfix(loss=f"{avgLoss:.4f}")
        return totalLoss / count if count > 0 else float("inf")

    @staticmethod
    def epochVal(model, dataLoader, criterion, device, classCount):
        model.eval()
        totalLoss, count = 0.0, 0
        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)
        
        pbar = tqdm(dataLoader, desc="Validating", leave=False)
        with torch.no_grad():
            for input, target in pbar:
                non_blocking = (device.type == 'cuda')
                input = input.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                
                output = model(input)
                loss = criterion(output, target)
                output_prob = torch.sigmoid(output)
                
                outGT = torch.cat((outGT, target), 0)
                outPRED = torch.cat((outPRED, output_prob), 0)
                
                totalLoss += loss.item()
                count += 1
                avgLoss = totalLoss / count
                pbar.set_postfix(loss=f"{avgLoss:.4f}")
        
        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, classCount)
        aurocMean = np.nanmean(aurocIndividual)
        
        avgLoss = totalLoss / count if count > 0 else float("inf")
        return avgLoss, float(aurocMean)

    @staticmethod
    def computeAUROC(dataGT, dataPRED, classCount):
        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except Exception as e:
                outAUROC.append(float("nan"))
        return outAUROC

    @staticmethod
    def test(pathDirData, pathFileTest, pathModel,
             nnClassCount, trBatchSize, transCrop,
             device=None,
             model_variant='swin_large_patch4_window7_224.ms_in22k_ft_in1k'):

        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                       'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                       'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                       'Pleural_Thickening', 'Hernia', 'No Finding']

        # ---- Auto detect device
        if device is None:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                print(f"\n{'='*80}")
                print(f"GPU Detection: Found {num_gpus} GPU(s) for testing")
                for i in range(num_gpus):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"{'='*80}\n")
                device = torch.device('cuda')
            else:
                print("Using CPU for testing")
                device = torch.device('cpu')

        # Load checkpoint
        print(f"Loading model from: {pathModel}")
        ckpt = torch.load(pathModel, map_location=device)
        
        model_variant_from_ckpt = ckpt.get('model_variant', model_variant)
        print(f"Loading model architecture: {model_variant_from_ckpt}")
        
        model = SwinTransformer(nnClassCount, isTrained=False, 
                                model_variant=model_variant_from_ckpt).to(device)

        state_dict = ckpt.get('state_dict', ckpt)
        try:
            model.load_state_dict(state_dict)
        except Exception:
            if any(k.startswith('module.') for k in state_dict.keys()):
                stripped = OrderedDict((k[7:], v) if k.startswith('module.') else (k, v)
                                        for k, v in state_dict.items())
                model.load_state_dict(stripped)
            else:
                raise
        
        # Multi-GPU for testing
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using DataParallel for testing with {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        
        model.eval()
        
        epoch = ckpt.get('epoch', 'unknown')
        best_loss = ckpt.get('best_loss', 'unknown')
        best_auroc = ckpt.get('best_auroc', 'unknown')
        print(f"Model loaded successfully! (epoch={epoch}, best_loss={best_loss}, best_auroc={best_auroc})")

        # Transforms
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        transformTest = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(transCrop),
            transforms.ToTensor(),
            normalize
        ])

        # Dataset & DataLoader
        datasetTest = DatasetGenerator(pathDirData, pathFileTest, transformTest)
        use_cuda = torch.cuda.is_available()
        num_workers = 4 if use_cuda else 0
        
        dataLoaderTest = DataLoader(
            datasetTest, 
            batch_size=trBatchSize,
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=use_cuda
        )

        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)

        print(f"Testing on {len(datasetTest)} images...")
        with torch.no_grad():
            for input, target in tqdm(dataLoaderTest, desc="Testing"):
                non_blocking = use_cuda
                input = input.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                
                out_logits = model(input)
                out_prob = torch.sigmoid(out_logits)
                
                outGT = torch.cat((outGT, target), 0)
                outPRED = torch.cat((outPRED, out_prob), 0)

        # Compute AUROC
        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.nanmean(aurocIndividual)

        print("\n" + "="*60)
        print(f"AUROC mean: {aurocMean:.4f}")
        print("="*60)
        for i, name in enumerate(CLASS_NAMES[:nnClassCount]):
            if i < len(aurocIndividual):
                print(f"{name:20s}: {aurocIndividual[i]:.4f}")
            else:
                break
        print("="*60)
        
        return aurocMean, aurocIndividual
