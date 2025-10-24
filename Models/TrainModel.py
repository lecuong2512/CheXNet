import os
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
    from Models.Model import DenseNet121, ConvNeXtV2Large
    from Models.read_data import DatasetGenerator
except ImportError:
    from Model import DenseNet121, ConvNeXtV2Large
    from read_data import DatasetGenerator


class ChexnetTrainer:

    @staticmethod
    def train(pathDirData, pathFileTrain, pathFileVal,
              nnIsTrained, nnClassCount,
              trBatchSize, trMaxEpoch,
              transCrop, pathModel='CheXNet/Trainedmodel/chexnetmodel.pth',
              checkpoint=None, model_type='convnextv2_large'):
        """
        Train model with support for multiple architectures.
        
        Args:
            model_type: 'densenet121' or 'convnextv2_large'
        """

        # ---- Device & Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Select model architecture
        if model_type.lower() == 'convnextv2_large':
            model = ConvNeXtV2Large(nnClassCount, nnIsTrained).to(device)
            print(f"Using ConvNeXtV2-Large architecture")
        else:
            model = DenseNet121(nnClassCount, nnIsTrained).to(device)
            print(f"Using DenseNet121 architecture")
            
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).to(device)

        # ---- Data transforms
        # ConvNeXtV2 sử dụng ImageNet normalization giống DenseNet
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        transformTrain = transforms.Compose([
            transforms.RandomResizedCrop(transCrop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transformVal = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(transCrop),
            transforms.ToTensor(),
            normalize
        ])

        # ---- Datasets & Loaders
        datasetTrain = DatasetGenerator(pathDirData, pathFileTrain, transformTrain)
        datasetVal   = DatasetGenerator(pathDirData, pathFileVal, transformVal)

        use_cuda = torch.cuda.is_available()
        dataLoaderTrain = DataLoader(datasetTrain, batch_size=trBatchSize,
                                     shuffle=True, num_workers=2 if use_cuda else 0, pin_memory=use_cuda)
        dataLoaderVal   = DataLoader(datasetVal, batch_size=trBatchSize,
                                     shuffle=False, num_workers=2 if use_cuda else 0, pin_memory=use_cuda)

        # ---- Optimizer & Scheduler
        # ConvNeXtV2 thường dùng learning rate thấp hơn
        lr = 5e-5 if model_type.lower() == 'convnextv2_large' else 1e-4
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

        # ---- Loss
        criterion = nn.BCELoss()

        # ---- Load checkpoint nếu có
        bestLoss = float("inf")
        if checkpoint:
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
            optimizer.load_state_dict(ckpt['optimizer'])
            bestLoss = ckpt.get('best_loss', bestLoss)

        # ---- Training loop
        print("\n" + "="*80)
        print(f"Starting training from epoch 1 to {trMaxEpoch}")
        print("="*80 + "\n")
        for epoch in range(trMaxEpoch):
            trainLoss = ChexnetTrainer.epochTrain(model, dataLoaderTrain, optimizer, criterion, device)
            valLoss = ChexnetTrainer.epochVal(model, dataLoaderVal, criterion, device)

            scheduler.step(valLoss)

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
                print(f"[{epoch+1}] [save] train_loss={trainLoss:.4f} val_loss={valLoss:.4f} -> {pathModel}")
            else:
                print(f"[{epoch+1}] [----] train_loss={trainLoss:.4f} val_loss={valLoss:.4f}")

    @staticmethod
    def epochTrain(model, dataLoader, optimizer, criterion, device):
        model.train()
        totalLoss, count = 0.0, 0
        for input, target in tqdm(dataLoader, desc="Training", unit="batch"):
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
        return totalLoss / count if count > 0 else float("inf")

    @staticmethod
    def epochVal(model, dataLoader, criterion, device):
        model.eval()
        totalLoss, count = 0.0, 0
        with torch.no_grad():
            for input, target in tqdm(dataLoader, desc="Validation", unit="batch"):
                non_blocking = torch.cuda.is_available()
                input = input.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                output = model(input)
                loss = criterion(output, target)
                totalLoss += loss.item()
                count += 1
        return totalLoss / count if count > 0 else float("inf")

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
        """
        Test model with support for multiple architectures.
        
        Args:
            model_type: 'densenet121' or 'convnextv2_large'. If None, will try to read from checkpoint.
        """

        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                       'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                       'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                       'Pleural_Thickening', 'Hernia', 'No Finding']

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint to determine model type if not specified
        ckpt = torch.load(pathModel, map_location=device)
        if model_type is None:
            model_type = ckpt.get('model_type', 'convnextv2_large')
        
        # Create model based on type
        if model_type.lower() == 'convnextv2_large':
            model = ConvNeXtV2Large(nnClassCount, isTrained=False).to(device)
            print(f"Testing with ConvNeXtV2-Large architecture")
        else:
            model = DenseNet121(nnClassCount, isTrained=False).to(device)
            print(f"Testing with DenseNet121 architecture")
        
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

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        transformTest = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(transCrop),
            transforms.ToTensor(),
            normalize
        ])

        datasetTest = DatasetGenerator(pathDirData, pathFileTest, transformTest)
        use_cuda = torch.cuda.is_available()
        dataLoaderTest = DataLoader(datasetTest, batch_size=trBatchSize,
                                    shuffle=False, num_workers=2 if use_cuda else 0, pin_memory=use_cuda)

        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)

        with torch.no_grad():
            for input, target in dataLoaderTest:
                non_blocking = torch.cuda.is_available()
                input = input.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                out = model(input)
                outGT = torch.cat((outGT, target), 0)
                outPRED = torch.cat((outPRED, out), 0)

        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.nanmean(aurocIndividual)

        print(f"\nAUROC mean: {aurocMean:.4f}")
        for i, name in enumerate(CLASS_NAMES[:nnClassCount]):
            print(f"{name:20s}: {aurocIndividual[i]:.4f}")
