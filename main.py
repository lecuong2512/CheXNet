import os
from Models.TrainModel import ChexnetTrainer

#-------------------------------------------------------------------------------- 

def main():
    runTrain()
    #runTest()
  
#--------------------------------------------------------------------------------   

def runTrain():
    # ---- Path to the directory with images
    pathDirData = '/kaggle/input/data'
    
    # ---- Paths to the dataset files
    pathFileTrain = '/kaggle/working/train_list.txt'
    pathFileVal   = '/kaggle/working/val_list.txt'
    pathFileTest  = '/kaggle/working/test_list.txt'
    
    # ---- Parameters
    nnIsTrained   = True
    nnClassCount  = 14
    trBatchSize   = 16
    trMaxEpoch    = 100
    imgtransCrop  = 224
    
    # ---- Model save path (cố định tên chexnetmodel.pth)
    pathModel = 'Trainedmodel/chexnetmodel.pth'
    
    print('Training DenseNet121 ...')
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal,
                         nnIsTrained, nnClassCount,
                         trBatchSize, trMaxEpoch,
                         imgtransCrop, pathModel,
                         checkpoint=None)
    
    print('Testing the trained model ...')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel,
                        nnClassCount, trBatchSize, imgtransCrop)

#-------------------------------------------------------------------------------- 

def runTest():
    pathDirData   = '/kaggle/input/data'
    pathFileTest  = '/kaggle/working/test_list.txt'
    nnClassCount  = 14
    trBatchSize   = 16
    imgtransCrop  = 224
    
    # ---- model đã train sẵn
    pathModel = '/kaggle/working/chexnetmodel.pth'
    
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel,
                        nnClassCount, trBatchSize, imgtransCrop)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()