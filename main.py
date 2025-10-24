import os
# Đảm bảo tên file trainer là TrainModel.py
from Models.TrainModel import ChexnetTrainer 

#-------------------------------------------------------------------------------- 

def main():
    runTrain()
    #runTest()
 
#--------------------------------------------------------------------------------  

def runTrain():
    # ---- Path to the directory with images
    pathDirData = './Database'
    
    # ---- Paths to the dataset files
    pathFileTrain = './Dataset/train_list.txt'
    pathFileVal   = './Dataset/val_list.txt'
    pathFileTest  = './Dataset/test_list.txt'
    
    # ---- Parameters
    nnIsTrained   = True
    nnClassCount  = 15
    trBatchSize   = 32  # <<< XEM LƯU Ý BÊN DƯỚI!
    trMaxEpoch    = 10
    imgtransCrop  = 224 # <-- Giữ nguyên 224x224
    
    # ---- THAY ĐỔI 1: Sửa thành tên SwinV1-Large (khớp với 224x224)
    # Tên cũ (sai): 'swinv2_large_patch4_window7_224_22kto1k'
    model_variant = 'swin_large_patch4_window7_224.ms_in22k_ft_in1k'
    
    # ---- Model save path (Tự động cập nhật tên)
    pathModel = './Trainedmodel/chexnetmodel.pth'
    
    print(f'Training {model_variant}...')
    
    # ---- THAY ĐỔI 2: Truyền 'model_variant' vào hàm train (giữ nguyên)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal,
                         nnIsTrained, nnClassCount,
                         trBatchSize, trMaxEpoch,
                         imgtransCrop, pathModel,
                         checkpoint=None,
                         model_variant=model_variant) 
    
    print('Testing the trained model ...')
    
    # ---- THAY ĐỔI 3: Truyền 'model_variant' vào hàm test (giữ nguyên)
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel,
                        nnClassCount, trBatchSize, imgtransCrop,
                        model_variant=model_variant)

#-------------------------------------------------------------------------------- 

def runTest():
    pathDirData   = './Database'
    pathFileTest  = './Dataset/test_list.txt'
    nnClassCount  = 15
    trBatchSize   = 32 # Giảm nếu bị OOM khi test
    imgtransCrop  = 224
    
    # ---- THAY ĐỔI 4: Chỉ định model_variant và pathModel
    model_variant = 'swin_large_patch4_window7_224.ms_in22k_ft_in1k'
    pathModel = './Trainedmodel/chexnetmodel.pth'
    
    # ---- THAY ĐỔI 5: Truyền 'model_variant' vào hàm test
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel,
                        nnClassCount, trBatchSize, imgtransCrop,
                        model_variant=model_variant)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()