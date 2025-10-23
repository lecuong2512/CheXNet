"""
Ví dụ cách sử dụng ConvNeXtV2-Large cho training và testing
"""

from Models import ChexnetTrainer

# ===== VÍ DỤ 1: TRAINING VỚI CONVNEXTV2-LARGE =====
def train_convnextv2():
    ChexnetTrainer.train(
        pathDirData='./Database',
        pathFileTrain='./Dataset/train_list.txt',
        pathFileVal='./Dataset/val_list.txt',
        nnIsTrained=True,  # Sử dụng pretrained weights
        nnClassCount=15,   # Số lượng class
        trBatchSize=16,    # Batch size (ConvNeXtV2-Large cần memory nhiều hơn)
        trMaxEpoch=100,
        transCrop=224,
        pathModel='CheXNet/Trainedmodel/chexnetmodel.pth',
        checkpoint=None,   # Hoặc đường dẫn đến checkpoint để resume
        model_type='convnextv2_large'  # CHỈ ĐỊNH SỬ DỤNG CONVNEXTV2
    )

# ===== VÍ DỤ 2: TRAINING VỚI DENSENET121 (GIỮ NGUYÊN) =====
def train_densenet():
    ChexnetTrainer.train(
        pathDirData='./Database',
        pathFileTrain='./Dataset/train_list.txt',
        pathFileVal='./Dataset/val_list.txt',
        nnIsTrained=True,
        nnClassCount=15,
        trBatchSize=32,    # DenseNet có thể dùng batch size lớn hơn
        trMaxEpoch=100,
        transCrop=224,
        pathModel='Trainedmodel/chexnetmodel.pth',
        checkpoint=None,
        model_type='densenet121'  # CHỈ ĐỊNH SỬ DỤNG DENSENET
    )

# ===== VÍ DỤ 3: TESTING VỚI CONVNEXTV2-LARGE =====
def test_convnextv2():
    ChexnetTrainer.test(
        pathDirData='./Database',
        pathFileTest='./Dataset/test_list.txt',
        pathModel='./Trainedmodel/chexnetmodel.pth',
        nnClassCount=15,
        trBatchSize=16,
        transCrop=224,
        device=None,  # Tự động detect CUDA
        model_type='convnextv2_large'  # CHỈ ĐỊNH LOẠI MODEL
    )

# ===== VÍ DỤ 4: TESTING VỚI DENSENET121 =====
def test_densenet():
    ChexnetTrainer.test(
        pathDirData='./Database',
        pathFileTest='./Dataset/test_list.txt',
        pathModel='./Trainedmodel/chexnetmodel.pth',
        nnClassCount=15,
        trBatchSize=32,
        transCrop=224,
        device=None,
        model_type='densenet121'  # CHỈ ĐỊNH LOẠI MODEL
    )

# ===== VÍ DỤ 5: AUTO-DETECT MODEL TYPE TỪ CHECKPOINT =====
def test_auto_detect():
    # Nếu không chỉ định model_type, sẽ tự động đọc từ checkpoint
    ChexnetTrainer.test(
        pathDirData='./Database',
        pathFileTest='./Dataset/test_list.txt',
        pathModel='./Trainedmodel/chexnetmodel.pth',
        nnClassCount=15,
        trBatchSize=16,
        transCrop=224
        # Không cần model_type, tự động detect
    )


if __name__ == '__main__':
    # Chạy training với ConvNeXtV2-Large
    print("=== Training ConvNeXtV2-Large ===")
    train_convnextv2()
    
    # Hoặc chạy testing
    # print("=== Testing ConvNeXtV2-Large ===")
    # test_convnextv2()