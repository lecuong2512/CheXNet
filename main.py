# main.py
import os
import sys
import torch

# =========================================================================
# VÁ LỖI MODULE: ÉP PYTHON NHẬN DIỆN THƯ MỤC 'Models'
# Đoạn code này giúp giải quyết lỗi ModuleNotFoundError trên Colab
# =========================================================================
# Lấy đường dẫn thư mục hiện tại (ví dụ: /content/CheXNet)
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
models_dir = os.path.join(current_dir, 'Models')

# Thêm thư mục Models vào danh sách tìm kiếm của Python
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
# =========================================================================

# Bây giờ có thể import bình thường như thể chúng nằm cùng cấp
from TrainModel import HybridTrainer
from config import TensorCoreConfig

def main():
    print("\n" + "="*80)
    print("Group Project: Hybrid CNN-ViT Multi-Label Chest X-ray Classification")
    print("Architecture: ConvNeXtV2 & SwinV2 | Residual Masking")
    print("="*80 + "\n")
    
    mode = input("Chọn chế độ [train/test]: ").strip().lower()
    
    if mode == 'train':
        runTrain()
    elif mode == 'test':
        runTest()
    else:
        print("❌ Lựa chọn không hợp lệ. Vui lòng chọn 'train' hoặc 'test'.")

def runTrain():
    print("\n[MODE] Khởi tạo huấn luyện Mô hình Lai")
    
    # 1. Chọn Backbone
    print("\n🏗️ Chọn cặp Backbone:")
    print("  1. Base-Base (ConvNeXtV2-Base + SwinV2-Base) - Phù hợp VRAM 12-16GB")
    print("  2. Large-Large (ConvNeXtV2-Large + SwinV2-Large) - Yêu cầu VRAM > 24GB")
    bb_choice = input("Lựa chọn [1/2]: ").strip() or "1"
    model_size = 'large' if bb_choice == '2' else 'base'
    
    # 2. Chọn Kích thước ảnh
    print("\n📐 Chọn kích thước ảnh đầu vào:")
    print("  1. 256x256 (Tốc độ nhanh, tiết kiệm VRAM)")
    print("  2. 384x384 (Độ phân giải cao, khuyến nghị cho VinDr)")
    size_choice = input("Lựa chọn [1/2]: ").strip() or "2"
    img_size = 384 if size_choice == '2' else 256
    
    # 3. Phân tích phần cứng và Batch Size
    config = TensorCoreConfig.get_optimal_hybrid_config(model_size, img_size)
    TensorCoreConfig.print_config(config)
    
    use_optimal = input("\nSử dụng Batch Size đề xuất? [y/n]: ").strip().lower()
    if use_optimal == 'y':
        trBatchSize = config['batch_size']
    else:
        trBatchSize = int(input("Nhập Batch Size thủ công (VD: 8, 16, 32): ").strip())

    # 4. Cấu hình Đường dẫn
    print("\n📂 Cấu hình đường dẫn dữ liệu:")
    pathDirData = input("Thư mục chứa ảnh [Database/]: ").strip() or 'Database'
    pathFileTrain = input("File CSV Train [Dataset/train_data.csv]: ").strip() or 'Dataset/train_data.csv'
    pathFileVal = input("File CSV Val [Dataset/val_data.csv]: ").strip() or 'Dataset/val_data.csv'
    pathModel = input("Đường dẫn lưu model [Trainedmodel/hybrid_model.pth]: ").strip() or 'Trainedmodel/hybrid_model.pth'
    
    trMaxEpoch = int(input("\nSố lượng Epochs tối đa [50]: ").strip() or "50")
    
    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN 2 GIAI ĐOẠN")
    print("="*80)
    
    HybridTrainer.train(
        pathDirData=pathDirData, 
        pathFileTrain=pathFileTrain, 
        pathFileVal=pathFileVal,
        model_size=model_size,
        img_size=img_size,
        trBatchSize=trBatchSize, 
        trMaxEpoch=trMaxEpoch,
        pathModel=pathModel
    )

def runTest():
    print("\n[MODE] Kiểm thử Mô hình Lai")

    from TrainModel import HybridTrainer
    from read_data import DatasetGenerator, FastDataLoader
    from Model import HybridCNNViTModel
    import torch
    import numpy as np
    import torchvision.transforms as transforms
    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    CLASS_NAMES = HybridTrainer.CLASS_NAMES

    # Cấu hình
    pathModel = input("Đường dẫn file model (.pth) [Trainedmodel/hybrid_model.pth]: ").strip() or 'Trainedmodel/hybrid_model.pth'
    pathDirData = input("Thư mục chứa ảnh [Database/]: ").strip() or 'Database'
    pathFileTest = input("File CSV Test [Dataset/test_data.csv]: ").strip() or 'Dataset/test_data.csv'
    save_dir = input("Thư mục lưu kết quả [Results/]: ").strip() or 'Results'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load checkpoint để lấy model_size và img_size
    # weights_only=False cần thiết vì checkpoint chứa numpy scalar (an toàn vì file do chính mình train)
    try:
        import torch.serialization, numpy as _np
        torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
        ckpt = torch.load(pathModel, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(pathModel, map_location=device, weights_only=False)
    model_size = ckpt.get('model_size', 'base')
    img_size = ckpt.get('img_size', 384)

    model_size_input = input(f"Model size (base/large) [{model_size}]: ").strip() or model_size
    img_size_input = input(f"Image size (256/384) [{img_size}]: ").strip()
    if img_size_input:
        img_size = int(img_size_input)

    # Build & load model
    print(f"\n🏗️ Loading model ({model_size_input.upper()}, {img_size}px)...")
    model = HybridCNNViTModel(num_classes=15, model_size=model_size_input, img_size=img_size).to(device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        state_dict = OrderedDict((k[7:], v) if k.startswith('module.') else (k, v) for k, v in state_dict.items())
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Loaded model (best AUROC lúc train: {ckpt.get('best_auroc', 'N/A')})")

    # Dataset
    transformTest = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size)
    ])
    datasetTest = DatasetGenerator(pathDirData, pathFileTest, transformTest)
    dataLoaderTest = FastDataLoader.create_dataloader(datasetTest, batch_size=16, shuffle=False, num_workers=4)

    # Inference
    print("\n🔍 Đang chạy inference...")
    allPreds, allTargets = [], []
    with torch.no_grad():
        from tqdm import tqdm
        for inputs, targets, _, _ in tqdm(dataLoaderTest, desc="Testing", ncols=100):
            inputs = inputs.to(device)
            logits, _ = model(inputs)
            allPreds.append(torch.sigmoid(logits).cpu())
            allTargets.append(targets.cpu())

    allPreds = torch.cat(allPreds, dim=0).numpy()
    allTargets = torch.cat(allTargets, dim=0).numpy()
    allBinary = (allPreds >= 0.5).astype(int)

    # Per-class AUROC
    print("\n" + "="*60)
    print("📊 KẾT QUẢ KIỂM THỬ")
    print("="*60)
    aurocs = []
    for i, name in enumerate(CLASS_NAMES):
        try:
            auc = roc_auc_score(allTargets[:, i], allPreds[:, i])
            aurocs.append(auc)
            print(f"  {name:<22}: AUROC = {auc:.4f}")
        except Exception:
            aurocs.append(float('nan'))
            print(f"  {name:<22}: AUROC = N/A (class không có trong test set)")
    mean_auroc = np.nanmean(aurocs)
    print(f"\n  {'Mean AUROC':<22}: {mean_auroc:.4f}")

    # Classification report
    print("\n📋 Classification Report (threshold=0.5):")
    print(classification_report(
        allTargets.flatten(),
        allBinary.flatten(),
        target_names=['Negative', 'Positive'],
        digits=4
    ))

    # Plot per-class AUROC bar chart
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(CLASS_NAMES, aurocs, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axhline(y=mean_auroc, color='tomato', linestyle='--', linewidth=1.5, label=f'Mean AUROC = {mean_auroc:.4f}')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUROC')
    ax.set_title('Per-class AUROC trên Test Set')
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    auroc_path = os.path.join(save_dir, 'test_auroc.png')
    plt.savefig(auroc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Đã lưu biểu đồ AUROC: {auroc_path}")

    # Confusion matrix (flattened multi-label)
    cm = confusion_matrix(allTargets.flatten(), allBinary.flatten())
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1']); ax.set_yticklabels(['True 0', 'True 1'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title('Confusion Matrix (tổng hợp)')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'test_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu Confusion Matrix: {cm_path}")
    print(f"\n✅ Hoàn tất kiểm thử. Kết quả lưu tại: {save_dir}/")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
