# main.py
import os
import sys
import torch

# =========================================================================
# Thêm thư mục Models vào sys.path để import đúng trên Colab
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
models_dir = os.path.join(current_dir, 'Models')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
# =========================================================================

from TrainModel import HybridTrainer
from config import TensorCoreConfig


from checkpoint_utils import load_checkpoint_safe, extract_state_dict


def main():
    print("\n" + "=" * 80)
    print("Group Project: Hybrid CNN-ViT Multi-Label Chest X-ray Classification")
    print("Architecture: ConvNeXtV2 & SwinV2 | Residual Masking")
    print("=" * 80 + "\n")

    print("Chọn chế độ:")
    print("  1. Train mới từ đầu")
    print("  2. Train tiếp từ checkpoint cũ (Resume)")
    print("  3. Test / Đánh giá")
    mode_choice = input("Lựa chọn [1/2/3]: ").strip()

    if mode_choice == '1':
        runTrain(resume=False)
    elif mode_choice == '2':
        runTrain(resume=True)
    elif mode_choice == '3':
        runTest()
    else:
        print("❌ Lựa chọn không hợp lệ.")


# ─────────────────────────────────────────────────────────────────────────────
def runTrain(resume: bool = False):
    print("\n[MODE] Khởi tạo huấn luyện Mô hình Lai")

    resume_path = None

    # ── A. Nếu resume: load checkpoint trước để tự động điền model_size / img_size
    if resume:
        resume_path = input(
            "\n📂 Đường dẫn checkpoint cũ (.pth) [Trainedmodel/hybrid_model.pth]: "
        ).strip() or 'Trainedmodel/hybrid_model.pth'

        if os.path.isfile(resume_path):
            device_tmp = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ckpt_tmp = load_checkpoint_safe(resume_path, device_tmp)
            ckpt_model_size = ckpt_tmp.get('model_size', None)
            ckpt_img_size   = ckpt_tmp.get('img_size',   None)
            ckpt_epoch      = ckpt_tmp.get('epoch',      '?')
            ckpt_stage      = ckpt_tmp.get('stage',      '?')
            ckpt_auroc      = ckpt_tmp.get('best_auroc', 0.0)
            print(f"\n✅ Checkpoint tìm thấy:")
            print(f"   Đã train đến epoch {ckpt_epoch}, stage {ckpt_stage}, "
                  f"best AUROC = {ckpt_auroc:.4f}")
            if ckpt_model_size:
                print(f"   model_size={ckpt_model_size}, img_size={ckpt_img_size}")
        else:
            ckpt_model_size = None
            ckpt_img_size   = None
            print(f"⚠️  Không tìm thấy '{resume_path}' — sẽ bắt đầu như train mới.")
            resume_path = None

    # ── B. Chọn Backbone
    print("\n🏗️ Chọn cặp Backbone:")
    print("  1. Base-Base (ConvNeXtV2-Base + SwinV2-Base)  — VRAM 12–16 GB")
    print("  2. Large-Large (ConvNeXtV2-Large + SwinV2-Large) — VRAM > 24 GB")
    if resume and ckpt_model_size:
        default_bb = '2' if ckpt_model_size == 'large' else '1'
        bb_choice = input(f"Lựa chọn [1/2] (mặc định {default_bb} từ checkpoint): ").strip() or default_bb
    else:
        bb_choice = input("Lựa chọn [1/2]: ").strip() or "1"
    model_size = 'large' if bb_choice == '2' else 'base'

    # ── C. Chọn Kích thước ảnh
    print("\n📐 Chọn kích thước ảnh đầu vào:")
    print("  1. 256×256 (Tốc độ nhanh, tiết kiệm VRAM)")
    print("  2. 384×384 (Độ phân giải cao, khuyến nghị cho VinDr)")
    if resume and ckpt_img_size:
        default_sz = '2' if ckpt_img_size == 384 else '1'
        sz_choice = input(f"Lựa chọn [1/2] (mặc định {default_sz} từ checkpoint): ").strip() or default_sz
    else:
        sz_choice = input("Lựa chọn [1/2]: ").strip() or "2"
    img_size = 384 if sz_choice == '2' else 256

    # ── D. Phân tích phần cứng & Batch Size
    config = TensorCoreConfig.get_optimal_hybrid_config(model_size, img_size)
    TensorCoreConfig.print_config(config)

    use_optimal = input("\nSử dụng Batch Size đề xuất? [y/n]: ").strip().lower()
    if use_optimal == 'y':
        trBatchSize = config['batch_size']
    else:
        trBatchSize = int(input("Nhập Batch Size thủ công (VD: 8, 16, 32): ").strip())

    # ── E. Đường dẫn dữ liệu
    print("\n📂 Cấu hình đường dẫn dữ liệu:")
    pathDirData   = input("Thư mục chứa ảnh [Database/]: ").strip() or 'Database'
    pathFileTrain = input("File CSV Train [Dataset/train_list.csv]: ").strip() or 'Dataset/train_list.csv'
    pathFileVal   = input("File CSV Val   [Dataset/val_list.csv]: ").strip() or 'Dataset/val_list.csv'
    pathModel     = input("Đường dẫn lưu model [Trainedmodel/hybrid_model.pth]: ").strip() or 'Trainedmodel/hybrid_model.pth'

    # ── F. Preload vào RAM
    print("\n💾 Preload dữ liệu vào RAM:")
    print("  1. Không — đọc ảnh từ đĩa mỗi epoch (an toàn, ít RAM)")
    print("  2. Có    — đọc toàn bộ vào RAM 1 lần (train nhanh hơn, cần nhiều RAM)")
    preload_choice = input("Lựa chọn [1/2]: ").strip() or "1"
    preload_images = (preload_choice == '2')
    num_workers_preload = 8
    if preload_images:
        nw = input("Số threads đọc song song [8]: ").strip()
        num_workers_preload = int(nw) if nw else 8

    # ── G. Số epoch
    trMaxEpoch = int(input("\nSố Epoch tối đa cho lần chạy này [50]: ").strip() or "50")

    print("\n" + "=" * 80)
    if resume and resume_path:
        print("🔄 TIẾP TỤC HUẤN LUYỆN TỪ CHECKPOINT")
    else:
        print("🚀 BẮT ĐẦU HUẤN LUYỆN MỚI — 2 GIAI ĐOẠN")
    print("=" * 80)

    HybridTrainer.train(
        pathDirData=pathDirData,
        pathFileTrain=pathFileTrain,
        pathFileVal=pathFileVal,
        model_size=model_size,
        img_size=img_size,
        trBatchSize=trBatchSize,
        trMaxEpoch=trMaxEpoch,
        pathModel=pathModel,
        preload_images=preload_images,
        num_workers_preload=num_workers_preload,
        resume_path=resume_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
def runTest():
    print("\n[MODE] Kiểm thử / Đánh giá Mô hình Lai")

    from read_data import DatasetGenerator, FastDataLoader
    from Model import HybridCNNViTModel
    import numpy as np
    import torchvision.transforms as transforms
    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    CLASS_NAMES = HybridTrainer.CLASS_NAMES
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Đường dẫn
    pathModel   = input("Đường dẫn file model (.pth) [Trainedmodel/hybrid_model.pth]: ").strip() or 'Trainedmodel/hybrid_model.pth'
    pathDirData = input("Thư mục chứa ảnh [Database/]: ").strip() or 'Database'
    pathFileTest = input("File CSV Test [Dataset/test_list.csv]: ").strip() or 'Dataset/test_list.csv'
    save_dir    = input("Thư mục lưu kết quả [Results/]: ").strip() or 'Results'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nDevice: {device}")

    # ── Load checkpoint
    ckpt = load_checkpoint_safe(pathModel, device)
    ckpt_model_size = ckpt.get('model_size', 'base')
    ckpt_img_size   = ckpt.get('img_size', 384)
    print(f"✅ Checkpoint: epoch={ckpt.get('epoch','?')}, stage={ckpt.get('stage','?')}, "
          f"best_auroc={ckpt.get('best_auroc',0.0):.4f}")
    print(f"   model_size={ckpt_model_size}, img_size={ckpt_img_size}")

    # ── Xác nhận hoặc ghi đè model_size / img_size
    print("\n🏗️ Chọn kích thước model:")
    print(f"  1. Base  (mặc định từ checkpoint: {'✓' if ckpt_model_size=='base' else ' '})")
    print(f"  2. Large (mặc định từ checkpoint: {'✓' if ckpt_model_size=='large' else ' '})")
    default_bb = '2' if ckpt_model_size == 'large' else '1'
    bb_in = input(f"Lựa chọn [1/2] (Enter = {default_bb}): ").strip() or default_bb
    model_size = 'large' if bb_in == '2' else 'base'

    print("\n📐 Chọn kích thước ảnh:")
    print(f"  1. 256×256   (mặc định từ checkpoint: {'✓' if ckpt_img_size==256 else ' '})")
    print(f"  2. 384×384   (mặc định từ checkpoint: {'✓' if ckpt_img_size==384 else ' '})")
    default_sz = '2' if ckpt_img_size == 384 else '1'
    sz_in = input(f"Lựa chọn [1/2] (Enter = {default_sz}): ").strip() or default_sz
    img_size = 384 if sz_in == '2' else 256

    # ── Build & load model
    print(f"\n🏗️ Đang load model ({model_size.upper()}, {img_size}px)...")
    model = HybridCNNViTModel(num_classes=15, model_size=model_size, img_size=img_size).to(device)
    state_dict = extract_state_dict(ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Dataset test
    transformTest = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
    ])
    datasetTest   = DatasetGenerator(pathDirData, pathFileTest, transformTest)
    dataLoaderTest = FastDataLoader.create_dataloader(
        datasetTest, batch_size=16, shuffle=False, num_workers=4
    )

    # ── Inference với TTA (Test-Time Augmentation)
    # TTA: chạy forward 2 lần (gốc + flip ngang), lấy trung bình logits
    # → ổn định hơn, AUROC +0.3-0.5% so với inference đơn
    print("\n🔍 Đang chạy inference + TTA...")
    allPreds, allTargets = [], []
    
    # Detect AMP dtype
    use_amp = device.type == 'cuda'
    if use_amp:
        compute_cap = torch.cuda.get_device_capability(0)
        amp_dtype = torch.bfloat16 if compute_cap >= (8, 0) else torch.float16
    else:
        amp_dtype = torch.float32
    
    with torch.no_grad():
        from tqdm import tqdm
        for inputs, targets, _, _ in tqdm(dataLoaderTest, desc="Testing+TTA", ncols=100):
            inputs = inputs.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                logits, _ = model(inputs)
                # TTA: lật ngang ảnh và lấy trung bình logits
                logits_flip, _ = model(torch.flip(inputs, dims=[3]))
            logits = ((logits.float() + logits_flip.float()) / 2.0)
            allPreds.append(torch.sigmoid(logits).cpu())
            allTargets.append(targets.cpu())

    allPreds   = torch.cat(allPreds,   dim=0).numpy()
    allTargets = torch.cat(allTargets, dim=0).numpy()
    allBinary  = (allPreds >= 0.5).astype(int)

    # ── Per-class AUROC
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ KIỂM THỬ")
    print("=" * 60)
    aurocs = []
    for i, name in enumerate(CLASS_NAMES):
        try:
            auc = roc_auc_score(allTargets[:, i], allPreds[:, i])
            aurocs.append(auc)
            print(f"  {name:<22}: AUROC = {auc:.4f}")
        except Exception:
            aurocs.append(float('nan'))
            print(f"  {name:<22}: AUROC = N/A")
    mean_auroc = float(__import__('numpy').nanmean(aurocs))
    print(f"\n  {'Mean AUROC':<22}: {mean_auroc:.4f}")

    # Classification report
    print("\n📋 Classification Report (threshold=0.5):")
    print(classification_report(
        allTargets.flatten(), allBinary.flatten(),
        target_names=['Negative', 'Positive'], digits=4
    ))

    # ── Biểu đồ AUROC
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(CLASS_NAMES, aurocs, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axhline(y=mean_auroc, color='tomato', linestyle='--', linewidth=1.5,
               label=f'Mean AUROC = {mean_auroc:.4f}')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUROC')
    ax.set_title('Per-class AUROC trên Test Set')
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha='right', fontsize=9)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    auroc_path = os.path.join(save_dir, 'test_auroc.png')
    plt.savefig(auroc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Biểu đồ AUROC: {auroc_path}")

    # ── Confusion matrix (flattened)
    cm = confusion_matrix(allTargets.flatten(), allBinary.flatten())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticklabels(['True 0', 'True 1'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title('Confusion Matrix (tổng hợp)')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'test_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion Matrix: {cm_path}")
    print(f"\n✅ Hoàn tất. Kết quả lưu tại: {save_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user")
    except Exception as e:
        import traceback
        print(f"\n\n❌ Error: {e}")
        traceback.print_exc()
