# main_finetune.py
# ============================================================================
# Fine-tune chuyên sâu từ checkpoint đã train (Epoch 12, AUROC 0.8651)
# Tách biệt hoàn toàn với main.py và TrainModel.py gốc
# ============================================================================
import os
import sys

# Fix tqdm trong Colab
if not sys.stderr.isatty():
    class DummyTTY(object):
        def __init__(self, stream):
            self.stream = stream
        def __getattr__(self, attr):
            return getattr(self.stream, attr)
        def isatty(self):
            return True
    sys.stderr = DummyTTY(sys.stderr)
    try:
        from tqdm import tqdm
        from functools import partialmethod
        tqdm.__init__ = partialmethod(tqdm.__init__, position=0)
    except ImportError:
        pass

if "HF_HUB_DISABLE_PROGRESS_BARS" in os.environ:
    del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]

# ============================================================================
# Thêm thư mục gốc (CheXNet/) và Models/ vào sys.path
# ============================================================================
finetune_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.dirname(finetune_dir)
models_dir = os.path.join(project_root, 'Models')
for p in [project_root, models_dir, finetune_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from config import TensorCoreConfig
from checkpoint_utils import load_checkpoint_safe
from TrainModel_finetune import FinetuneTrainer


def main():
    print("\n" + "=" * 80)
    print("🔬 FINE-TUNE CHUYÊN SÂU — Hybrid CNN-ViT")
    print("   Từ checkpoint đã train, áp dụng: Hard Drop LR + ASL Gamma Tweak")
    print("   + Color Augmentation + Tăng Weight Decay")
    print("=" * 80 + "\n")

    # ── 1. Đường dẫn checkpoint nguồn ──
    checkpoint_path = input(
        "📂 Đường dẫn checkpoint nguồn (.pth) [Trainedmodel/hybrid_model.pth]: "
    ).strip() or 'Trainedmodel/hybrid_model.pth'

    if not os.path.isfile(checkpoint_path):
        # Thử tìm trong thư mục gốc
        alt_path = os.path.join(project_root, checkpoint_path)
        if os.path.isfile(alt_path):
            checkpoint_path = alt_path
        else:
            print(f"❌ Không tìm thấy checkpoint: {checkpoint_path}")
            return

    device_tmp = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_checkpoint_safe(checkpoint_path, device_tmp)
    ckpt_epoch = ckpt.get('epoch', '?')
    ckpt_stage = ckpt.get('stage', '?')
    ckpt_auroc = ckpt.get('best_auroc', 0.0)
    ckpt_model_size = ckpt.get('model_size', 'large')
    ckpt_img_size = ckpt.get('img_size', 384)

    print(f"\n✅ Checkpoint: epoch={ckpt_epoch}, stage={ckpt_stage}, AUROC={ckpt_auroc:.4f}")
    print(f"   model_size={ckpt_model_size}, img_size={ckpt_img_size}")

    # ── 2. Khóa cứng model_size và img_size từ checkpoint ──
    model_size = ckpt_model_size
    img_size = ckpt_img_size
    print(f"\n🔒 Khóa cứng: model_size={model_size}, img_size={img_size}")

    # ── 3. Phân tích phần cứng & Batch Size ──
    config = TensorCoreConfig.get_optimal_hybrid_config(model_size, img_size)
    TensorCoreConfig.print_config(config)

    use_optimal = input("\nSử dụng Batch Size đề xuất? [y/n]: ").strip().lower()
    if use_optimal == 'y':
        trBatchSize = config['batch_size']
    else:
        trBatchSize = int(input("Nhập Batch Size thủ công (VD: 8, 16, 32): ").strip())

    # ── 4. Đường dẫn dữ liệu ──
    print("\n📂 Cấu hình đường dẫn dữ liệu:")
    pathDirData = input("Thư mục chứa ảnh [Database/]: ").strip() or 'Database'
    pathFileTrain = input("File CSV Train [Dataset/train_list.csv]: ").strip() or 'Dataset/train_list.csv'
    pathFileVal = input("File CSV Val   [Dataset/val_list.csv]: ").strip() or 'Dataset/val_list.csv'

    # Đường dẫn output TÁCH BIỆT
    default_output = os.path.join(os.path.dirname(checkpoint_path), 'hybrid_model_finetuned.pth')
    pathModel = input(f"Đường dẫn lưu model fine-tuned [{default_output}]: ").strip() or default_output

    # ── 5. Preload vào RAM ──
    print("\n💾 Preload dữ liệu vào RAM:")
    print("  1. Không — đọc ảnh từ đĩa mỗi epoch (an toàn, ít RAM)")
    print("  2. Có    — đọc toàn bộ vào RAM 1 lần (train nhanh hơn, cần nhiều RAM)")
    preload_choice = input("Lựa chọn [1/2]: ").strip() or "1"
    preload_images = (preload_choice == '2')
    num_workers_preload = 8
    if preload_images:
        nw = input("Số threads đọc song song [8]: ").strip()
        num_workers_preload = int(nw) if nw else 8

    # ── 6. Số epoch (mặc định 10, đủ cho fine-tune LR thấp) ──
    trMaxEpoch = int(input("\nSố Epoch tối đa [10]: ").strip() or "10")

    # ── 7. Num workers ──
    nw_input = input("Số luồng đọc ảnh (num_workers) [4]: ").strip()
    num_workers_train = int(nw_input) if nw_input else 4

    # ── 8. Hiển thị cấu hình Fine-tune ──
    print("\n" + "=" * 80)
    print("🔬 CẤU HÌNH FINE-TUNE")
    print("=" * 80)
    print(f"  Checkpoint nguồn : {checkpoint_path} (epoch {ckpt_epoch}, AUROC {ckpt_auroc:.4f})")
    print(f"  Model output     : {pathModel}")
    print(f"  Model            : {model_size.upper()}, {img_size}px")
    print(f"  Batch Size       : {trBatchSize}")
    print(f"  Max Epochs       : {trMaxEpoch}")
    print(f"  ── Thay đổi so với train gốc ──")
    print(f"  LR backbone      : 1e-4  →  3e-5   (giảm 3.3x)")
    print(f"  LR head          : 5e-4  →  1e-4   (giảm 5x)")
    print(f"  Weight decay     : 0.05  →  0.08   (tăng regularization)")
    print(f"  ASL gamma_neg    : 4     →  2      (nới lỏng, cải thiện AUROC)")
    print(f"  EMA decay        : 0.999 →  0.9995 (ổn định hơn)")
    print(f"  UW clamp         : [-4,4]→  [-3,3] (cân bằng ASL/Dice)")
    print(f"  Color Augment    : ❌    →  ✅     (chống overfit)")
    print(f"  Checkpoint save  : ~4.2GB→  ~1.2GB (bỏ optimizer/scheduler)")
    print("=" * 80)

    confirm = input("\n🚀 Bắt đầu Fine-tune? [y/n]: ").strip().lower()
    if confirm != 'y':
        print("❌ Đã hủy.")
        return

    # ── 9. Chạy Fine-tune ──
    FinetuneTrainer.train(
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
        checkpoint_path=checkpoint_path,
        num_workers_train=num_workers_train,
    )


if __name__ == '__main__':
    main()
