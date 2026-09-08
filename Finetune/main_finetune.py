# main_finetune.py
# ============================================================================
# SOTA Fine-tune v2 — 3-Phase Strategy Entry Point
#
# Pha 1: Freeze backbone + Warmup head
# Pha 2: Unfreeze + MixUp + R-Drop
# Pha 3: SWA + Snapshot Ensemble
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
from TrainModel_finetune import SOTAFinetuneTrainer


def main():
    print("\n" + "=" * 80)
    print("🔬 SOTA FINE-TUNE v2 — 3-Phase Strategy")
    print("   Pha 1: Freeze Backbone + Warmup Head")
    print("   Pha 2: Unfreeze + MixUp + R-Drop")
    print("   Pha 3: SWA + Snapshot Ensemble")
    print("=" * 80 + "\n")

    # ── 1. Đường dẫn checkpoint nguồn ──
    checkpoint_path = input(
        "📂 Đường dẫn checkpoint nguồn (.pth) [Trainedmodel/hybrid_model.pth]: "
    ).strip() or 'Trainedmodel/hybrid_model.pth'

    if not os.path.isfile(checkpoint_path):
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
    del ckpt  # Free memory

    print(f"\n✅ Checkpoint: epoch={ckpt_epoch}, stage={ckpt_stage}, AUROC={ckpt_auroc:.4f}")
    print(f"   model_size={ckpt_model_size}, img_size={ckpt_img_size}")

    # ── 2. Khóa cứng model_size và img_size ──
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

    default_output = os.path.join(os.path.dirname(checkpoint_path), 'hybrid_model_sota.pth')
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

    # ── 6. Số epoch tối đa ──
    trMaxEpoch = int(input("\nSố Epoch tối đa [12]: ").strip() or "12")

    # ── 7. Num workers ──
    nw_input = input("Số luồng đọc ảnh (num_workers) [4]: ").strip()
    num_workers_train = int(nw_input) if nw_input else 4

    # ── 8. SOTA Fine-tune parameters ──
    print("\n🔧 Cấu hình SOTA Fine-tune (Enter = giữ mặc định):")
    freeze_input = input("  Số epoch freeze backbone [3]: ").strip()
    freeze_epochs = int(freeze_input) if freeze_input else 3

    swa_input = input("  Epoch bắt đầu SWA [9]: ").strip()
    swa_start = int(swa_input) if swa_input else 9

    mixup_input = input("  MixUp alpha (0=tắt) [0.2]: ").strip()
    mixup_alpha = float(mixup_input) if mixup_input else 0.2

    rdrop_input = input("  R-Drop alpha (0=tắt) [0.5]: ").strip()
    rdrop_alpha = float(rdrop_input) if rdrop_input else 0.5

    ls_input = input("  Label Smoothing (0=tắt) [0.02]: ").strip()
    label_smoothing = float(ls_input) if ls_input else 0.02

    snap_input = input("  Số snapshots cho ensemble [3]: ").strip()
    n_snapshots = int(snap_input) if snap_input else 3

    # ── 9. Hiển thị cấu hình ──
    print("\n" + "=" * 80)
    print("🔬 CẤU HÌNH SOTA FINE-TUNE v2")
    print("=" * 80)
    print(f"  Checkpoint nguồn : {checkpoint_path} (epoch {ckpt_epoch}, AUROC {ckpt_auroc:.4f})")
    print(f"  Model output     : {pathModel}")
    print(f"  Model            : {model_size.upper()}, {img_size}px")
    print(f"  Batch Size       : {trBatchSize}")
    print(f"  Max Epochs       : {trMaxEpoch}")
    print(f"")
    print(f"  ── FIX so với v1 (giữ nguyên training gốc) ──")
    print(f"  ASL gamma_neg    : 4 (v1 lỗi: 2) ✅")
    print(f"  Weight decay     : 0.05 (v1 lỗi: 0.08) ✅")
    print(f"  EMA decay        : 0.999 (v1 lỗi: 0.9995) ✅")
    print(f"  UW clamp         : [-4,4] (v1 lỗi: [-3,3]) ✅")
    print(f"  Weights source   : EMA (v1 lỗi: training) ✅")
    print(f"")
    print(f"  ── Kỹ thuật SOTA mới ──")
    print(f"  Label Smoothing  : {label_smoothing}")
    print(f"  MixUp alpha      : {mixup_alpha}")
    print(f"  R-Drop alpha     : {rdrop_alpha}")
    print(f"  Snapshots        : {n_snapshots}")
    print(f"")
    print(f"  ── 3 Pha Training ──")
    print(f"  Pha 1 (ep 1-{freeze_epochs})    : ❄️  Freeze backbone, Head LR=2e-4, Warmup 2ep")
    print(f"  Pha 2 (ep {freeze_epochs+1}-{swa_start-1})   : 🔓 Unfreeze, Backbone=5e-6, Head=5e-5, MixUp+R-Drop")
    print(f"  Pha 3 (ep {swa_start}-{trMaxEpoch})  : 📊 SWA LR=1e-5, Snapshot Ensemble")
    print("=" * 80)

    confirm = input("\n🚀 Bắt đầu SOTA Fine-tune? [y/n]: ").strip().lower()
    if confirm != 'y':
        print("❌ Đã hủy.")
        return

    # ── 10. Chạy Fine-tune ──
    SOTAFinetuneTrainer.train(
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
        # SOTA params
        freeze_epochs=freeze_epochs,
        swa_start=swa_start,
        mixup_alpha=mixup_alpha,
        rdrop_alpha=rdrop_alpha,
        label_smoothing=label_smoothing,
        n_snapshots=n_snapshots,
    )


if __name__ == '__main__':
    main()
