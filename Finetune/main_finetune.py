# main_finetune.py
# ============================================================================
# SOTA Fine-tune v2.1 — 3-Phase Strategy Entry Point (Interactive & CLI)
#
# Pha 1: Freeze backbone + Warmup head + gate
# Pha 2: Unfreeze từ Best EMA + Mask-Aware MixUp + Multi-Sample Dropout
# Pha 3: SWA + Snapshot Ensemble
# ============================================================================
import os
import sys
import argparse
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

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


def parse_args():
    parser = argparse.ArgumentParser(description="SOTA Fine-tune v2.1 (3-Phase Strategy)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Đường dẫn checkpoint nguồn (.pth)")
    parser.add_argument('--output', type=str, default=None, help="Đường dẫn lưu model sau fine-tune (.pth)")
    parser.add_argument('--data-dir', type=str, default='Database', help="Thư mục chứa ảnh")
    parser.add_argument('--train-csv', type=str, default='Dataset/train_list.csv', help="File CSV Train")
    parser.add_argument('--val-csv', type=str, default='Dataset/val_list.csv', help="File CSV Val")
    parser.add_argument('--batch-size', type=int, default=0, help="Batch size (0 = tự động theo phần cứng)")
    parser.add_argument('--epochs', type=int, default=6, help="Số epoch fine-tune tối đa (khuyến nghị 5-6)")
    parser.add_argument('--freeze-epochs', type=int, default=2, help="Số epoch Pha 1 freeze backbone")
    parser.add_argument('--swa-start', type=int, default=5, help="Epoch bắt đầu Pha 3 SWA")
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help="MixUp alpha ở Pha 2+ (0 = tắt)")
    parser.add_argument('--rdrop-alpha', type=float, default=0.0, help="R-Drop alpha (0 = tắt, đã có MSD 5x trong Head)")
    parser.add_argument('--label-smoothing', type=float, default=0.02, help="Label smoothing")
    parser.add_argument('--snapshots', type=int, default=3, help="Số snapshots cho ensemble")
    parser.add_argument('--preload', action='store_true', help="Preload ảnh vào RAM")
    parser.add_argument('--num-workers-preload', type=int, default=8, help="Số luồng preload")
    parser.add_argument('--num-workers', type=int, default=4, help="Số luồng DataLoader")
    parser.add_argument('--yes', '-y', action='store_true', help="Chạy ngay không hỏi xác nhận (CLI mode)")
    return parser.parse_args()


def main():
    args = parse_args()
    cli_mode = (args.checkpoint is not None) or args.yes

    print("\n" + "=" * 80)
    print("🔬 SOTA FINE-TUNE v2.1 — 3-Phase Strategy")
    print("   Pha 1: Freeze Backbone + Warmup Head & Gate")
    print("   Pha 2: Unfreeze từ Best EMA + Mask-Aware MixUp + Multi-Sample Dropout")
    print("   Pha 3: SWA + Snapshot Ensemble")
    print("=" * 80 + "\n")

    # ── 1. Đường dẫn checkpoint nguồn ──
    if cli_mode:
        checkpoint_path = args.checkpoint or 'Trainedmodel/hybrid_model.pth'
    else:
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

    model_size = ckpt_model_size
    img_size = ckpt_img_size
    print(f"\n🔒 Khóa cứng: model_size={model_size}, img_size={img_size}")

    # ── 3. Phân tích phần cứng & Batch Size ──
    config = TensorCoreConfig.get_optimal_hybrid_config(model_size, img_size)
    TensorCoreConfig.print_config(config)

    if cli_mode:
        trBatchSize = args.batch_size if args.batch_size > 0 else config['batch_size']
        pathDirData = args.data_dir
        pathFileTrain = args.train_csv
        pathFileVal = args.val_csv
        default_output = os.path.join(os.path.dirname(checkpoint_path) or 'Trainedmodel', f'hybrid_{model_size}_sota.pth')
        pathModel = args.output or default_output
        preload_images = args.preload
        num_workers_preload = args.num_workers_preload
        trMaxEpoch = args.epochs
        num_workers_train = args.num_workers
        freeze_epochs = args.freeze_epochs
        swa_start = args.swa_start
        mixup_alpha = args.mixup_alpha
        rdrop_alpha = args.rdrop_alpha
        label_smoothing = args.label_smoothing
        n_snapshots = args.snapshots
    else:
        use_optimal = input("\nSử dụng Batch Size đề xuất? [y/n]: ").strip().lower()
        if use_optimal == 'y':
            trBatchSize = config['batch_size']
        else:
            trBatchSize = int(input("Nhập Batch Size thủ công (VD: 8, 16, 32): ").strip())

        print("\n📂 Cấu hình đường dẫn dữ liệu:")
        pathDirData = input("Thư mục chứa ảnh [Database/]: ").strip() or 'Database'
        pathFileTrain = input("File CSV Train [Dataset/train_list.csv]: ").strip() or 'Dataset/train_list.csv'
        pathFileVal = input("File CSV Val   [Dataset/val_list.csv]: ").strip() or 'Dataset/val_list.csv'

        default_output = os.path.join(os.path.dirname(checkpoint_path) or 'Trainedmodel', 'hybrid_model_sota.pth')
        pathModel = input(f"Đường dẫn lưu model fine-tuned [{default_output}]: ").strip() or default_output

        print("\n💾 Preload dữ liệu vào RAM:")
        print("  1. Không — đọc ảnh từ đĩa mỗi epoch (an toàn, ít RAM)")
        print("  2. Có    — đọc toàn bộ vào RAM 1 lần (train nhanh hơn, cần nhiều RAM)")
        preload_choice = input("Lựa chọn [1/2]: ").strip() or "1"
        preload_images = (preload_choice == '2')
        num_workers_preload = 8
        if preload_images:
            nw = input("Số threads đọc song song [8]: ").strip()
            num_workers_preload = int(nw) if nw else 8

        trMaxEpoch = int(input("\nSố Epoch tối đa [6]: ").strip() or "6")
        nw_input = input("Số luồng đọc ảnh (num_workers) [4]: ").strip()
        num_workers_train = int(nw_input) if nw_input else 4

        print("\n🔧 Cấu hình SOTA Fine-tune (Enter = giữ mặc định tối ưu):")
        freeze_input = input("  Số epoch freeze backbone [2]: ").strip()
        freeze_epochs = int(freeze_input) if freeze_input else 2

        swa_input = input("  Epoch bắt đầu SWA [5]: ").strip()
        swa_start = int(swa_input) if swa_input else 5

        mixup_input = input("  MixUp alpha (0=tắt) [0.2]: ").strip()
        mixup_alpha = float(mixup_input) if mixup_input else 0.2

        rdrop_input = input("  R-Drop alpha (0=tắt, khuyến nghị 0 vì đã có MSD 5x) [0.0]: ").strip()
        rdrop_alpha = float(rdrop_input) if rdrop_input else 0.0

        ls_input = input("  Label Smoothing (0=tắt) [0.02]: ").strip()
        label_smoothing = float(ls_input) if ls_input else 0.02

        snap_input = input("  Số snapshots cho ensemble [3]: ").strip()
        n_snapshots = int(snap_input) if snap_input else 3

    # ── Hiển thị cấu hình ──
    print("\n" + "=" * 80)
    print("🔬 CẤU HÌNH SOTA FINE-TUNE v2.1")
    print("=" * 80)
    print(f"  Checkpoint nguồn : {checkpoint_path} (epoch {ckpt_epoch}, AUROC {ckpt_auroc:.4f})")
    print(f"  Model output     : {pathModel}")
    print(f"  Model            : {model_size.upper()}, {img_size}px")
    print(f"  Batch Size       : {trBatchSize}")
    print(f"  Max Epochs       : {trMaxEpoch}")
    print(f"  Label Smoothing  : {label_smoothing}")
    print(f"  MixUp alpha      : {mixup_alpha} (Mask-aware)")
    print(f"  R-Drop alpha     : {rdrop_alpha}")
    print(f"  Snapshots        : {n_snapshots}")
    print(f"  Pha 1 (ep 1-{freeze_epochs})    : ❄️  Freeze backbone, Head+Gate LR=2e-4")
    print(f"  Pha 2 (ep {freeze_epochs+1}-{swa_start-1})    : 🔓 Unfreeze từ Best EMA, Backbone=5e-6, Head=5e-5")
    print(f"  Pha 3 (ep {swa_start}-{trMaxEpoch})    : 📊 SWA LR=5e-6, Snapshot Ensemble")
    print("=" * 80)

    if not cli_mode:
        confirm = input("\n🚀 Bắt đầu SOTA Fine-tune? [y/n]: ").strip().lower()
        if confirm != 'y':
            print("❌ Đã hủy.")
            return

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
        freeze_epochs=freeze_epochs,
        swa_start=swa_start,
        mixup_alpha=mixup_alpha,
        rdrop_alpha=rdrop_alpha,
        label_smoothing=label_smoothing,
        n_snapshots=n_snapshots,
    )


if __name__ == '__main__':
    main()
