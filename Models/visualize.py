# visualize.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import os

# [GIỮ NGUYÊN CÁC HÀM CŨ Ở ĐÂY: plot_auroc_bars, plot_roc_curves, plot_confusion_matrices...]

def plot_training_progress(train_bce, val_bce, train_dice, val_auroc, save_dir):
    """
    Vẽ biểu đồ tiến trình học để theo dõi sự dịch chuyển 2 giai đoạn
    """
    epochs = range(1, len(train_bce) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. BCE Loss
    axes[0].plot(epochs, train_bce, 'b-', label='Train BCE Loss')
    axes[0].plot(epochs, val_bce, 'r--', label='Val BCE Loss')
    axes[0].set_title('Tiến trình BCE Loss (Phân loại)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Dice Loss (Định vị vùng bệnh)
    if any(d > 0 for d in train_dice):
        axes[1].plot(epochs, train_dice, 'g-', label='Train Dice Loss (VinDr)')
        axes[1].set_title('Tiến trình Dice Loss (Giai đoạn 1)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Không có dữ liệu Dice Loss', ha='center', va='center')
        axes[1].set_title('Tiến trình Dice Loss')
        
    # 3. AUROC
    axes[2].plot(epochs, val_auroc, 'm-', label='Validation AUROC', linewidth=2)
    axes[2].axhline(y=0.75, color='orange', linestyle='--', label='Ngưỡng chuyển Phase')
    axes[2].set_title('Tiến trình AUROC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Mean AUROC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Đã lưu: training_progress.png")
