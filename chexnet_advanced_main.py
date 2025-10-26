import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from tqdm import tqdm
import json

from Models.Model import create_distributed_model
from Models.read_data import create_dataloaders
from Models.advanced_trainer import AdvancedTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='CheXNet Advanced Training')
    
    # Paths
    parser.add_argument('--image_root', type=str, default='CheXNet/Database',
                        help='Thư mục gốc chứa images_00*')
    parser.add_argument('--dataset_dir', type=str, default='CheXNet/Dataset',
                        help='Thư mục chứa train.csv, val.csv, test.csv')
    parser.add_argument('--save_dir', type=str, default='CheXNet/Trainedmodel',
                        help='Thư mục lưu checkpoints')
    parser.add_argument('--results_dir', type=str, default='CheXNet/results',
                        help='Thư mục lưu kết quả test')
    
    # Training
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Chế độ train hoặc test')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Số epoch')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Số workers cho DataLoader')
    
    # Advanced Training Strategy
    parser.add_argument('--freeze_epochs', type=int, default=3,
                        help='Số epochs freeze backbone (0 = không freeze)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Số epochs warmup learning rate')
    parser.add_argument('--gradual_unfreeze', action='store_true', default=True,
                        help='Dùng gradual unfreezing (default: True)')
    parser.add_argument('--no_gradual_unfreeze', dest='gradual_unfreeze', action='store_false',
                        help='Không dùng gradual unfreezing')
    parser.add_argument('--unfreeze_interval', type=int, default=3,
                        help='Số epochs giữa mỗi lần unfreeze layer (nếu dùng gradual)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (0 = không dùng)')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Dùng Focal Loss thay vì BCE')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'onecycle'],
                        help='Learning rate scheduler')
    
    # Checkpoint
    parser.add_argument('--resume', type=str, default=None,
                        help='Đường dẫn checkpoint để train tiếp')
    parser.add_argument('--test_checkpoint', type=str, default=None,
                        help='Đường dẫn checkpoint để test')
    
    return parser.parse_args()


def test_model(model, test_loader, device, results_dir, diseases):
    """Test model và tạo các biểu đồ kết quả"""
    model.eval()
    all_labels = []
    all_outputs = []
    
    print("\n" + "="*70)
    print("BẮT ĐẦU TESTING")
    print("="*70 + "\n")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device, memory_format=torch.channels_last)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())
    
    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Tính AUROC
    aurocs = []
    results = {}
    
    print("\n" + "="*70)
    print("KẾT QUẢ AUROC CHO TỪNG BỆNH")
    print("="*70)
    
    for i, disease in enumerate(diseases):
        try:
            if len(np.unique(all_labels[:, i])) > 1:
                auroc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                aurocs.append(auroc)
                results[disease] = auroc
                print(f"{disease:25s}: {auroc:.4f}")
            else:
                aurocs.append(np.nan)
                results[disease] = None
                print(f"{disease:25s}: N/A (chỉ một class)")
        except Exception as e:
            aurocs.append(np.nan)
            results[disease] = None
            print(f"{disease:25s}: Error - {e}")
    
    valid_aurocs = [a for a in aurocs if not np.isnan(a)]
    mean_auroc = np.mean(valid_aurocs) if valid_aurocs else 0.0
    results['mean_auroc'] = mean_auroc
    
    print("="*70)
    print(f"MEAN AUROC: {mean_auroc:.4f}")
    print("="*70 + "\n")
    
    # Lưu kết quả
    with open(os.path.join(results_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 1. Biểu đồ AUROC
    plt.figure(figsize=(14, 6))
    valid_diseases = [diseases[i] for i in range(len(aurocs)) if not np.isnan(aurocs[i])]
    valid_aurocs_plot = [aurocs[i] for i in range(len(aurocs)) if not np.isnan(aurocs[i])]
    
    colors = ['#2ecc71' if a >= 0.8 else '#f39c12' if a >= 0.7 else '#e74c3c' for a in valid_aurocs_plot]
    bars = plt.bar(range(len(valid_aurocs_plot)), valid_aurocs_plot, color=colors, alpha=0.8)
    plt.xticks(range(len(valid_aurocs_plot)), valid_diseases, rotation=45, ha='right')
    plt.ylabel('AUROC', fontsize=12)
    plt.title(f'AUROC cho từng bệnh (Mean AUROC: {mean_auroc:.4f})', fontsize=14, fontweight='bold')
    plt.axhline(y=mean_auroc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_auroc:.4f}')
    plt.ylim([0.5, 1.0])
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'auroc_per_disease.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curves
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, disease in enumerate(diseases):
        ax = axes[i]
        if not np.isnan(aurocs[i]) and len(np.unique(all_labels[:, i])) > 1:
            fpr, tpr, _ = roc_curve(all_labels[:, i], all_outputs[:, i])
            ax.plot(fpr, tpr, linewidth=2, label=f'AUROC = {aurocs[i]:.4f}')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(disease, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
            ax.set_title(disease, fontweight='bold')
    
    plt.suptitle('ROC Curves cho từng bệnh', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, disease in enumerate(diseases):
        ax = axes[i]
        if not np.isnan(aurocs[i]):
            preds = (all_outputs[:, i] > 0.5).astype(int)
            cm = confusion_matrix(all_labels[:, i], preds)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax.set_title(disease, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
            ax.set_title(disease, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle('Confusion Matrix (threshold=0.5)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction Distribution
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, disease in enumerate(diseases):
        ax = axes[i]
        if not np.isnan(aurocs[i]):
            positive_preds = all_outputs[all_labels[:, i] == 1, i]
            negative_preds = all_outputs[all_labels[:, i] == 0, i]
            
            ax.hist(negative_preds, bins=50, alpha=0.6, color='blue', label='Negative (True)')
            ax.hist(positive_preds, bins=50, alpha=0.6, color='red', label='Positive (True)')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Count')
            ax.set_title(disease, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
            ax.set_title(disease, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle('Phân bố dự đoán: Thực tế Positive vs Negative', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'prediction_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Đã lưu tất cả kết quả vào: {results_dir}\n")
    
    return mean_auroc, aurocs


def plot_training_history(history, save_dir):
    """Vẽ biểu đồ training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # AUROC
    axes[0, 1].plot(history['train_auroc'], label='Train AUROC', linewidth=2)
    axes[0, 1].plot(history['val_auroc'], label='Val AUROC', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUROC')
    axes[0, 1].set_title('Training & Validation AUROC', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(history['learning_rates'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Best Val AUROC
    best_epoch = np.argmax(history['val_auroc']) + 1
    best_auroc = max(history['val_auroc'])
    axes[1, 1].plot(history['val_auroc'], linewidth=2, color='orange')
    axes[1, 1].axhline(y=best_auroc, color='red', linestyle='--', linewidth=2)
    axes[1, 1].axvline(x=best_epoch-1, color='red', linestyle='--', linewidth=2)
    axes[1, 1].scatter([best_epoch-1], [best_auroc], color='red', s=100, zorder=5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation AUROC')
    axes[1, 1].set_title(f'Best Val AUROC: {best_auroc:.4f} (Epoch {best_epoch})', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu training history chart -> {save_dir}/training_history.png")


def main():
    args = parse_args()
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print("✓ Tensor Core optimizations enabled")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nSử dụng device: {device}")
    if torch.cuda.is_available():
        print(f"Số GPU khả dụng: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        capability = torch.cuda.get_device_capability(0)
        print(f"  Compute Capability: {capability[0]}.{capability[1]}")
        
        torch.cuda.empty_cache()
        print("\n✓ Đã clear CUDA cache")
    
    # Disease names
    diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
                'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
                'Pleural_Thickening', 'Hernia', 'No Finding']
    
    # Create dataloaders
    print("\nĐang tạo DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        image_root_dir=args.image_root,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nĐang tạo model...")
    model, device = create_distributed_model(
        num_classes=15,
        pretrained=True,
        dropout_rate=args.dropout,
        local_rank=0
    )
    
    if args.mode == 'train':
        # Setup optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Setup scheduler
        if args.scheduler == 'cosine':
            # Cosine Annealing sau warmup
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - args.warmup_epochs,
                eta_min=1e-6
            )
            print(f"✓ Sử dụng CosineAnnealingLR scheduler")
        else:  # onecycle
            steps_per_epoch = len(train_loader)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.lr,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos'
            )
            print(f"✓ Sử dụng OneCycleLR scheduler")
        
        # Training config
        config = {
            'freeze_epochs': args.freeze_epochs,
            'warmup_epochs': args.warmup_epochs,
            'gradual_unfreeze': args.gradual_unfreeze,
            'unfreeze_interval': args.unfreeze_interval,
            'label_smoothing': args.label_smoothing,
            'use_focal_loss': args.use_focal_loss
        }
        
        # Create trainer
        trainer = AdvancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            save_dir=args.save_dir,
            num_classes=15,
            config=config
        )
        
        # Train
        trainer.train(
            num_epochs=args.epochs,
            scheduler=scheduler,
            resume_from=args.resume
        )
        
        # Plot training history
        plot_training_history(trainer.history, args.save_dir)
        
    elif args.mode == 'test':
        # Load checkpoint
        if args.test_checkpoint is None:
            args.test_checkpoint = os.path.join(args.save_dir, 'chexnetmodel.pth')
        
        print(f"\nĐang load checkpoint: {args.test_checkpoint}")
        checkpoint = torch.load(args.test_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint từ epoch {checkpoint['epoch']}")
        print(f"✓ Best AUROC: {checkpoint['best_auroc']:.4f}")
        
        # Test
        mean_auroc, aurocs = test_model(
            model=model,
            test_loader=test_loader,
            device=device,
            results_dir=args.results_dir,
            diseases=diseases
        )


if __name__ == '__main__':
    main()
