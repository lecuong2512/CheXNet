import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import os

def plot_auroc_bars(auroc_scores, class_names, save_dir):
    """Plot AUROC scores as bar chart"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#2ecc71' if score >= 0.8 else '#f39c12' if score >= 0.7 else '#e74c3c' 
              for score in auroc_scores]
    
    bars = ax.barh(range(len(class_names)), auroc_scores, color=colors, alpha=0.8)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('AUROC Score', fontsize=12, fontweight='bold')
    ax.set_title('AUROC Scores by Disease', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, auroc_scores)):
        if not np.isnan(score):
            ax.text(score + 0.01, i, f'{score:.3f}', 
                   va='center', fontsize=10, fontweight='bold')
    
    # Add reference lines
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (0.7)')
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (0.8)')
    ax.legend(loc='lower right')
    
    # Add mean AUROC
    mean_auroc = np.nanmean(auroc_scores)
    ax.text(0.02, 0.98, f'Mean AUROC: {mean_auroc:.4f}', 
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'auroc_bars.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: auroc_bars.png")

def plot_roc_curves(targets, predictions, class_names, save_dir):
    """Plot ROC curves for all classes"""
    num_classes = len(class_names)
    
    # Create subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i in range(num_classes):
        ax = axes[i]
        
        try:
            fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'AUROC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{class_names[i]}', fontweight='bold')
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            ax.set_title(f'{class_names[i]}', fontweight='bold')
    
    plt.suptitle('ROC Curves for All Diseases', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: roc_curves.png")

def plot_confusion_matrices(targets, predictions, class_names, save_dir, threshold=0.5):
    """Plot confusion matrices for each class"""
    num_classes = len(class_names)
    
    # Create subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    pred_binary = (predictions >= threshold).astype(int)
    
    for i in range(num_classes):
        ax = axes[i]
        
        # Binary confusion matrix for this class
        cm = confusion_matrix(targets[:, i], pred_binary[:, i])
        
        # Handle case where confusion matrix might not be 2x2
        if cm.shape != (2, 2):
            # Pad to 2x2 if necessary
            cm_padded = np.zeros((2, 2), dtype=int)
            cm_padded[:cm.shape[0], :cm.shape[1]] = cm
            cm = cm_padded
        
        # Normalize
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                   ax=ax, cbar=False, square=True,
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax.set_title(f'{class_names[i]}', fontweight='bold')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
        
        # Add counts
        for j in range(2):
            for k in range(2):
                ax.text(k+0.5, j+0.7, f'({cm[j,k]})', 
                       ha='center', va='center', fontsize=8, color='gray')
    
    plt.suptitle(f'Confusion Matrices (threshold={threshold})', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: confusion_matrices.png")

def plot_prediction_distribution(targets, predictions, class_names, save_dir):
    """Plot prediction score distributions"""
    num_classes = len(class_names)
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i in range(num_classes):
        ax = axes[i]
        
        # Separate predictions by ground truth
        pos_preds = predictions[targets[:, i] == 1, i]
        neg_preds = predictions[targets[:, i] == 0, i]
        
        # Plot histograms only if there's data
        if len(neg_preds) > 0:
            ax.hist(neg_preds, bins=50, alpha=0.6, color='blue', 
                   label=f'Negative (n={len(neg_preds)})', density=True)
        if len(pos_preds) > 0:
            ax.hist(pos_preds, bins=50, alpha=0.6, color='red', 
                   label=f'Positive (n={len(pos_preds)})', density=True)
        
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_xlabel('Prediction Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{class_names[i]}', fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Prediction Score Distributions', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: prediction_distributions.png")

def plot_performance_comparison(targets, predictions, auroc_scores, class_names, save_dir, threshold=0.5):
    """Plot comprehensive performance comparison"""
    pred_binary = (predictions >= threshold).astype(int)
    
    # Compute metrics for each class
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(len(class_names)):
        try:
            precisions.append(precision_score(targets[:, i], pred_binary[:, i], zero_division=0))
            recalls.append(recall_score(targets[:, i], pred_binary[:, i], zero_division=0))
            f1_scores.append(f1_score(targets[:, i], pred_binary[:, i], zero_division=0))
        except Exception as e:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = np.arange(len(class_names))
    width = 0.2
    
    ax.barh(x - 1.5*width, auroc_scores, width, label='AUROC', alpha=0.8, color='#3498db')
    ax.barh(x - 0.5*width, precisions, width, label='Precision', alpha=0.8, color='#2ecc71')
    ax.barh(x + 0.5*width, recalls, width, label='Recall', alpha=0.8, color='#f39c12')
    ax.barh(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8, color='#e74c3c')
    
    ax.set_yticks(x)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: performance_comparison.png")

def plot_results(targets, predictions, auroc_scores, class_names, save_dir='CheXNet/Results'):
    """
    Visualize test results with multiple plots
    
    Args:
        targets: Ground truth labels (N, num_classes)
        predictions: Predicted probabilities (N, num_classes)
        auroc_scores: List of AUROC scores per class
        class_names: List of disease names
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    print("\nGenerating visualizations...")
    
    # 1. AUROC Bar Plot
    plot_auroc_bars(auroc_scores, class_names, save_dir)
    
    # 2. ROC Curves
    plot_roc_curves(targets, predictions, class_names, save_dir)
    
    # 3. Confusion Matrices (binary for each class)
    plot_confusion_matrices(targets, predictions, class_names, save_dir)
    
    # 4. Prediction Distribution
    plot_prediction_distribution(targets, predictions, class_names, save_dir)
    
    # 5. Class-wise Performance Comparison
    plot_performance_comparison(targets, predictions, auroc_scores, class_names, save_dir)
    
    print(f"\n✅ All plots saved to: {save_dir}")
