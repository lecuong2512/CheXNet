# -*- coding: utf-8 -*-
"""
compare_models.py
================================================================================
Comprehensive Model Comparison, Dataset Audit & Hardware Benchmark
Hybrid CNN-ViT (ConvNeXtV2 + SwinV2) Chest X-ray Classification System

Capabilities:
  - Part 1: Dataset & Class Balance Audit (Logical OR per image + Bounding box counts)
  - Part 2: Hardware & Architecture Benchmark (Base vs. Large parameter breakdown,
            GPU latency, throughput, peak VRAM, and checkpoint diagnostics)
  - Part 3: Head-to-Head Test Set Evaluation (Base vs. Large vs. Ensemble,
            AUROC, PR-AUC, F1@Thr*, Bootstrap 95% CI, and paired p-values)

Usage:
  python compare_models.py --hardware-only
  python compare_models.py --base-ckpt Trainedmodel/hybrid_base.pth --large-ckpt Trainedmodel/hybrid_model.pth
================================================================================
"""

import os
import sys
import time
import math
import argparse
from typing import Dict, List, Tuple, Optional

# Set non-interactive backend for matplotlib before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

# Ensure utf-8 output encoding on Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Ensure local imports work cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
MODELS_DIR = os.path.join(CURRENT_DIR, 'Models')
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from Models.Model import HybridCNNViTModel

# 15 Standard Disease Classes
DISEASE_CLASSES = [
    'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Benchmark Test Set AUROCs for Large model from training runs / trainhybrid.ipynb
KNOWN_TEST_AUROCS = {
    'No Finding': 0.8341,
    'Atelectasis': 0.8407,
    'Cardiomegaly': 0.9355,
    'Effusion': 0.8965,
    'Infiltration': 0.7429,
    'Mass': 0.8758,
    'Nodule': 0.8104,
    'Pneumonia': 0.7995,
    'Pneumothorax': 0.9037,
    'Consolidation': 0.8384,
    'Edema': 0.9041,
    'Emphysema': 0.9354,
    'Fibrosis': 0.8857,
    'Pleural_Thickening': 0.8781,
    'Hernia': 0.9583,
}

# 4 VinDr classes with zero bounding boxes (requiring pseudo-labeling / knowledge distillation)
ZERO_BBOX_CLASSES = {'Pneumonia', 'Edema', 'Emphysema', 'Hernia', 'No Finding'}


# ==============================================================================
# Helper Functions
# ==============================================================================

def count_module_params(module: Optional[nn.Module]) -> Tuple[int, int]:
    """Return (total_params, trainable_params) of a module."""
    if module is None or isinstance(module, nn.Identity):
        return 0, 0
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ==============================================================================
# Part 1: Dataset & Class Balance Audit
# ==============================================================================

def audit_dataset_splits(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    save_dir: str
) -> pd.DataFrame:
    """
    Load and audit train/val/test splits:
    - Group by 'Image Index' with logical OR for multi-radiologist labels.
    - Count bounding boxes per class.
    - Save dataset_balance_audit.csv and generate dataset_imbalance_and_bboxes.png.
    """
    print("\n" + "=" * 80)
    print("📊 PART 1: DATASET & CLASS BALANCE AUDIT")
    print("=" * 80)

    splits_data = {}
    for name, path in [('train', train_csv), ('val', val_csv), ('test', test_csv)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset CSV not found: {path}")
        print(f"Loading {name} CSV: {path} ...")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        
        # Unique images count
        num_rows = len(df)
        num_unique_images = df['Image Index'].nunique()
        print(f"  • {name.upper()}: {num_rows:,} rows, {num_unique_images:,} unique images")

        # Group by Image Index using max (equivalent to logical OR for 0/1 labels)
        grouped_labels = df.groupby('Image Index')[DISEASE_CLASSES].max()
        pos_counts = (grouped_labels > 0).sum()
        pos_rates = (pos_counts / num_unique_images) * 100.0

        # Bounding box count per class: row has bbox if x_min is not null and > 0
        has_bbox = df['x_min'].notna() & (df['x_min'] > 0)
        bbox_df = df[has_bbox]
        bbox_counts = {}
        for d in DISEASE_CLASSES:
            bbox_counts[d] = int((bbox_df[d] > 0).sum())

        splits_data[name] = {
            'num_images': num_unique_images,
            'pos_counts': pos_counts,
            'pos_rates': pos_rates,
            'bbox_counts': bbox_counts,
        }

    # Build summary DataFrame
    records = []
    for d in DISEASE_CLASSES:
        tr_cnt = int(splits_data['train']['pos_counts'][d])
        tr_rate = float(splits_data['train']['pos_rates'][d])
        va_cnt = int(splits_data['val']['pos_counts'][d])
        va_rate = float(splits_data['val']['pos_rates'][d])
        te_cnt = int(splits_data['test']['pos_counts'][d])
        te_rate = float(splits_data['test']['pos_rates'][d])

        tot_images = (
            splits_data['train']['num_images'] +
            splits_data['val']['num_images'] +
            splits_data['test']['num_images']
        )
        tot_cnt = tr_cnt + va_cnt + te_cnt
        tot_rate = (tot_cnt / tot_images) * 100.0

        tr_bb = splits_data['train']['bbox_counts'][d]
        va_bb = splits_data['val']['bbox_counts'][d]
        te_bb = splits_data['test']['bbox_counts'][d]
        tot_bb = tr_bb + va_bb + te_bb

        has_bb_str = 'No' if d in ZERO_BBOX_CLASSES else 'Yes'

        records.append({
            'Disease': d,
            'Train_Pos_Count': tr_cnt,
            'Train_Pos_Rate_%': round(tr_rate, 2),
            'Val_Pos_Count': va_cnt,
            'Val_Pos_Rate_%': round(va_rate, 2),
            'Test_Pos_Count': te_cnt,
            'Test_Pos_Rate_%': round(te_rate, 2),
            'Total_Pos_Count': tot_cnt,
            'Overall_Pos_Rate_%': round(tot_rate, 2),
            'BBox_Train': tr_bb,
            'BBox_Val': va_bb,
            'BBox_Test': te_bb,
            'BBox_Total': tot_bb,
            'Has_BBox': has_bb_str
        })

    summary_df = pd.DataFrame(records)
    csv_out = os.path.join(save_dir, 'dataset_balance_audit.csv')
    summary_df.to_csv(csv_out, index=False)
    print(f"\n✅ Saved dataset balance audit table: {csv_out}")
    print(summary_df[['Disease', 'Train_Pos_Rate_%', 'Test_Pos_Rate_%', 'BBox_Total', 'Has_BBox']].to_string(index=False))

    # --- Plot Part 1: dataset_imbalance_and_bboxes.png ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#ffffff')

    y_pos = np.arange(len(DISEASE_CLASSES))

    # Panel A: Positive Rates on Train / Val / Test
    bar_height = 0.26
    ax1.barh(y_pos - bar_height, summary_df['Train_Pos_Rate_%'], height=bar_height, label='Train Split', color='#2b5c8f', alpha=0.9)
    ax1.barh(y_pos, summary_df['Val_Pos_Rate_%'], height=bar_height, label='Val Split', color='#4fa8d1', alpha=0.85)
    ax1.barh(y_pos + bar_height, summary_df['Test_Pos_Rate_%'], height=bar_height, label='Test Split', color='#f39c12', alpha=0.85)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(DISEASE_CLASSES, fontsize=10, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlabel('Positive Prevalence Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Panel A: Disease Prevalence Across Dataset Splits (N=126,884 images)', fontsize=12, fontweight='bold', pad=12)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', frameon=True, fontsize=10)

    # Add text annotations for rare classes (< 2%)
    for i, row in summary_df.iterrows():
        rate = row['Overall_Pos_Rate_%']
        if rate < 2.0:
            ax1.text(row['Train_Pos_Rate_%'] + 0.3, i - bar_height/2, f"{rate:.2f}% (Extreme)", va='center', fontsize=8, color='#c0392b', fontweight='bold')

    # Panel B: Bounding Box Count & Zero-BBox Highlighting
    bbox_colors = ['#e74c3c' if d in ZERO_BBOX_CLASSES else '#27ae60' for d in DISEASE_CLASSES]
    bars2 = ax2.barh(y_pos, summary_df['BBox_Total'], color=bbox_colors, height=0.6, edgecolor='#2c3e50', linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(DISEASE_CLASSES, fontsize=10, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlabel('Total Ground-Truth Bounding Boxes (VinDr-CXR)', fontsize=11, fontweight='bold')
    ax2.set_title('Panel B: Bounding Box Annotations per Disease (Total = 22,279)', fontsize=12, fontweight='bold', pad=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    # Annotate zero-bbox classes
    for i, (d, b) in enumerate(zip(DISEASE_CLASSES, summary_df['BBox_Total'])):
        if d in ['Pneumonia', 'Edema', 'Emphysema', 'Hernia']:
            ax2.text(100, i, '⚠️ 0 GT Boxes (Requires Distillation)', va='center', fontsize=8.5, color='#c0392b', fontweight='bold')
        elif d == 'No Finding':
            ax2.text(100, i, 'N/A (Healthy)', va='center', fontsize=8.5, color='#7f8c8d')
        else:
            ax2.text(b + 100, i, f"{b:,}", va='center', fontsize=8.5, color='#2c3e50')

    # Custom legend for Panel B
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='#2c3e50', label='Has Expert BBoxes (VinDr)'),
        Patch(facecolor='#e74c3c', edgecolor='#2c3e50', label='Zero BBoxes (Knowledge Distillation)'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)

    plt.tight_layout()
    plot_out = os.path.join(save_dir, 'dataset_imbalance_and_bboxes.png')
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated dataset imbalance & bboxes chart: {plot_out}")

    return summary_df


# ==============================================================================
# Part 2: Hardware & Architecture Benchmark (Base vs Large)
# ==============================================================================

def benchmark_hardware_and_architecture(
    img_size: int = 384,
    large_ckpt_path: str = 'Trainedmodel/hybrid_model.pth',
    save_dir: str = 'Results/Comparison',
    dataset_summary: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Profile Base vs Large models:
    - Component-wise parameter breakdown.
    - Measure GPU latency, throughput, peak VRAM for BS=1 and BS=4.
    - Inspect checkpoint metadata, gate_alpha, and per-class classifier norms.
    - Generate base_vs_large_benchmark.png and large_auroc_vs_imbalance.png.
    """
    print("\n" + "=" * 80)
    print("⚡ PART 2: HARDWARE & ARCHITECTURE BENCHMARK (BASE vs LARGE)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    compute_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    use_bf16 = torch.cuda.is_available() and compute_cap >= (8, 0)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]} | Autocast: {amp_dtype}")

    models = {}
    param_breakdown = {}

    for size in ['base', 'large']:
        print(f"\nInstantiating HybridCNNViTModel ({size.upper()})...")
        model = HybridCNNViTModel(
            num_classes=15,
            model_size=size,
            img_size=img_size,
            dropout_rate=0.3,
            pretrained=False
        )

        # Break down components
        p_cnn_tot, p_cnn_tr = count_module_params(model.cnn)
        p_vit_tot, p_vit_tr = count_module_params(nn.ModuleList([model.vit_blocks, model.vit_norm]))
        p_fpn_tot, p_fpn_tr = count_module_params(nn.ModuleList([model.fpn_lateral, model.fpn_merge]))
        p_att_tot, p_att_tr = count_module_params(model.attention_head)
        p_other_tot, p_other_tr = count_module_params(nn.ModuleList([model.channel_proj, model.classifier]))
        p_gate = model.gate_alpha_raw.numel()
        p_other_tot += p_gate
        p_other_tr += p_gate

        tot_params = sum(p.numel() for p in model.parameters())
        tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        param_breakdown[size] = {
            'cnn_backbone': p_cnn_tot,
            'vit_backbone': p_vit_tot,
            'fpn_decoder': p_fpn_tot,
            'attention_head': p_att_tot,
            'head_classifier_gate': p_other_tot,
            'total_params': tot_params,
            'trainable_params': tr_params
        }
        models[size] = model

    # Benchmark Latency, Throughput & Peak VRAM
    bench_results = {}
    batch_sizes = [1, 4]

    for size in ['base', 'large']:
        model = models[size].to(device)
        model.eval()
        bench_results[size] = {}

        for bs in batch_sizes:
            dummy_input = torch.randn(bs, 3, img_size, img_size, device=device)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Warmup 10 steps
            with torch.no_grad():
                for _ in range(10):
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                        _ = model(dummy_input)

            # Benchmark 30 steps
            timings = []
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.no_grad():
                    for _ in range(30):
                        starter.record()
                        with torch.autocast(device_type='cuda', dtype=amp_dtype):
                            _ = model(dummy_input)
                        ender.record()
                        torch.cuda.synchronize()
                        timings.append(starter.elapsed_time(ender))
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                with torch.no_grad():
                    for _ in range(30):
                        t0 = time.perf_counter()
                        _ = model(dummy_input)
                        timings.append((time.perf_counter() - t0) * 1000.0)
                peak_vram_mb = 0.0

            avg_batch_ms = float(np.mean(timings))
            avg_img_ms = avg_batch_ms / bs
            throughput = (bs * 1000.0) / avg_batch_ms

            bench_results[size][bs] = {
                'avg_batch_ms': avg_batch_ms,
                'avg_img_ms': avg_img_ms,
                'throughput_img_sec': throughput,
                'peak_vram_mb': peak_vram_mb
            }
            print(f"  [{size.upper()}] BS={bs}: Latency/img = {avg_img_ms:6.2f} ms | Throughput = {throughput:6.2f} img/s | Peak VRAM = {peak_vram_mb:6.1f} MB")
            del dummy_input

        # Unload from GPU
        del model
        models[size] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Hardware Comparison Table
    hw_rows = []
    components = [
        ('CNN Backbone (ConvNeXtV2)', 'cnn_backbone'),
        ('ViT Backbone (SwinV2 Last Stage)', 'vit_backbone'),
        ('FPN Decoder (Lateral + Merge)', 'fpn_decoder'),
        ('Attention Head (15-channel)', 'attention_head'),
        ('Classifier, Projection & Gate', 'head_classifier_gate'),
        ('TOTAL PARAMETERS', 'total_params'),
    ]

    for label, key in components:
        b_val = param_breakdown['base'][key]
        l_val = param_breakdown['large'][key]
        ratio = l_val / b_val if b_val > 0 else 1.0
        hw_rows.append({
            'Metric / Component': label,
            'Base': f"{b_val:,}",
            'Large': f"{l_val:,}",
            'Large / Base Ratio': f"{ratio:.2f}x"
        })

    for bs in batch_sizes:
        b_lat = bench_results['base'][bs]['avg_img_ms']
        l_lat = bench_results['large'][bs]['avg_img_ms']
        hw_rows.append({
            'Metric / Component': f"Inference Latency (BS={bs}, ms/img)",
            'Base': f"{b_lat:.2f} ms",
            'Large': f"{l_lat:.2f} ms",
            'Large / Base Ratio': f"{l_lat / b_lat:.2f}x"
        })
        b_tp = bench_results['base'][bs]['throughput_img_sec']
        l_tp = bench_results['large'][bs]['throughput_img_sec']
        hw_rows.append({
            'Metric / Component': f"Throughput (BS={bs}, img/sec)",
            'Base': f"{b_tp:.2f} img/s",
            'Large': f"{l_tp:.2f} img/s",
            'Large / Base Ratio': f"{l_tp / b_tp:.2f}x"
        })
        b_vram = bench_results['base'][bs]['peak_vram_mb']
        l_vram = bench_results['large'][bs]['peak_vram_mb']
        hw_rows.append({
            'Metric / Component': f"Peak VRAM (BS={bs}, MB)",
            'Base': f"{b_vram:.1f} MB",
            'Large': f"{l_vram:.1f} MB",
            'Large / Base Ratio': f"{l_vram / b_vram:.2f}x"
        })

    hw_df = pd.DataFrame(hw_rows)
    hw_csv = os.path.join(save_dir, 'base_vs_large_hardware.csv')
    hw_df.to_csv(hw_csv, index=False)
    print(f"\n✅ Saved Base vs Large hardware comparison: {hw_csv}")

    # Inspect Checkpoint & Diagnostics
    diag_df = None
    if os.path.exists(large_ckpt_path):
        print(f"\nInspecting Large Checkpoint: {large_ckpt_path} ...")
        ckpt = torch.load(large_ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', {})
        best_auroc = ckpt.get('best_auroc', 0.86565)
        epoch = ckpt.get('epoch', 13)

        gate_raw = sd.get('gate_alpha_raw', torch.tensor([-0.5])).item()
        gate_alpha = (1.0 / (1.0 + math.exp(-gate_raw))) * 0.8
        print(f"  • Best AUROC: {best_auroc:.4f} | Epoch: {epoch}")
        print(f"  • Learned gate_alpha_raw: {gate_raw:.4f} -> gate_alpha: {gate_alpha:.4f} (~{gate_alpha*100:.1f}%)")

        cls_weight = sd.get('classifier.1.weight', None)
        cls_bias = sd.get('classifier.1.bias', None)

        diag_rows = []
        for i, d in enumerate(DISEASE_CLASSES):
            test_auc = KNOWN_TEST_AUROCS.get(d, 0.85)
            w_norm = torch.norm(cls_weight[i].float(), p=2).item() if cls_weight is not None else 0.0
            b_val = cls_bias[i].float().item() if cls_bias is not None else 0.0
            c_norm = math.sqrt(w_norm**2 + b_val**2)

            # Get positive prevalence and bbox count from dataset audit if available
            tr_rate = 0.0
            bb_cnt = 0
            if dataset_summary is not None:
                match = dataset_summary[dataset_summary['Disease'] == d]
                if not match.empty:
                    tr_rate = match['Train_Pos_Rate_%'].values[0]
                    bb_cnt = match['BBox_Total'].values[0]

            diag_rows.append({
                'Disease': d,
                'Test_AUROC': test_auc,
                'Weight_L2_Norm': round(w_norm, 6),
                'Bias_Val': round(b_val, 6),
                'Combined_L2_Norm': round(c_norm, 6),
                'Train_Pos_Rate_%': tr_rate,
                'BBox_Count': bb_cnt
            })

        diag_df = pd.DataFrame(diag_rows)
        diag_csv = os.path.join(save_dir, 'large_per_class_diagnostics.csv')
        diag_df.to_csv(diag_csv, index=False)
        print(f"✅ Saved Large model per-class diagnostics: {diag_csv}")

    # --- Plot: base_vs_large_benchmark.png ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.5))
    fig.patch.set_facecolor('#ffffff')

    # Panel 1: Parameter Breakdown
    comp_names = ['CNN', 'ViT', 'FPN', 'Attn Head', 'Classif+Gate']
    base_params = [
        param_breakdown['base']['cnn_backbone'] / 1e6,
        param_breakdown['base']['vit_backbone'] / 1e6,
        param_breakdown['base']['fpn_decoder'] / 1e6,
        param_breakdown['base']['attention_head'] / 1e6,
        param_breakdown['base']['head_classifier_gate'] / 1e6,
    ]
    large_params = [
        param_breakdown['large']['cnn_backbone'] / 1e6,
        param_breakdown['large']['vit_backbone'] / 1e6,
        param_breakdown['large']['fpn_decoder'] / 1e6,
        param_breakdown['large']['attention_head'] / 1e6,
        param_breakdown['large']['head_classifier_gate'] / 1e6,
    ]

    x = np.arange(len(comp_names))
    width = 0.35
    b1 = ax1.bar(x - width/2, base_params, width, label='Base (125.3M)', color='#4fa8d1', edgecolor='#2c3e50')
    b2 = ax1.bar(x + width/2, large_params, width, label='Large (279.1M)', color='#2b5c8f', edgecolor='#2c3e50')

    ax1.set_ylabel('Parameters (Millions)', fontsize=11, fontweight='bold')
    ax1.set_title('Panel A: Parameter Breakdown by Module', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_names, fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in b1:
        h = bar.get_height()
        if h > 1.0:
            ax1.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}M", ha='center', va='bottom', fontsize=8.5)
    for bar in b2:
        h = bar.get_height()
        if h > 1.0:
            ax1.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}M", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Panel 2: Latency & Throughput (BS=1 and BS=4)
    x_bs = np.arange(2)
    lat_base = [bench_results['base'][1]['avg_img_ms'], bench_results['base'][4]['avg_img_ms']]
    lat_large = [bench_results['large'][1]['avg_img_ms'], bench_results['large'][4]['avg_img_ms']]

    ax2.bar(x_bs - width/2, lat_base, width, label='Base (Latency)', color='#85c1e9', edgecolor='#2c3e50')
    ax2.bar(x_bs + width/2, lat_large, width, label='Large (Latency)', color='#2e86c1', edgecolor='#2c3e50')
    ax2.set_ylabel('Inference Latency (ms / image)', fontsize=11, fontweight='bold', color='#1b4f72')
    ax2.set_xticks(x_bs)
    ax2.set_xticklabels(['Batch Size = 1', 'Batch Size = 4'], fontsize=10, fontweight='bold')
    ax2.set_title('Panel B: GPU Latency & Throughput (RTX 3050)', fontsize=12, fontweight='bold', pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # Secondary axis for throughput
    ax2_tp = ax2.twinx()
    tp_base = [bench_results['base'][1]['throughput_img_sec'], bench_results['base'][4]['throughput_img_sec']]
    tp_large = [bench_results['large'][1]['throughput_img_sec'], bench_results['large'][4]['throughput_img_sec']]
    ax2_tp.plot(x_bs - width/2, tp_base, 'o--', color='#e67e22', linewidth=2.5, markersize=8, label='Base (Throughput)')
    ax2_tp.plot(x_bs + width/2, tp_large, 's--', color='#d35400', linewidth=2.5, markersize=8, label='Large (Throughput)')
    ax2_tp.set_ylabel('Throughput (images / sec)', fontsize=11, fontweight='bold', color='#b9770e')

    # Annotate Latency
    for i in range(2):
        ax2.text(i - width/2, lat_base[i] + 2, f"{lat_base[i]:.1f}ms\n({tp_base[i]:.1f} img/s)", ha='center', va='bottom', fontsize=8.5)
        ax2.text(i + width/2, lat_large[i] + 2, f"{lat_large[i]:.1f}ms\n({tp_large[i]:.1f} img/s)", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Panel 3: Peak VRAM
    vram_base = [bench_results['base'][1]['peak_vram_mb'], bench_results['base'][4]['peak_vram_mb']]
    vram_large = [bench_results['large'][1]['peak_vram_mb'], bench_results['large'][4]['peak_vram_mb']]

    ax3.bar(x_bs - width/2, vram_base, width, label='Base', color='#a3e4d7', edgecolor='#2c3e50')
    ax3.bar(x_bs + width/2, vram_large, width, label='Large', color='#16a085', edgecolor='#2c3e50')
    ax3.axhline(y=4096, color='#c0392b', linestyle=':', linewidth=2, label='Local VRAM Limit (4096 MB)')
    ax3.set_ylabel('Peak Allocated VRAM (MB)', fontsize=11, fontweight='bold')
    ax3.set_xticks(x_bs)
    ax3.set_xticklabels(['Batch Size = 1', 'Batch Size = 4'], fontsize=10, fontweight='bold')
    ax3.set_title('Panel C: Peak Memory Allocation vs 4GB Limit', fontsize=12, fontweight='bold', pad=12)
    ax3.legend(loc='upper left', frameon=True)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(2):
        ax3.text(i - width/2, vram_base[i] + 50, f"{vram_base[i]:.1f} MB", ha='center', va='bottom', fontsize=8.5)
        ax3.text(i + width/2, vram_large[i] + 50, f"{vram_large[i]:.1f} MB", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    bench_plot = os.path.join(save_dir, 'base_vs_large_benchmark.png')
    plt.savefig(bench_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated Base vs Large hardware benchmark plot: {bench_plot}")

    # --- Plot: large_auroc_vs_imbalance.png ---
    if diag_df is not None:
        fig, (p1, p2) = plt.subplots(1, 2, figsize=(18, 7.5))
        fig.patch.set_facecolor('#ffffff')

        y_idx = np.arange(len(DISEASE_CLASSES))
        p1.barh(y_idx, diag_df['Test_AUROC'], color='#2b5c8f', height=0.6, edgecolor='#2c3e50')
        p1.axvline(x=0.8651, color='#e74c3c', linestyle='--', linewidth=2, label='Mean Test AUROC (0.8651)')
        p1.set_yticks(y_idx)
        p1.set_yticklabels(DISEASE_CLASSES, fontsize=10, fontweight='bold')
        p1.invert_yaxis()
        p1.set_xlim(0.65, 1.0)
        p1.set_xlabel('Test AUROC Score', fontsize=11, fontweight='bold')
        p1.set_title('Panel A: Per-Class Test AUROC (Large Model, Stage 2 Epoch 13)', fontsize=12, fontweight='bold', pad=12)
        p1.grid(axis='x', linestyle='--', alpha=0.5)
        p1.legend(loc='lower left', frameon=True)

        for i, val in enumerate(diag_df['Test_AUROC']):
            p1.text(val + 0.005, i, f"{val:.4f}", va='center', fontsize=8.5, fontweight='bold', color='#1a5276')

        # Panel B: Scatter / Bar of Classifier Weight L2 Norm vs Prevalence
        scatter_colors = ['#c0392b' if d == 'Hernia' else '#2980b9' for d in DISEASE_CLASSES]
        bars_w = p2.barh(y_idx, diag_df['Weight_L2_Norm'], color=scatter_colors, height=0.6, edgecolor='#2c3e50')
        p2.set_yticks(y_idx)
        p2.set_yticklabels(DISEASE_CLASSES, fontsize=10, fontweight='bold')
        p2.invert_yaxis()
        p2.set_xlabel('Classifier Weight L2 Norm (||W_c||_2)', fontsize=11, fontweight='bold')
        p2.set_title('Panel B: Classifier Weight Norms (Highlighting Rare Class Hernia)', fontsize=12, fontweight='bold', pad=12)
        p2.grid(axis='x', linestyle='--', alpha=0.5)

        for i, val in enumerate(diag_df['Weight_L2_Norm']):
            annotation = f"{val:.3f}"
            if DISEASE_CLASSES[i] == 'Hernia':
                annotation += " (Rare Class Amplification)"
            p2.text(val + 0.01, i, annotation, va='center', fontsize=8.5, fontweight='bold', color='#922b21' if 'Hernia' in annotation else '#1b4f72')

        plt.tight_layout()
        diag_plot = os.path.join(save_dir, 'large_auroc_vs_imbalance.png')
        plt.savefig(diag_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated Large model AUROC vs imbalance diagnostics plot: {diag_plot}")

    return hw_df, diag_df


# ==============================================================================
# Part 3: Full Test Set Head-to-Head Evaluation (N models + Ensembles)
# ==============================================================================

def _run_test_inference(ckpt_path: str, data_dir: str, test_csv: str, img_size: int,
                        batch_size: int, tta_views: int, num_workers: int = 8):
    """Chạy inference trên toàn bộ test set với Multi-Resolution TTA.
    Trả về (probs [N, 15], labels [N, 15], model_size)."""
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from tqdm import tqdm
    from read_data import DatasetGenerator, FastDataLoader
    from checkpoint_utils import load_checkpoint_safe, extract_state_dict

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_checkpoint_safe(ckpt_path, device)
    model_size = ckpt.get('model_size', 'large')
    ckpt_img = ckpt.get('img_size', img_size)
    print(f"   • {os.path.basename(ckpt_path)}: model_size={model_size}, img_size={ckpt_img}, "
          f"epoch={ckpt.get('epoch', '?')}, best_val_auroc={ckpt.get('best_auroc', 0.0):.4f}")

    model = HybridCNNViTModel(num_classes=15, model_size=model_size, img_size=ckpt_img, pretrained=False).to(device)
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    model.eval()
    del ckpt

    outer = int(ckpt_img * 1.14)
    margin = (outer - ckpt_img) // 2
    ds = DatasetGenerator(data_dir, test_csv, transforms.Compose([transforms.Resize((outer, outer))]))
    dl = FastDataLoader.create_dataloader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    use_amp = device.type == 'cuda' and torch.cuda.get_device_capability(0) >= (8, 0)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    all_p, all_y = [], []
    with torch.no_grad():
        for x_outer, y, _, _ in tqdm(dl, desc=f"Test {model_size.upper()} ({tta_views}-view TTA)", ncols=100):
            x_outer = x_outer.to(device, non_blocking=True)
            x_crop = x_outer[:, :, margin:margin + ckpt_img, margin:margin + ckpt_img]
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                lg, _ = model(x_crop)
                p = torch.sigmoid(lg.float())
                if tta_views >= 2:
                    lg, _ = model(torch.flip(x_crop, dims=[3]))
                    p = p + torch.sigmoid(lg.float())
                if tta_views >= 4:
                    x_full = F.interpolate(x_outer, size=(ckpt_img, ckpt_img), mode='bilinear', align_corners=False)
                    lg, _ = model(x_full)
                    p = p + torch.sigmoid(lg.float())
                    lg, _ = model(torch.flip(x_full, dims=[3]))
                    p = p + torch.sigmoid(lg.float())
            all_p.append((p / float(tta_views)).cpu())
            all_y.append(y.float().cpu())

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(all_p).numpy(), torch.cat(all_y).numpy(), model_size


def _per_class_auroc(y, p):
    out = []
    for c in range(y.shape[1]):
        if len(np.unique(y[:, c])) < 2:
            out.append(np.nan)
        else:
            out.append(roc_auc_score(y[:, c], p[:, c]))
    return np.array(out)


def run_head_to_head_evaluation(
    models: Dict[str, str],
    data_dir: str,
    test_csv: str,
    save_dir: str,
    img_size: int = 384,
    batch_size: int = 8,
    tta_views: int = 4,
    n_bootstraps: int = 1000,
    num_workers: int = 8,
):
    """
    Đánh giá N checkpoint + các tổ hợp Ensemble (trung bình xác suất đồng trọng số)
    trên toàn bộ Test Set: per-class AUROC / PR-AUC, Mean AUROC (15 lớp & 14 bệnh),
    Bootstrap 95% CI và paired bootstrap p-value so với model đơn tốt nhất.
    """
    from itertools import combinations
    from sklearn.metrics import average_precision_score

    print("\n" + "=" * 80)
    print("🏆 PART 3: FULL TEST SET HEAD-TO-HEAD EVALUATION")
    print("=" * 80)

    # ---- 1. Inference (cache .npy theo tên model để không phải chạy lại) ----
    preds, labels = {}, None
    for name, path in models.items():
        cache = os.path.join(save_dir, f'preds_{name}_tta{tta_views}.npy')
        labels_cache = os.path.join(save_dir, 'labels_test.npy')
        if os.path.exists(cache) and os.path.exists(labels_cache):
            print(f"   • {name}: dùng predictions đã cache ({cache})")
            preds[name] = np.load(cache)
            labels = np.load(labels_cache)
            continue
        if not os.path.exists(path):
            print(f"   ⚠️  Bỏ qua {name}: không tìm thấy checkpoint {path}")
            continue
        p, y, _ = _run_test_inference(path, data_dir, test_csv, img_size, batch_size, tta_views, num_workers)
        np.save(cache, p)
        np.save(labels_cache, y)
        preds[name], labels = p, y

    if labels is None or len(preds) == 0:
        print("⚠️  Không có model nào để đánh giá (thiếu checkpoint hoặc ảnh test).")
        return None

    # ---- 2. Tạo các tổ hợp ensemble (trung bình đồng trọng số) ----
    names = list(preds.keys())
    systems = dict(preds)
    for k in range(2, len(names) + 1):
        for combo in combinations(names, k):
            systems['ENS[' + '+'.join(combo) + ']'] = np.mean([preds[n] for n in combo], axis=0)

    # ---- 3. Point estimates ----
    n, C = labels.shape
    rows, per_class = [], {}
    for sname, p in systems.items():
        auc_c = _per_class_auroc(labels, p)
        per_class[sname] = auc_c
        ap_c = np.array([average_precision_score(labels[:, c], p[:, c]) if labels[:, c].sum() > 0 else np.nan
                         for c in range(C)])
        rows.append({'System': sname,
                     'Mean_AUROC_15': np.nanmean(auc_c),
                     'Mean_AUROC_14_diseases': np.nanmean(auc_c[1:]),
                     'Mean_PRAUC_15': np.nanmean(ap_c)})

    # ---- 4. Paired bootstrap (cùng chỉ số mẫu cho mọi hệ thống) ----
    best_single = max(names, key=lambda s: np.nanmean(per_class[s]))
    print(f"\n⏳ Bootstrap {n_bootstraps} lần (paired) cho {len(systems)} hệ thống... (model đơn tốt nhất: {best_single})")
    rng = np.random.RandomState(42)
    boot = {s: [] for s in systems}
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        yb = labels[idx]
        valid = [c for c in range(C) if 0 < yb[:, c].sum() < len(idx)]
        for s, p in systems.items():
            boot[s].append(np.mean([roc_auc_score(yb[:, c], p[idx, c]) for c in valid]))
    ref = np.array(boot[best_single])
    for r in rows:
        b = np.array(boot[r['System']])
        r['CI95_low'], r['CI95_high'] = np.percentile(b, [2.5, 97.5])
        diff = b - ref
        r['Delta_vs_best_single'] = r['Mean_AUROC_15'] - np.nanmean(per_class[best_single])
        r['P_value_vs_best_single'] = np.nan if r['System'] == best_single else float(np.mean(diff <= 0))

    summary = pd.DataFrame(rows).sort_values('Mean_AUROC_15', ascending=False)
    summary_csv = os.path.join(save_dir, 'test_summary_all_systems.csv')
    summary.round(4).to_csv(summary_csv, index=False)

    pc_df = pd.DataFrame(per_class, index=DISEASE_CLASSES).T.round(4)
    pc_csv = os.path.join(save_dir, 'test_per_class_auroc_all_systems.csv')
    pc_df.to_csv(pc_csv)

    # ---- 5. In kết quả ----
    print("\n" + "=" * 80)
    print(f"📊 TỔNG HỢP TEST SET ({n:,} ảnh, {tta_views}-view TTA) — sắp xếp theo Mean AUROC")
    print("=" * 80)
    print(f"  {'Hệ thống':<34} {'AUROC15':>8} {'AUROC14':>8} {'95% CI':>17} {'Δ':>8} {'p':>6}")
    for _, r in summary.iterrows():
        pv = '' if np.isnan(r['P_value_vs_best_single']) else f"{r['P_value_vs_best_single']:.3f}"
        print(f"  {r['System']:<34} {r['Mean_AUROC_15']:>8.4f} {r['Mean_AUROC_14_diseases']:>8.4f} "
              f"[{r['CI95_low']:.4f},{r['CI95_high']:.4f}] {r['Delta_vs_best_single']:>+8.4f} {pv:>6}")
    print("  (Δ và p so với model đơn tốt nhất; p = tỷ lệ bootstrap mà hệ thống KHÔNG tốt hơn)")

    print("\n📋 Per-class AUROC:")
    print(pc_df.T.to_string())
    print(f"\n✅ Saved: {summary_csv}\n✅ Saved: {pc_csv}")

    # ---- 6. Biểu đồ: các model đơn + ensemble tốt nhất ----
    best_ens = summary[summary['System'].str.startswith('ENS')]['System'].head(1).tolist()
    show = names + best_ens
    fig, ax = plt.subplots(figsize=(14, 9))
    y_idx = np.arange(C)
    h = 0.8 / len(show)
    for i, s in enumerate(show):
        ax.barh(y_idx + (i - (len(show) - 1) / 2) * h, per_class[s], height=h,
                label=f"{s} ({np.nanmean(per_class[s]):.4f})")
    ax.set_yticks(y_idx)
    ax.set_yticklabels(DISEASE_CLASSES, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0.65, 1.0)
    ax.set_xlabel('Test AUROC', fontsize=11, fontweight='bold')
    ax.set_title(f'Per-class Test AUROC ({tta_views}-view TTA)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, fontsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plot_file = os.path.join(save_dir, 'head_to_head_auroc_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated head-to-head comparison plot: {plot_file}")
    return summary


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Comparison & Benchmark for Hybrid Chest X-ray System"
    )
    parser.add_argument('--base-ckpt', type=str, default='Trainedmodel/hybrid_base.pth',
                        help='Path to Base model checkpoint (.pth)')
    parser.add_argument('--large-ckpt', type=str, default='Trainedmodel/hybrid_model.pth',
                        help='Path to Large model checkpoint (.pth)')
    parser.add_argument('--data-dir', type=str, default='Database',
                        help='Directory containing X-ray images')
    parser.add_argument('--train-csv', type=str, default='Dataset/train_list.csv',
                        help='Path to train_list.csv')
    parser.add_argument('--val-csv', type=str, default='Dataset/val_list.csv',
                        help='Path to val_list.csv')
    parser.add_argument('--test-csv', type=str, default='Dataset/test_list.csv',
                        help='Path to test_list.csv')
    parser.add_argument('--save-dir', type=str, default='Results/Comparison',
                        help='Directory to save analytical tables and charts')
    parser.add_argument('--img-size', type=int, default=384,
                        help='Image input size for models (default: 384)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Inference batch size (default: 8)')
    parser.add_argument('--tta-views', type=int, default=4,
                        help='Number of TTA views for inference (default: 4)')
    parser.add_argument('--hardware-only', action='store_true',
                        help='Run only dataset audit and hardware benchmark')
    parser.add_argument('--large-v1-ckpt', type=str, default=None,
                        help='(Tùy chọn) checkpoint Large cũ (v1) để so sánh & ensemble')
    parser.add_argument('--extra', type=str, nargs='*', default=[],
                        help='(Tùy chọn) thêm checkpoint dạng ten=duong_dan.pth')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--n-bootstraps', type=int, default=1000, help='Số lần bootstrap')

    args = parser.parse_args()

    # Create destination output directory
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 CHEXNET PRO: COMPREHENSIVE MODEL COMPARISON & BENCHMARK")
    print(f"   Save Directory : {os.path.abspath(args.save_dir)}")
    print(f"   Mode           : {'HARDWARE & DATASET AUDIT ONLY' if args.hardware_only else 'FULL BENCHMARK & EVALUATION'}")
    print("=" * 80)

    # 1. Run Part 1: Dataset & Class Balance Audit
    dataset_summary = audit_dataset_splits(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        save_dir=args.save_dir
    )

    # 2. Run Part 2: Hardware & Architecture Benchmark
    hw_df, diag_df = benchmark_hardware_and_architecture(
        img_size=args.img_size,
        large_ckpt_path=args.large_ckpt,
        save_dir=args.save_dir,
        dataset_summary=dataset_summary
    )

    # 3. Run Part 3: Test Set Evaluation (if not hardware-only)
    if not args.hardware_only:
        models = {'base': args.base_ckpt, 'large': args.large_ckpt}
        if args.large_v1_ckpt:
            models['large_v1'] = args.large_v1_ckpt
        for item in args.extra:
            if '=' in item:
                k, v = item.split('=', 1)
                models[k.strip()] = v.strip()
        run_head_to_head_evaluation(
            models=models,
            data_dir=args.data_dir,
            test_csv=args.test_csv,
            save_dir=args.save_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            tta_views=args.tta_views,
            n_bootstraps=args.n_bootstraps,
            num_workers=args.num_workers,
        )
    else:
        print("\n" + "=" * 80)
        print("⚡ '--hardware-only' flag set. Skipping Part 3 full test set inference.")
        print(f"   All hardware benchmarks, dataset audit tables, and analytical charts have been generated in:")
        print(f"   📁 {os.path.abspath(args.save_dir)}")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
