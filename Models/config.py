# -*- coding: utf-8 -*-
# config.py
import torch
import os
from typing import Dict, Optional

class TensorCoreConfig:
    @staticmethod
    def get_optimal_hybrid_config(model_size: str = 'base', image_size: int = 384) -> Dict:
        """
        Tính toán tài nguyên phần cứng cho kiến trúc Lai (ConvNeXtV2 + SwinV2)
        Hỗ trợ AMP (BF16/FP16) để tối ưu VRAM và throughput trên Tensor Core GPUs.
        """
        if not torch.cuda.is_available():
            return {
                'batch_size': 2,
                'gpu_name': 'CPU',
                'has_tensor_cores': False,
                'use_amp': False,
                'amp_dtype': torch.float32,
            }
        
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        has_tensor_cores = compute_cap[0] >= 7

        # AMP: BF16 chỉ dùng trên Ampere+ (compute cap >= 8.0) vì cuDNN
        # convolution kernels yêu cầu phần cứng BF16 thực sự.
        # torch.cuda.is_bf16_supported() trả True trên T4 (cc 7.5) do
        # PyTorch hỗ trợ emulated BF16, nhưng cuDNN sẽ lỗi runtime.
        use_amp = has_tensor_cores
        if use_amp:
            amp_dtype = torch.bfloat16 if compute_cap >= (8, 0) else torch.float16
        else:
            amp_dtype = torch.float32

        # Ước tính VRAM tiêu thụ cho mỗi sample (GB)
        # AMP BF16/FP16 giảm ~40% memory cho activations so với FP32
        vram_per_sample_fp32 = {
            'base':  {256: 0.35, 384: 0.8},
            'large': {256: 0.8,  384: 1.8}
        }
        vram_per_sample_amp = {
            'base':  {256: 0.22, 384: 0.50},
            'large': {256: 0.50, 384: 1.10}
        }
        
        base_model_vram = 2.0 if model_size == 'base' else 5.5
        # A100 quản lý memory tốt hơn — giữ lại 10% thay vì 15%
        available_memory = total_memory_gb * 0.90 - base_model_vram
        available_memory = max(0.5, available_memory)
        
        sample_cost_table = vram_per_sample_amp if use_amp else vram_per_sample_fp32
        sample_cost = sample_cost_table[model_size][image_size]
        estimated_batch = int(available_memory / sample_cost)
        
        # Làm tròn batch size cho Tensor Cores (bội số của 8)
        if has_tensor_cores and estimated_batch >= 8:
            batch_size = (estimated_batch // 8) * 8
        else:
            batch_size = max(2, estimated_batch)
            
        # Giới hạn an toàn
        batch_size = min(batch_size, 128)

        return {
            'model_size': model_size,
            'image_size': image_size,
            'batch_size': batch_size,
            'gpu_name': gpu_name,
            'total_memory_gb': total_memory_gb,
            'has_tensor_cores': has_tensor_cores,
            'compute_capability': f"{compute_cap[0]}.{compute_cap[1]}",
            'use_amp': use_amp,
            'amp_dtype': amp_dtype,
        }
    
    @staticmethod
    def print_config(config: Dict):
        print("\n" + "="*80)
        print("🚀 Tensor Core & VRAM Analysis")
        print("="*80)
        print(f"GPU: {config['gpu_name']}")
        print(f"Memory: {config['total_memory_gb']:.1f} GB")
        print(f"Compute Capability: {config['compute_capability']}")
        
        if config['has_tensor_cores']:
            print(f"✓ Tensor Cores: ENABLED (Mixed Precision / TF32)")

        if config.get('use_amp'):
            dtype_name = 'BF16' if config['amp_dtype'] == torch.bfloat16 else 'FP16'
            print(f"✓ AMP: ENABLED ({dtype_name}) — VRAM giảm ~40%, throughput tăng 2-3x")
        else:
            print(f"✗ AMP: DISABLED (FP32)")
            
        print(f"\n💡 Cấu hình đề xuất cho Hybrid ({config['model_size'].upper()}):")
        print(f"  • Kích thước ảnh: {config['image_size']}x{config['image_size']}")
        print(f"  • Batch Size (per GPU): {config['batch_size']}")
        
        if config['model_size'] == 'large' and config['image_size'] == 384:
            print(f"  ⚠ CẢNH BÁO VRAM: Đã tự động kích hoạt Gradient Checkpointing.")
        print("="*80 + "\n")
