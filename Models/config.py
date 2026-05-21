# config.py
import torch
import os
from typing import Dict, Optional

class TensorCoreConfig:
    @staticmethod
    def get_optimal_hybrid_config(model_size: str = 'base', image_size: int = 384) -> Dict:
        """
        Tính toán tài nguyên phần cứng cho kiến trúc Lai (ConvNeXtV2 + SwinV2)
        """
        if not torch.cuda.is_available():
            return {
                'batch_size': 2,
                'gpu_name': 'CPU',
                'has_tensor_cores': False,
            }
        
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        has_tensor_cores = compute_cap[0] >= 7

        # Ước tính VRAM tiêu thụ cho mỗi sample (GB) với kiến trúc Hybrid
        # SwinV2 ngốn rất nhiều RAM ở size 384
        vram_per_sample = {
            'base': {256: 0.35, 384: 0.8},
            'large': {256: 0.8, 384: 1.8}
        }
        
        base_model_vram = 2.0 if model_size == 'base' else 5.5
        available_memory = total_memory_gb * 0.85 # Giữ lại 15% an toàn
        usable_memory = max(0.5, available_memory - base_model_vram)
        
        sample_cost = vram_per_sample[model_size][image_size]
        estimated_batch = int(usable_memory / sample_cost)
        
        # Làm tròn batch size cho Tensor Cores (bội số của 8)
        if has_tensor_cores and estimated_batch >= 8:
            batch_size = (estimated_batch // 8) * 8
        else:
            batch_size = max(2, estimated_batch)
            
        # Giới hạn an toàn để tránh sập RAM
        batch_size = min(batch_size, 64)

        return {
            'model_size': model_size,
            'image_size': image_size,
            'batch_size': batch_size,
            'gpu_name': gpu_name,
            'total_memory_gb': total_memory_gb,
            'has_tensor_cores': has_tensor_cores,
            'compute_capability': f"{compute_cap[0]}.{compute_cap[1]}"
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
            
        print(f"\n💡 Cấu hình đề xuất cho Hybrid ({config['model_size'].upper()}):")
        print(f"  • Kích thước ảnh: {config['image_size']}x{config['image_size']}")
        print(f"  • Batch Size (per GPU): {config['batch_size']}")
        
        if config['model_size'] == 'large' and config['image_size'] == 384:
            print(f"  ⚠ CẢNH BÁO VRAM: Đã tự động kích hoạt Gradient Checkpointing.")
        print("="*80 + "\n")
