# config.py - Optimal configurations for Tensor Core GPUs
import torch
import os
from typing import Dict, Optional, Tuple
import warnings

class TensorCoreConfig:
    """
    Optimal batch sizes and settings for different GPU architectures
    to maximize Tensor Core utilization
    """
    
    # Recommended batch sizes (per GPU) for different image sizes with ConvNeXtV2-Large
    BATCH_SIZES = {
        # Format: {gpu_name: {image_size: {precision: batch_size}}}
        'V100': {
            224: {'fp32': 16, 'fp16': 32, 'bf16': 32},
            384: {'fp32': 8, 'fp16': 16, 'bf16': 16},
            512: {'fp32': 4, 'fp16': 10, 'bf16': 10},
        },
        'A100': {
            224: {'fp32': 48, 'fp16': 96, 'bf16': 96},
            384: {'fp32': 24, 'fp16': 48, 'bf16': 48},
            512: {'fp32': 12, 'fp16': 28, 'bf16': 28},
        },
        'RTX 3090': {
            224: {'fp32': 24, 'fp16': 48, 'bf16': 48},
            384: {'fp32': 12, 'fp16': 24, 'bf16': 24},
            512: {'fp32': 6, 'fp16': 14, 'bf16': 14},
        },
        'RTX 4090': {
            224: {'fp32': 32, 'fp16': 64, 'bf16': 64},
            384: {'fp32': 16, 'fp16': 32, 'bf16': 32},
            512: {'fp32': 8, 'fp16': 20, 'bf16': 20},
        },
        'H100': {
            224: {'fp32': 64, 'fp16': 128, 'bf16': 128, 'fp8': 192},
            384: {'fp32': 32, 'fp16': 64, 'bf16': 64, 'fp8': 96},
            512: {'fp32': 16, 'fp16': 40, 'bf16': 40, 'fp8': 60},
        },
    }
    
    # Tensor Core matrix dimensions for optimal performance
    OPTIMAL_DIMS = {
        'volta': 8,      # Volta: 8x8x8
        'turing': 8,     # Turing: 8x8x8
        'ampere': 16,    # Ampere: 16x16x16 (TF32), 16x8x16 (FP16)
        'ada': 16,       # Ada: 16x16x16
        'hopper': 16,    # Hopper: 16x16x16
    }
    
    @staticmethod
    def get_optimal_config(image_size: int = 384, force_dtype: Optional[str] = None) -> Dict:
        """
        Get optimal configuration based on available GPU
        
        Args:
            image_size: Input image size (224, 384, or 512)
            force_dtype: Force specific dtype ('fp32', 'fp16', 'bf16', 'fp8')
        
        Returns:
            dict: Configuration with batch_size, dtype, optimizations
        """
        if not torch.cuda.is_available():
            return {
                'batch_size': 4,
                'dtype': 'fp32',
                'num_workers': min(4, os.cpu_count() or 4),
                'has_tensor_cores': False,
                'prefetch_factor': 2,
                'pin_memory': False,
            }
        
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Detect GPU architecture
        arch = None
        if compute_cap[0] == 7 and compute_cap[1] == 0:
            arch = 'volta'
        elif compute_cap[0] == 7 and compute_cap[1] == 5:
            arch = 'turing'
        elif compute_cap[0] == 8 and compute_cap[1] in [0, 6]:
            arch = 'ampere'
        elif compute_cap[0] == 8 and compute_cap[1] == 9:
            arch = 'ada'
        elif compute_cap[0] == 9 and compute_cap[1] == 0:
            arch = 'hopper'
        
        has_tensor_cores = arch is not None
        
        # Determine optimal dtype
        if force_dtype:
            dtype = force_dtype
        elif arch in ['ampere', 'ada', 'hopper']:
            dtype = 'bf16'  # bfloat16 for Ampere+
        elif arch in ['volta', 'turing']:
            dtype = 'fp16'  # float16 for older architectures
        else:
            dtype = 'fp32'
        
        # Round image size to nearest supported size
        supported_sizes = [224, 384, 512]
        image_size = min(supported_sizes, key=lambda x: abs(x - image_size))
        
        # Estimate batch size based on memory and image size
        memory_per_sample_gb = {
            224: {'fp32': 0.4, 'fp16': 0.25, 'bf16': 0.25},
            384: {'fp32': 0.8, 'fp16': 0.5, 'bf16': 0.5},
            512: {'fp32': 1.4, 'fp16': 0.9, 'bf16': 0.9},
        }
        
        base_memory_gb = 3.0  # Base model memory
        available_memory_gb = total_memory_gb * 0.80  # Leave 20% margin
        usable_memory_gb = max(0.5, available_memory_gb - base_memory_gb)
        
        estimated_batch = int(usable_memory_gb / memory_per_sample_gb[image_size][dtype])
        
        # Round to Tensor Core optimal dimensions
        if has_tensor_cores:
            optimal_dim = TensorCoreConfig.OPTIMAL_DIMS[arch]
            batch_size = (estimated_batch // optimal_dim) * optimal_dim
            batch_size = max(optimal_dim, batch_size)  # At least one multiple
        else:
            batch_size = max(2, estimated_batch)
        
        # Clamp to reasonable values
        batch_size = min(batch_size, 128)  # Max 128 per GPU
        batch_size = max(batch_size, 2)    # Min 2
        
        # Optimal number of workers (rule of thumb: 4 workers per GPU)
        num_workers = min(8, max(4, (os.cpu_count() or 4) // max(1, torch.cuda.device_count())))
        
        # Prefetch factor for better data loading
        prefetch_factor = 2 if num_workers > 0 else None
        
        config = {
            'batch_size': batch_size,
            'dtype': dtype,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
            'pin_memory': True,
            'persistent_workers': num_workers > 0,
            'has_tensor_cores': has_tensor_cores,
            'architecture': arch,
            'gpu_name': gpu_name,
            'total_memory_gb': total_memory_gb,
            'compute_capability': f"{compute_cap[0]}.{compute_cap[1]}",
            'image_size': image_size,
            'use_compile': compute_cap[0] >= 8,  # torch.compile for Ampere+
        }
        
        return config
    
    @staticmethod
    def print_config(config: Dict):
        """Print configuration in a nice format"""
        print("\n" + "="*80)
        print("🚀 Tensor Core Optimization Configuration")
        print("="*80)
        print(f"\nGPU: {config['gpu_name']}")
        print(f"Memory: {config['total_memory_gb']:.1f} GB")
        print(f"Compute Capability: {config['compute_capability']}")
        
        if config['has_tensor_cores']:
            print(f"Architecture: {config['architecture'].upper()}")
            print(f"✓ Tensor Cores: ENABLED")
        else:
            print(f"⚠ Tensor Cores: NOT AVAILABLE")
        
        print(f"\nRecommended Settings:")
        print(f"  Image Size: {config['image_size']}x{config['image_size']}")
        print(f"  Batch Size (per GPU): {config['batch_size']}")
        print(f"  Precision: {config['dtype'].upper()}")
        print(f"  Num Workers: {config['num_workers']}")
        print(f"  Prefetch Factor: {config.get('prefetch_factor', 'N/A')}")
        print(f"  Pin Memory: {config['pin_memory']}")
        print(f"  Persistent Workers: {config.get('persistent_workers', False)}")
        
        if config.get('use_compile'):
            print(f"  torch.compile: Recommended ✓")
        
        if config['has_tensor_cores']:
            print(f"\n💡 Tips for {config['architecture'].upper()}:")
            if config['architecture'] in ['ampere', 'ada', 'hopper']:
                print(f"  • Use bfloat16 for best performance")
                print(f"  • Enable TF32: torch.backends.cuda.matmul.allow_tf32 = True")
                print(f"  • Use channels_last memory format")
                print(f"  • Batch size should be multiple of 16")
                print(f"  • Consider torch.compile() for 20-30% speedup")
            elif config['architecture'] in ['volta', 'turing']:
                print(f"  • Use float16 with gradient scaling")
                print(f"  • Batch size should be multiple of 8")
        
        print("="*80 + "\n")
    
    @staticmethod
    def apply_optimizations(model: torch.nn.Module, dtype: str = 'bf16', compile_model: bool = False) -> torch.nn.Module:
        """
        Apply Tensor Core optimizations to model
        
        Args:
            model: PyTorch model
            dtype: 'bf16', 'fp16', or 'fp32'
            compile_model: Whether to use torch.compile (PyTorch 2.0+)
        
        Returns:
            Optimized model
        """
        if not torch.cuda.is_available():
            warnings.warn("No CUDA device detected, skipping optimizations")
            return model
        
        compute_cap = torch.cuda.get_device_capability(0)
        has_tensor_cores = compute_cap[0] >= 7
        
        if not has_tensor_cores:
            print("⚠ No Tensor Cores detected, skipping optimizations")
            return model
        
        print("\n🔧 Applying Tensor Core optimizations...")
        
        # Enable TF32 for Ampere+
        if compute_cap[0] >= 8 and dtype in ['bf16', 'fp32']:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  ✓ TF32 enabled")
        
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        print("  ✓ cuDNN benchmark enabled")
        
        # Convert to channels_last
        try:
            model = model.to(memory_format=torch.channels_last)
            print("  ✓ Channels-last memory format")
        except Exception as e:
            print(f"  ⚠ Could not convert to channels_last: {e}")
        
        # torch.compile for PyTorch 2.0+ on Ampere+
        if compile_model and compute_cap[0] >= 8:
            try:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                model = torch.compile(model, mode='max-autotune')
                print("  ✓ torch.compile() applied (expect 20-30% speedup)")
            except Exception as e:
                print(f"  ⚠ Could not compile model: {e}")
        
        print("✓ Optimizations applied\n")
        
        return model


def compute_class_weights(dataset, num_classes: int = 15, smooth: float = 0.1) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset
    
    Args:
        dataset: Dataset with labels
        num_classes: Number of classes
        smooth: Smoothing factor to prevent extreme weights
    
    Returns:
        Tensor of class weights
    """
    print("\n📊 Computing class weights for imbalanced data...")
    
    # Count positive samples per class
    class_counts = torch.zeros(num_classes)
    total_samples = len(dataset)
    
    for _, labels in dataset:
        class_counts += labels
    
    # Compute weights: inverse frequency with smoothing
    class_freqs = class_counts / total_samples
    class_weights = 1.0 / (class_freqs + smooth)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("Class weights computed:")
    for i, weight in enumerate(class_weights):
        freq = class_freqs[i].item()
        print(f"  Class {i:2d}: freq={freq:.4f}, weight={weight:.4f}")
    
    return class_weights


def estimate_training_time(dataset_size: int, batch_size: int, epochs: int, 
                          samples_per_sec: Optional[float] = None,
                          architecture: Optional[str] = None,
                          dtype: str = 'bf16') -> Tuple[float, Dict]:
    """
    Estimate training time
    
    Args:
        dataset_size: Number of training samples
        batch_size: Total batch size (across all GPUs)
        epochs: Number of epochs
        samples_per_sec: Throughput (if known), else estimate
        architecture: GPU architecture name
        dtype: Precision type
    
    Returns:
        Tuple of (estimated hours, detailed breakdown)
    """
    if samples_per_sec is None:
        # Rough estimates for ConvNeXtV2-Large at 384x384
        gpu_throughput = {
            ('volta', 'fp16'): 120,
            ('turing', 'fp16'): 100,
            ('ampere', 'bf16'): 250,
            ('ampere', 'fp16'): 230,
            ('ada', 'bf16'): 300,
            ('hopper', 'bf16'): 450,
            ('hopper', 'fp8'): 600,
        }
        
        key = (architecture, dtype) if architecture else None
        samples_per_sec = gpu_throughput.get(key, 100)  # Conservative default
    
    total_samples = dataset_size * epochs
    total_seconds = total_samples / samples_per_sec
    total_hours = total_seconds / 3600
    
    breakdown = {
        'total_hours': total_hours,
        'total_days': total_hours / 24,
        'hours_per_epoch': total_hours / epochs,
        'samples_per_sec': samples_per_sec,
        'total_iterations': dataset_size // batch_size * epochs,
    }
    
    return total_hours, breakdown


# Example usage
if __name__ == '__main__':
    # Test configuration
    for size in [224, 384, 512]:
        config = TensorCoreConfig.get_optimal_config(image_size=size)
        TensorCoreConfig.print_config(config)
        
        # Estimate training time
        train_size = 86524  # NIH ChestX-ray14 typical split
        time_hours, breakdown = estimate_training_time(
            dataset_size=train_size,
            batch_size=config['batch_size'] * max(1, torch.cuda.device_count()),
            epochs=100,
            architecture=config.get('architecture'),
            dtype=config['dtype']
        )
        
        print(f"\nEstimated training time @ {size}x{size}:")
        print(f"  Total: {time_hours:.1f} hours ({breakdown['total_days']:.1f} days)")
        print(f"  Per epoch: {breakdown['hours_per_epoch']:.1f} hours")
        print(f"  Throughput: {breakdown['samples_per_sec']:.0f} samples/sec")
        print("\n" + "-"*80 + "\n")
