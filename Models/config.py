# config.py - Optimal configurations for Tensor Core GPUs
import torch

class TensorCoreConfig:
    """
    Optimal batch sizes and settings for different GPU architectures
    to maximize Tensor Core utilization
    """
    
    # Recommended batch sizes (per GPU) for 384x384 images with ConvNeXtV2-Large
    BATCH_SIZES = {
        # Volta (V100) - 16GB
        'V100': {
            'fp32': 8,
            'fp16': 16,
            'bf16': 16,
        },
        # Turing (RTX 2080 Ti) - 11GB
        'RTX 2080 Ti': {
            'fp32': 4,
            'fp16': 12,
        },
        # Ampere (A100) - 40GB/80GB
        'A100': {
            'fp32': 24,
            'fp16': 48,
            'bf16': 48,  # Recommended for A100
        },
        # Ampere (RTX 3090) - 24GB
        'RTX 3090': {
            'fp32': 12,
            'fp16': 24,
            'bf16': 24,
        },
        # Ampere (RTX 3080) - 10GB
        'RTX 3080': {
            'fp32': 6,
            'fp16': 14,
            'bf16': 14,
        },
        # Ada Lovelace (RTX 4090) - 24GB
        'RTX 4090': {
            'fp32': 16,
            'fp16': 32,
            'bf16': 32,  # Recommended for RTX 4090
        },
        # Ada Lovelace (RTX 4080) - 16GB
        'RTX 4080': {
            'fp32': 10,
            'fp16': 22,
            'bf16': 22,
        },
        # Hopper (H100) - 80GB
        'H100': {
            'fp32': 32,
            'fp16': 64,
            'bf16': 64,  # Recommended for H100
            'fp8': 96,   # FP8 support on H100
        },
    }
    
    # Tensor Core matrix dimensions for optimal performance
    # Dimensions should be multiples of these for best utilization
    OPTIMAL_DIMS = {
        'volta': 8,      # Volta: 8x8x8
        'turing': 8,     # Turing: 8x8x8
        'ampere': 16,    # Ampere: 16x16x16 (TF32), 16x8x16 (FP16)
        'ada': 16,       # Ada: 16x16x16
        'hopper': 16,    # Hopper: 16x16x16
    }
    
    @staticmethod
    def get_optimal_config(image_size=384):
        """
        Get optimal configuration based on available GPU
        
        Returns:
            dict: Configuration with batch_size, dtype, optimizations
        """
        if not torch.cuda.is_available():
            return {
                'batch_size': 4,
                'dtype': 'fp32',
                'num_workers': 0,
                'has_tensor_cores': False
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
        elif compute_cap[0] == 8 and compute_cap[1] == 0:
            arch = 'ampere'
        elif compute_cap[0] == 8 and compute_cap[1] == 6:
            arch = 'ampere'
        elif compute_cap[0] == 8 and compute_cap[1] == 9:
            arch = 'ada'
        elif compute_cap[0] == 9 and compute_cap[1] == 0:
            arch = 'hopper'
        
        has_tensor_cores = arch is not None
        
        # Determine optimal dtype
        if arch in ['ampere', 'ada', 'hopper']:
            dtype = 'bf16'  # bfloat16 for Ampere+
        elif arch in ['volta', 'turing']:
            dtype = 'fp16'  # float16 for older architectures
        else:
            dtype = 'fp32'
        
        # Estimate batch size based on memory and image size
        # Rule of thumb: ConvNeXtV2-Large uses ~12GB for batch=16 at 384x384 with fp16
        memory_per_sample_gb = {
            'fp32': 0.8,
            'fp16': 0.5,
            'bf16': 0.5,
        }
        
        base_memory_gb = 2.0  # Base model memory
        available_memory_gb = total_memory_gb * 0.85  # Leave 15% margin
        usable_memory_gb = available_memory_gb - base_memory_gb
        
        estimated_batch = int(usable_memory_gb / memory_per_sample_gb[dtype])
        
        # Round to Tensor Core optimal dimensions
        if has_tensor_cores:
            optimal_dim = TensorCoreConfig.OPTIMAL_DIMS[arch]
            batch_size = (estimated_batch // optimal_dim) * optimal_dim
            batch_size = max(optimal_dim, batch_size)  # At least one multiple
        else:
            batch_size = max(2, estimated_batch)
        
        # Clamp to reasonable values
        batch_size = min(batch_size, 64)  # Max 64 per GPU
        batch_size = max(batch_size, 4)   # Min 4
        
        # Optimal number of workers
        num_workers = min(8, torch.get_num_threads() // 2)
        
        config = {
            'batch_size': batch_size,
            'dtype': dtype,
            'num_workers': num_workers,
            'has_tensor_cores': has_tensor_cores,
            'architecture': arch,
            'gpu_name': gpu_name,
            'total_memory_gb': total_memory_gb,
            'compute_capability': f"{compute_cap[0]}.{compute_cap[1]}",
        }
        
        return config
    
    @staticmethod
    def print_config(config):
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
        print(f"  Batch Size (per GPU): {config['batch_size']}")
        print(f"  Precision: {config['dtype'].upper()}")
        print(f"  Num Workers: {config['num_workers']}")
        
        if config['has_tensor_cores']:
            print(f"\n💡 Tips for {config['architecture'].upper()}:")
            if config['architecture'] in ['ampere', 'ada', 'hopper']:
                print(f"  • Use bfloat16 for best performance")
                print(f"  • Enable TF32: torch.backends.cuda.matmul.allow_tf32 = True")
                print(f"  • Use channels_last memory format")
                print(f"  • Batch size should be multiple of 16")
            elif config['architecture'] in ['volta', 'turing']:
                print(f"  • Use float16 with gradient scaling")
                print(f"  • Batch size should be multiple of 8")
        
        print("="*80 + "\n")
    
    @staticmethod
    def apply_optimizations(model, dtype='bf16'):
        """
        Apply Tensor Core optimizations to model
        
        Args:
            model: PyTorch model
            dtype: 'bf16', 'fp16', or 'fp32'
        
        Returns:
            Optimized model
        """
        if not torch.cuda.is_available():
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
        
        # CUDA graphs for Ampere+ (requires static shapes)
        if compute_cap[0] >= 8:
            print("  💡 Consider using CUDA graphs for static batch sizes")
        
        print("✓ Optimizations applied\n")
        
        return model


def estimate_training_time(dataset_size, batch_size, epochs, samples_per_sec=None):
    """
    Estimate training time
    
    Args:
        dataset_size: Number of training samples
        batch_size: Batch size
        epochs: Number of epochs
        samples_per_sec: Throughput (if known), else estimate
    
    Returns:
        Estimated time in hours
    """
    if samples_per_sec is None:
        # Rough estimates for ConvNeXtV2-Large at 384x384
        gpu_throughput = {
            'V100_fp16': 120,
            'A100_bf16': 250,
            'RTX3090_bf16': 180,
            'RTX4090_bf16': 220,
            'H100_bf16': 400,
        }
        
        # Use conservative estimate
        samples_per_sec = 100
    
    total_samples = dataset_size * epochs
    total_seconds = total_samples / samples_per_sec
    total_hours = total_seconds / 3600
    
    return total_hours


# Example usage
if __name__ == '__main__':
    config = TensorCoreConfig.get_optimal_config(image_size=384)
    TensorCoreConfig.print_config(config)
    
    # Estimate training time for NIH ChestX-ray14
    train_size = 86524  # Typical split
    time_hours = estimate_training_time(
        dataset_size=train_size,
        batch_size=config['batch_size'] * torch.cuda.device_count(),
        epochs=100
    )
    
    print(f"Estimated training time: {time_hours:.1f} hours ({time_hours/24:.1f} days)")