# benchmark.py - Benchmark training performance with Tensor Core optimizations
import torch
import torch.nn as nn
import time
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from Models.Model import ConvNeXtV2Model

def benchmark_inference(model, batch_size, image_size, num_iterations=100, warmup=10, dtype=torch.float16):
    """
    Benchmark inference speed
    
    Args:
        model: PyTorch model
        batch_size: Batch size to test
        image_size: Input image size
        num_iterations: Number of iterations to test
        warmup: Number of warmup iterations
        dtype: Data type (torch.float16, torch.bfloat16, torch.float32)
    
    Returns:
        dict: Benchmark results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Convert to channels_last if using Tensor Cores
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        model = model.to(memory_format=torch.channels_last)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        dummy_input = dummy_input.to(memory_format=torch.channels_last)
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            if dtype == torch.float32:
                _ = model(dummy_input)
            else:
                with autocast(dtype=dtype):
                    _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking inference ({num_iterations} iterations)...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            
            if dtype == torch.float32:
                _ = model(dummy_input)
            else:
                with autocast(dtype=dtype):
                    _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.time()
            times.append(end - start)
    
    times = np.array(times)
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / mean_time
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    results = {
        'batch_size': batch_size,
        'image_size': image_size,
        'dtype': str(dtype),
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_samples_per_sec': throughput,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved,
    }
    
    return results


def benchmark_training(model, batch_size, image_size, num_iterations=50, warmup=5, dtype=torch.float16):
    """
    Benchmark training speed (forward + backward)
    
    Returns:
        dict: Benchmark results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # Convert to channels_last if using Tensor Cores
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        model = model.to(memory_format=torch.channels_last)
    
    # Create dummy input and target
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
    dummy_target = torch.randint(0, 2, (batch_size, 15), dtype=torch.float32, device=device)
    
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        dummy_input = dummy_input.to(memory_format=torch.channels_last)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Gradient scaler for fp16
    scaler = GradScaler() if dtype == torch.float16 else GradScaler(enabled=False)
    
    # Warmup
    print(f"\nWarming up training ({warmup} iterations)...")
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        
        if dtype == torch.float32:
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
        else:
            with autocast(dtype=dtype):
                output = model(dummy_input)
                loss = criterion(output, dummy_target)
            
            if dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
        
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    print(f"Benchmarking training ({num_iterations} iterations)...")
    times = []
    
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        
        if dtype == torch.float32:
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
        else:
            with autocast(dtype=dtype):
                output = model(dummy_input)
                loss = criterion(output, dummy_target)
            
            if dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / mean_time
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    results = {
        'batch_size': batch_size,
        'image_size': image_size,
        'dtype': str(dtype),
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_samples_per_sec': throughput,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved,
    }
    
    return results


def compare_dtypes(batch_size=16, image_size=384):
    """Compare performance across different data types"""
    print("\n" + "="*80)
    print("Comparing Data Types Performance")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    model = ConvNeXtV2Model(num_classes=15, pretrained=False)
    compute_cap = torch.cuda.get_device_capability()
    
    # Determine which dtypes to test
    dtypes_to_test = [torch.float32, torch.float16]
    if compute_cap[0] >= 8:  # Ampere+
        dtypes_to_test.append(torch.bfloat16)
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"Testing batch_size={batch_size}, image_size={image_size}")
    
    # Inference benchmark
    print("\n" + "-"*80)
    print("INFERENCE BENCHMARK")
    print("-"*80)
    
    inference_results = []
    for dtype in dtypes_to_test:
        print(f"\nTesting {dtype}...")
        model_copy = ConvNeXtV2Model(num_classes=15, pretrained=False)
        results = benchmark_inference(model_copy, batch_size, image_size, 
                                     num_iterations=100, dtype=dtype)
        inference_results.append(results)
        del model_copy
        torch.cuda.empty_cache()
    
    # Training benchmark
    print("\n" + "-"*80)
    print("TRAINING BENCHMARK")
    print("-"*80)
    
    training_results = []
    for dtype in dtypes_to_test:
        print(f"\nTesting {dtype}...")
        model_copy = ConvNeXtV2Model(num_classes=15, pretrained=False)
        results = benchmark_training(model_copy, batch_size, image_size, 
                                    num_iterations=50, dtype=dtype)
        training_results.append(results)
        del model_copy
        torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    print(f"{'Dtype':<12} {'Time (ms)':<15} {'Throughput':<20} {'Memory (GB)'}")
    print("-"*80)
    
    for res in inference_results:
        dtype_name = res['dtype'].split('.')[-1]
        print(f"{dtype_name:<12} {res['mean_time_ms']:<15.2f} "
              f"{res['throughput_samples_per_sec']:<20.1f} "
              f"{res['memory_allocated_gb']:<.2f}")
    
    print("\n" + "="*80)
    print("TRAINING RESULTS")
    print("="*80)
    print(f"{'Dtype':<12} {'Time (ms)':<15} {'Throughput':<20} {'Memory (GB)'}")
    print("-"*80)
    
    for res in training_results:
        dtype_name = res['dtype'].split('.')[-1]
        print(f"{dtype_name:<12} {res['mean_time_ms']:<15.2f} "
              f"{res['throughput_samples_per_sec']:<20.1f} "
              f"{res['memory_allocated_gb']:<.2f}")
    
    # Calculate speedups
    if len(inference_results) > 1:
        fp32_time = inference_results[0]['mean_time_ms']
        print("\n" + "="*80)
        print("SPEEDUP vs FP32")
        print("="*80)
        print(f"{'Dtype':<12} {'Inference':<15} {'Training'}")
        print("-"*80)
        
        for i, res in enumerate(inference_results[1:], 1):
            dtype_name = res['dtype'].split('.')[-1]
            inf_speedup = fp32_time / res['mean_time_ms']
            train_speedup = training_results[0]['mean_time_ms'] / training_results[i]['mean_time_ms']
            print(f"{dtype_name:<12} {inf_speedup:<15.2f}x {train_speedup:.2f}x")


def find_max_batch_size(image_size=384, dtype=torch.bfloat16):
    """Find maximum batch size that fits in GPU memory"""
    print("\n" + "="*80)
    print("Finding Maximum Batch Size")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Data type: {dtype}")
    
    batch_sizes = [2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    max_working_batch = 0
    
    for batch_size in batch_sizes:
        try:
            print(f"\nTesting batch_size={batch_size}...", end=" ")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model = ConvNeXtV2Model(num_classes=15, pretrained=False).cuda()
            optimizer = torch.optim.AdamW(model.parameters())
            criterion = nn.BCEWithLogitsLoss()
            
            # Test with training loop
            dummy_input = torch.randn(batch_size, 3, image_size, image_size, device='cuda')
            dummy_target = torch.randint(0, 2, (batch_size, 15), dtype=torch.float32, device='cuda')
            
            with autocast(dtype=dtype):
                output = model(dummy_input)
                loss = criterion(output, dummy_target)
            
            loss.backward()
            optimizer.step()
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"✓ SUCCESS ({memory_used:.2f} GB)")
            max_working_batch = batch_size
            
            del model, optimizer, dummy_input, dummy_target
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ OOM")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    print(f"\n{'='*80}")
    print(f"Maximum batch size: {max_working_batch}")
    print(f"{'='*80}\n")
    
    return max_working_batch


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python benchmark.py compare [batch_size] [image_size]")
        print("  python benchmark.py maxbatch [image_size]")
        print("\nExamples:")
        print("  python benchmark.py compare 16 384")
        print("  python benchmark.py maxbatch 384")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == 'compare':
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        image_size = int(sys.argv[3]) if len(sys.argv) > 3 else 384
        compare_dtypes(batch_size, image_size)
    
    elif command == 'maxbatch':
        image_size = int(sys.argv[2]) if len(sys.argv) > 2 else 384
        find_max_batch_size(image_size)
    
    else:
        print(f"Unknown command: {command}")