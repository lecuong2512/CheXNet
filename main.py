# main.py - Optimized entry point (Merged DDP features)
import os
import torch
from Models.TrainModel import ChexnetTrainer

def main():
    """Main entry point with enhanced configuration"""
    print("\n" + "="*80)
    print("ChestX-ray14 Classification - ConvNeXtV2-Large")
    # === [SỬA] === Tiêu đề từ Đoạn 1
    print("Multi-GPU Optimized | DistributedDataParallel | Progressive Fine-tuning")
    print("="*80 + "\n")
    
    # Choose mode
    mode = input("Select mode [train/test/resume]: ").strip().lower()
    
    if mode == 'train':
        runTrain()
    elif mode == 'test':
        runTest()
    elif mode == 'resume':
        runResume()
    else:
        print("❌ Invalid mode. Use 'train', 'test', or 'resume'")

def runTrain():
    """Train from scratch or pretrained with optimal configuration"""
    print("\n[MODE] Training from scratch")
    
    # ---- Auto-detect optimal configuration
    try:
        from Models.config import TensorCoreConfig
        
        # Ask for image size first
        print("\n📐 Select image size:")
        print("  1. 224x224 (fastest, lower accuracy)")
        print("  2. 384x384 (balanced, recommended)")
        print("  3. 512x512 (slowest, highest accuracy)")
        
        size_choice = input("Choice [1/2/3]: ").strip() or "2"
        image_sizes = {'1': 224, '2': 384, '3': 512}
        imgtransCrop = image_sizes.get(size_choice, 384)
        
        config = TensorCoreConfig.get_optimal_config(image_size=imgtransCrop)
        TensorCoreConfig.print_config(config)
        
        # Ask user to confirm or customize
        use_optimal = input("\nUse recommended settings? [Y/n]: ").strip().lower()
        
        # === [SỬA] === Logic Batch Size từ Đoạn 1 (PER GPU)
        if use_optimal != 'n':
            # Batch size PER GPU (not total)
            trBatchSize = config['batch_size']
            num_gpus = max(1, torch.cuda.device_count())
            use_compile = config.get('use_compile', False)
            
            print(f"\n✓ Batch size per GPU: {trBatchSize}")
            print(f"✓ Number of GPUs: {num_gpus}")
            print(f"✓ Effective batch size: {trBatchSize * num_gpus}")
            print(f"✓ Image size: {imgtransCrop}x{imgtransCrop}")
            if use_compile:
               print(f"✓ torch.compile: Enabled (20-30% speedup expected)")
        else:
            trBatchSize = int(input("Enter batch size PER GPU: "))
            imgtransCrop = int(input(f"Enter image size [current: {imgtransCrop}]: ") or imgtransCrop)
            use_compile = input("Use torch.compile? [y/n]: ").strip().lower() == 'y'
        # === [HẾT SỬA] ===
            
    except ImportError:
        # Fallback if config module not available
        trBatchSize = 16
        imgtransCrop = 384
        use_compile = False
        print("\n⚠ Auto-config not available, using defaults")
    
    # ---- Advanced options
    print("\n⚙️ Advanced options:")
    use_class_weights = input("  Use class weights for imbalanced data? [Y/n]: ").strip().lower() != 'n'
    
    # === [SỬA] === Logic Tinh chỉnh từ Đoạn 1
    print("\n🔧 Progressive Fine-tuning Strategy:")
    print("  Phase 1 (Epoch 0-N): Train ONLY classifier (backbone frozen)")
    print("  Phase 2 (Epoch N+): Unfreeze backbone, fine-tune entire model with lower LR")
    fine_tune_epoch = int(input("  Unfreeze backbone at epoch [30]: ").strip() or "30")
    
    # Giữ lại tùy chọn 'use_preload' từ Đoạn 2
    use_preload = input("  Preload all images to RAM? (faster but needs ~50GB RAM) [y/N]: ").strip().lower() == 'y'
    
    # ---- Paths
    pathDirData = input("\nImage directory [CheXNet/Database]: ").strip() or 'CheXNet/Database'
    pathFileTrain = input("Train list [CheXNet/Dataset/train_list.txt]: ").strip() or 'CheXNet/Dataset/train_list.txt'
    pathFileVal = input("Val list [CheXNet/Dataset/val_list.txt]: ").strip() or 'CheXNet/Dataset/val_list.txt'
    
    # ---- Hyperparameters
    nnIsTrained = True  # Use ImageNet pretrained weights
    nnClassCount = 15   # 14 diseases + 1 "No Finding"
    trMaxEpoch = int(input("\nMax epochs [100]: ").strip() or "100")
    
    # ---- Model paths
    pathModel = input("Model save path [CheXNet/Trainedmodel/chexnetmodel.pth]: ").strip() or 'CheXNet/Trainedmodel/chexnetmodel.pth'
    
    print(f"\n📋 Final Configuration:")
    print(f"  Classes: {nnClassCount}")
    # === [SỬA] === Cập nhật in thông tin batch size
    print(f"  Batch size per GPU: {trBatchSize}")
    print(f"  Total GPUs: {torch.cuda.device_count()}")
    print(f"  Effective batch size: {trBatchSize * max(1, torch.cuda.device_count())}")
    # === [HẾT SỬA] ===
    print(f"  Max epochs: {trMaxEpoch}")
    print(f"  Image size: {imgtransCrop}x{imgtransCrop}")
    print(f"  Pretrained: {nnIsTrained}")
    print(f"  Class weights: {use_class_weights}")
    print(f"  Fine-tuning strategy:")
    print(f"    - Phase 1 (0-{fine_tune_epoch}): Classifier only")
    print(f"    - Phase 2 ({fine_tune_epoch}+): Full model (backbone LR: 5e-5, classifier LR: 1e-4)")
    print(f"  torch.compile: {use_compile}")
    print(f"  Preload images: {use_preload}") # Giữ lại từ Đoạn 2
    
    # Estimate training time
    try:
        from Models.config import estimate_training_time
        time_hours, breakdown = estimate_training_time(
            dataset_size=86524,  # Approximate
            # === [SỬA] === Tính effective batch size
            batch_size=trBatchSize * max(1, torch.cuda.device_count()),
            # === [HẾT SỬA] ===
            epochs=trMaxEpoch,
            architecture=config.get('architecture'),
            dtype=config.get('dtype')
        )
        print(f"\n⏱️ Estimated training time:")
        print(f"  Total: ~{time_hours:.1f} hours ({breakdown['total_days']:.1f} days)")
        print(f"  Per epoch: ~{breakdown['hours_per_epoch']:.1f} hours")
    except:
        pass
    
    confirm = input("\n▶️ Start training? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("Training cancelled.")
        return
    
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)

    # === [THÊM] === Hướng dẫn Đa GPU (DDP) từ Đoạn 1
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\n💡 TIP: For multi-GPU training, use:")
        print(f"   torchrun --nproc_per_node={num_gpus} main.py")
        print(f"\nOr set environment variables manually:")
        print(f"   RANK=0 WORLD_SIZE={num_gpus} LOCAL_RANK=0 python main.py")
        print("\nContinuing with single-process mode...")
    # === [HẾT THÊM] ===
    
    # Train
    ChexnetTrainer.train(
        pathDirData, pathFileTrain, pathFileVal,
        nnIsTrained, nnClassCount,
        trBatchSize, trMaxEpoch,
        imgtransCrop, pathModel,
        checkpoint=None,
        start_epoch=0,
        use_class_weights=use_class_weights,
        fine_tune_epoch=fine_tune_epoch,
        use_compile=use_compile
    )
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    
    # Test after training
    test_now = input("\nRun testing now? [y/n]: ").strip().lower()
    if test_now == 'y':
        runTest(pathModel=pathModel, image_size=imgtransCrop)

# === [THAY THẾ] === Toàn bộ hàm runResume từ Đoạn 1
def runResume():
    """Resume training from checkpoint"""
    print("\n[MODE] Resume training from checkpoint")
    
    # Model path
    pathModel = input("Checkpoint path [CheXNet/Trainedmodel/chexnetmodel.pth]: ").strip() or 'CheXNet/Trainedmodel/chexnetmodel.pth'
    
    if not os.path.exists(pathModel):
        print(f"\n❌ Checkpoint not found: {pathModel}")
        return
    
    # Load checkpoint info
    ckpt = torch.load(pathModel, map_location='cpu', weights_only=False)
    start_epoch = ckpt.get('epoch', 0)
    best_auroc = ckpt.get('best_auroc', 0.0)
    config = ckpt.get('config', {})
    
    print(f"\n📊 Checkpoint info:")
    print(f"  Epoch: {start_epoch}")
    print(f"  Best AUROC: {best_auroc:.4f}")
    print(f"  Image size: {config.get('image_size', 'unknown')}")
    print(f"  Num classes: {config.get('num_classes', 'unknown')}")
    print(f"  Fine-tune epoch: {config.get('fine_tune_epoch', 30)}")
    
    # Paths
    pathDirData = input("\nImage directory [CheXNet/Database]: ").strip() or 'CheXNet/Database'
    pathFileTrain = input("Train list [CheXNet/Dataset/train_list.txt]: ").strip() or 'CheXNet/Dataset/train_list.txt'
    pathFileVal = input("Val list [CheXNet/Dataset/val_list.txt]: ").strip() or 'CheXNet/Dataset/val_list.txt'
    
    # Hyperparameters
    nnIsTrained = True
    nnClassCount = config.get('num_classes', 15)
    imgtransCrop = config.get('image_size', 384)
    trMaxEpoch = int(input(f"Continue until epoch [100]: ").strip() or "100")
    
    # Batch size - NOW CONFIGURABLE (Logic từ Đoạn 1)
    try:
        from Models.config import TensorCoreConfig
        auto_config = TensorCoreConfig.get_optimal_config(image_size=imgtransCrop)
        suggested_batch = auto_config['batch_size']
        print(f"\n💡 Suggested batch size per GPU: {suggested_batch}")
    except:
        suggested_batch = 16
    
    trBatchSize = int(input(f"Batch size per GPU [{suggested_batch}]: ").strip() or str(suggested_batch))
    
    # Advanced options (Logic linh hoạt từ Đoạn 1)
    use_class_weights = input("\nUse class weights? [Y/n]: ").strip().lower() != 'n'
    fine_tune_epoch = config.get('fine_tune_epoch', 30)
    
    print(f"\n  Progressive fine-tuning at epoch: {fine_tune_epoch}")
    change_fine_tune = input("  Change fine-tune epoch? [y/N]: ").strip().lower() == 'y'
    if change_fine_tune:
        fine_tune_epoch = int(input(f"  Unfreeze backbone at epoch [{fine_tune_epoch}]: ").strip() or str(fine_tune_epoch))
    
    use_compile = input("  Use torch.compile? [Y/n]: ").strip().lower() != 'n'
    
    print(f"\n▶️ Resuming from epoch {start_epoch}...")
    print(f"  Batch size per GPU: {trBatchSize}")
    print(f"  Total GPUs: {torch.cuda.device_count()}")
    print(f"  Fine-tune epoch: {fine_tune_epoch}")
    
    # Resume training
    ChexnetTrainer.train(
        pathDirData, pathFileTrain, pathFileVal,
        nnIsTrained, nnClassCount,
        trBatchSize, trMaxEpoch,
        imgtransCrop, pathModel,
        checkpoint=pathModel,
        start_epoch=start_epoch,
        use_class_weights=use_class_weights,
        fine_tune_epoch=fine_tune_epoch,
        use_compile=use_compile
    )
    
    print("\n✅ Training resumed and completed!")

# === [THAY THẾ] === Toàn bộ hàm runTest từ Đoạn 1
def runTest(pathModel=None, image_size=None):
    """Test trained model with configurable batch size"""
    print("\n[MODE] Testing")
    
    # Paths
    pathDirData = input("Image directory [CheXNet/Database]: ").strip() or 'CheXNet/Database'
    pathFileTest = input("Test list [CheXNet/Dataset/test_list.txt]: ").strip() or 'CheXNet/Dataset/test_list.txt'
    
    if pathModel is None:
        pathModel = input("Model path [CheXNet/Trainedmodel/chexnetmodel.pth]: ").strip() or 'CheXNet/Trainedmodel/chexnetmodel.pth'
    
    if not os.path.exists(pathModel):
        print(f"\n❌ Model not found: {pathModel}")
        print("Please train the model first.")
        return
    
    # Load config from checkpoint
    ckpt = torch.load(pathModel, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    
    # Parameters
    nnClassCount = config.get('num_classes', 15)
    if image_size is None:
        imgtransCrop = config.get('image_size', 384)
    else:
        imgtransCrop = image_size
    
    # Batch size - NOW CONFIGURABLE (Logic từ Đoạn 1)
    try:
        from Models.config import TensorCoreConfig
        auto_config = TensorCoreConfig.get_optimal_config(image_size=imgtransCrop)
        suggested_batch = auto_config['batch_size'] * 2  # Can use larger batch for inference
        print(f"\n💡 Suggested test batch size: {suggested_batch}")
    except:
        suggested_batch = 32
    
    trBatchSize = int(input(f"Batch size [{suggested_batch}]: ").strip() or str(suggested_batch))
    
    # Test-Time Augmentation
    use_tta = input("\nUse Test-Time Augmentation (TTA)? [y/N]: ").strip().lower() == 'y'
    num_tta = 5
    if use_tta:
        num_tta = int(input("Number of TTA augmentations [5]: ").strip() or "5")
        print(f"✓ TTA enabled with {num_tta} augmentations (improves AUROC by ~0.5-1%)")
    
    print(f"\n🔍 Testing model: {pathModel}")
    print(f"   Image size: {imgtransCrop}x{imgtransCrop}")
    print(f"   Batch size: {trBatchSize}")
    print(f"   TTA: {'Enabled' if use_tta else 'Disabled'}")
    
    # Test
    aurocMean, aurocIndividual, allPreds, allTargets = ChexnetTrainer.test(
        pathDirData, pathFileTest, pathModel,
        nnClassCount, trBatchSize, imgtransCrop,
        use_tta=use_tta,
        num_tta=num_tta
    )
    
    # Visualize results
    try:
        visualize = input("\n📊 Generate visualization plots? [Y/n]: ").strip().lower()
        if visualize != 'n':
            from Models.visualize import plot_results
            save_dir = f'CheXNet/Results/test_{"tta" if use_tta else "standard"}'
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"\n📈 Generating visualizations...")
            plot_results(
                allTargets, allPreds, aurocIndividual,
                ChexnetTrainer.CLASS_NAMES[:nnClassCount],
                save_dir=save_dir
            )
            print(f"✓ Plots saved to: {save_dir}")
    except ImportError:
        print("⚠ Visualization module not found.")
    except Exception as e:
        print(f"⚠ Error generating plots: {e}")
    
    # Generate heatmaps
    try:
        heatmap = input("\n🔥 Generate Grad-CAM heatmaps? [y/N]: ").strip().lower()
        if heatmap == 'y':
            from Models.head_map import generate_multi_class_heatmap
            
            sample_image = input("Enter image path (or press Enter for default): ").strip()
            if not sample_image:
                sample_image = 'CheXNet/Database/images_001/images/00000001_000.png'
            
            if os.path.exists(sample_image):
                print(f"\n🔥 Generating heatmaps for: {sample_image}")
                generate_multi_class_heatmap(
                    pathModel, sample_image,
                    ChexnetTrainer.CLASS_NAMES[:nnClassCount],
                    num_classes=nnClassCount,
                    image_size=imgtransCrop,
                    top_k=5,
                    save_dir='CheXNet/Heatmaps'
                )
            else:
                print(f"⚠ Image not found: {sample_image}")
    except ImportError:
        print("⚠ Heatmap module not found.")
    except Exception as e:
        print(f"⚠ Error generating heatmaps: {e}")
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETED!")
    print(f"🎯 Final Mean AUROC: {aurocMean:.4f}")
    if use_tta:
        print(f"   (with {num_tta} Test-Time Augmentations)")
    print("="*80)
# === [HẾT THAY THẾ] ===

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
