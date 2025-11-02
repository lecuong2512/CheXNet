# main.py - Optimized entry point
import os
import torch
from Models.TrainModel import ChexnetTrainer

def main():
    """Main entry point with enhanced configuration"""
    print("\n" + "="*80)
    print("ChestX-ray14 Classification - ConvNeXtV2-Large")
    print("Multi-GPU Optimized | Tensor Core Accelerated")
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
        use_optimal = input("\nUse recommended settings? [y/n]: ").strip().lower()
        
        if use_optimal == 'y':
            trBatchSize = config['batch_size'] * max(1, torch.cuda.device_count())
            num_workers = config['num_workers']
            use_compile = config.get('use_compile', False)
            print(f"\n✓ Using optimal batch size: {trBatchSize} (total across {torch.cuda.device_count()} GPUs)")
            print(f"✓ Num workers: {num_workers}")
            print(f"✓ Image size: {imgtransCrop}x{imgtransCrop}")
            if use_compile:
                print(f"✓ torch.compile: Enabled (20-30% speedup expected)")
        else:
            trBatchSize = int(input("Enter total batch size: "))
            imgtransCrop = int(input(f"Enter image size [current: {imgtransCrop}]: ") or imgtransCrop)
            use_compile = input("Use torch.compile? [y/n]: ").strip().lower() == 'y'
            
    except ImportError:
        # Fallback if config module not available
        trBatchSize = 16
        imgtransCrop = 384
        use_compile = False
        print("\n⚠ Auto-config not available, using defaults")
    
    # ---- Advanced options
    print("\n⚙️ Advanced options:")
    use_class_weights = input("  Use class weights for imbalanced data? [Y/n]: ").strip().lower() != 'n'
    fine_tune_epoch = int(input("  Unfreeze backbone at epoch [default: 30]: ").strip() or "30")
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
    print(f"  Batch size: {trBatchSize}")
    print(f"  Max epochs: {trMaxEpoch}")
    print(f"  Image size: {imgtransCrop}x{imgtransCrop}")
    print(f"  Pretrained: {nnIsTrained}")
    print(f"  Class weights: {use_class_weights}")
    print(f"  Fine-tune at epoch: {fine_tune_epoch}")
    print(f"  torch.compile: {use_compile}")
    print(f"  Preload images: {use_preload}")
    
    # Estimate training time
    try:
        from Models.config import estimate_training_time
        time_hours, breakdown = estimate_training_time(
            dataset_size=86524,  # Approximate
            batch_size=trBatchSize,
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
    
    # Train
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    
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

def runResume():
    """Resume training from checkpoint"""
    print("\n[MODE] Resume training from checkpoint")
    
    # ---- Model path
    pathModel = input("Checkpoint path [CheXNet/Trainedmodel/chexnetmodel.pth]: ").strip() or 'CheXNet/Trainedmodel/chexnetmodel.pth'
    
    if not os.path.exists(pathModel):
        print(f"\n❌ Checkpoint not found: {pathModel}")
        return
    
    # Load checkpoint to get info
    ckpt = torch.load(pathModel, map_location='cpu', weights_only=False)
    start_epoch = ckpt.get('epoch', 0)
    best_auroc = ckpt.get('best_auroc', 0.0)
    config = ckpt.get('config', {})
    
    print(f"\n📊 Checkpoint info:")
    print(f"  Epoch: {start_epoch}")
    print(f"  Best AUROC: {best_auroc:.4f}")
    print(f"  Image size: {config.get('image_size', 'unknown')}")
    print(f"  Num classes: {config.get('num_classes', 'unknown')}")
    
    # ---- Paths
    pathDirData = input("\nImage directory [CheXNet/Database]: ").strip() or 'CheXNet/Database'
    pathFileTrain = input("Train list [CheXNet/Dataset/train_list.txt]: ").strip() or 'CheXNet/Dataset/train_list.txt'
    pathFileVal = input("Val list [CheXNet/Dataset/val_list.txt]: ").strip() or 'CheXNet/Dataset/val_list.txt'
    
    # ---- Hyperparameters
    nnIsTrained = True
    nnClassCount = config.get('num_classes', 15)
    imgtransCrop = config.get('image_size', 384)
    trMaxEpoch = int(input(f"Continue until epoch [100]: ").strip() or "100")
    
    # Get batch size
    try:
        from Models.config import TensorCoreConfig
        auto_config = TensorCoreConfig.get_optimal_config(image_size=imgtransCrop)
        trBatchSize = auto_config['batch_size'] * max(1, torch.cuda.device_count())
    except:
        trBatchSize = 16
    
    # Advanced options
    use_class_weights = input("\nUse class weights? [Y/n]: ").strip().lower() != 'n'
    fine_tune_epoch = int(input("Fine-tune epoch [30]: ").strip() or "30")
    use_compile = input("Use torch.compile? [y/N]: ").strip().lower() == 'y'
    
    print(f"\n▶️ Resuming from epoch {start_epoch}...")
    
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

def runTest(pathModel=None, image_size=None):
    """Test trained model"""
    print("\n[MODE] Testing")
    
    # ---- Paths
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
    
    # ---- Parameters
    nnClassCount = config.get('num_classes', 15)
    if image_size is None:
        imgtransCrop = config.get('image_size', 384)
    else:
        imgtransCrop = image_size
    
    trBatchSize = int(input(f"Batch size [32]: ").strip() or "32")
    
    print(f"\n🔍 Testing configuration:")
    print(f"   Model: {pathModel}")
    print(f"   Image size: {imgtransCrop}x{imgtransCrop}")
    print(f"   Batch size: {trBatchSize}")
    print(f"   Test samples: {pathFileTest}")
    
    confirm = input("\n▶️ Start testing? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("Testing cancelled.")
        return
    
    # Test
    print("\n" + "="*80)
    print("🚀 STARTING TESTING")
    print("="*80)
    
    aurocMean, aurocIndividual, allPreds, allTargets = ChexnetTrainer.test(
        pathDirData, pathFileTest, pathModel,
        nnClassCount, trBatchSize, imgtransCrop
    )
    
    # Visualize results
    try:
        visualize = input("\n📊 Generate visualization plots? [Y/n]: ").strip().lower()
        if visualize != 'n':
            from Models.visualize import plot_results
            save_dir = 'CheXNet/Results/test_results'
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"\n📈 Generating visualizations...")
            plot_results(
                allTargets, allPreds, aurocIndividual, 
                ChexnetTrainer.CLASS_NAMES[:nnClassCount],
                save_dir=save_dir
            )
            print(f"✓ Plots saved to: {save_dir}")
    except ImportError:
        print("⚠ Visualization module not found. Skipping plots.")
    except Exception as e:
        print(f"⚠ Error generating plots: {e}")
    
    # Save predictions
    try:
        save_preds = input("\n💾 Save predictions to file? [y/N]: ").strip().lower()
        if save_preds == 'y':
            import numpy as np
            save_dir = 'CheXNet/Results/predictions'
            os.makedirs(save_dir, exist_ok=True)
            
            np.save(f"{save_dir}/predictions.npy", allPreds)
            np.save(f"{save_dir}/targets.npy", allTargets)
            np.save(f"{save_dir}/auroc_individual.npy", aurocIndividual)
            
            print(f"✓ Predictions saved to: {save_dir}")
            print(f"  - predictions.npy: shape {allPreds.shape}")
            print(f"  - targets.npy: shape {allTargets.shape}")
            print(f"  - auroc_individual.npy: shape {len(aurocIndividual)}")
    except Exception as e:
        print(f"⚠ Error saving predictions: {e}")
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETED!")
    print(f"🎯 Final Mean AUROC: {aurocMean:.4f}")
    print("="*80)
    
    return aurocMean, aurocIndividual, allPreds, allTargets

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
