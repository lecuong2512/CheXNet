def runTrain():
    """Train from scratch or pretrained"""
    print("\n[MODE] Training from scratch")
    
    # ---- Auto-detect optimal configuration
    from Models.config import TensorCoreConfig
    
    config = TensorCoreConfig.get_optimal_config(image_size=384)
    TensorCoreConfig.print_config(config)
    
    # Ask user to confirm or customize
    use_optimal = input("Use recommended settings? [y/n]: ").strip().lower()
    
    if use_optimal == 'y':
        trBatchSize = config['batch_size'] * max(1, torch.cuda.device_count())
        imgtransCrop = 384
        print(f"\n✓ Using optimal batch size: {trBatchSize} (total across GPUs)")
    else:
        trBatchSize = int(input("Enter batch size (total): "))
        imgtransCrop = int(input("Enter image size [384/512]: "))
    
    # ---- Paths
    pathDirData = 'CheXNet/Database'
    pathFileTrain = 'CheXNet/Dataset/train_list.txt'
    pathFileVal = 'CheXNet/Dataset/val_list.txt'
    
    # ---- Hyperparameters
    nnIsTrained = True  # Use ImageNet pretrained weights
    nnClassCount = 15   # 14 diseases + 1 "No Finding"
    trMaxEpoch = 100
    
    # ---- Model paths
    pathModel = 'CheXNet/Trainedmodel/chexnetmodel.pth'
    
    print(f"\nConfiguration:")
    print(f"  Classes: {nnClassCount}")
    print(f"  # main.py")
import os
from Models.TrainModel import ChexnetTrainer

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("ChestX-ray14 Classification - ConvNeXtV2-Large")
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
        print("Invalid mode. Use 'train', 'test', or 'resume'")

def runTrain():
    """Train from scratch or pretrained"""
    import torch
    print("\n[MODE] Training from scratch")
    
    # ---- Auto-detect optimal configuration
    try:
        from Models.config import TensorCoreConfig
        
        config = TensorCoreConfig.get_optimal_config(image_size=384)
        TensorCoreConfig.print_config(config)
        
        # Ask user to confirm or customize
        use_optimal = input("Use recommended settings? [y/n]: ").strip().lower()
        
        if use_optimal == 'y':
            trBatchSize = config['batch_size'] * max(1, torch.cuda.device_count())
            imgtransCrop = 384
            print(f"\n✓ Using optimal batch size: {trBatchSize} (total across {torch.cuda.device_count()} GPUs)")
        else:
            trBatchSize = int(input("Enter batch size (total): "))
            imgtransCrop = int(input("Enter image size [384/512]: ") or "384")
    except ImportError:
        # Fallback if config module not available
        trBatchSize = 16
        imgtransCrop = 384
        print("\n⚠ Auto-config not available, using defaults")
    
    # ---- Paths
    pathDirData = 'CheXNet/Database'
    pathFileTrain = 'CheXNet/Dataset/train_list.txt'
    pathFileVal = 'CheXNet/Dataset/val_list.txt'
    
    # ---- Hyperparameters
    nnIsTrained = True  # Use ImageNet pretrained weights
    nnClassCount = 15   # 14 diseases + 1 "No Finding"
    trMaxEpoch = 100
    
    # ---- Model paths
    pathModel = 'CheXNet/Trainedmodel/chexnetmodel.pth'
    
    print(f"\nFinal Configuration:")
    print(f"  Classes: {nnClassCount}")
    print(f"  Batch size: {trBatchSize}")
    print(f"  Max epochs: {trMaxEpoch}")
    print(f"  Image size: {imgtransCrop}x{imgtransCrop}")
    print(f"  Pretrained: {nnIsTrained}")
    
    # Train
    ChexnetTrainer.train(
        pathDirData, pathFileTrain, pathFileVal,
        nnIsTrained, nnClassCount,
        trBatchSize, trMaxEpoch,
        imgtransCrop, pathModel,
        checkpoint=None,
        start_epoch=0
    )
    
    print("\n[DONE] Training completed!")
    
    # Test after training
    test_now = input("\nRun testing now? [y/n]: ").strip().lower()
    if test_now == 'y':
        runTest()

def runResume():
    """Resume training from checkpoint"""
    print("\n[MODE] Resume training from checkpoint")
    
    # ---- Paths
    pathDirData = 'CheXNet/Database'
    pathFileTrain = 'CheXNet/Dataset/train_list.txt'
    pathFileVal = 'CheXNet/Dataset/val_list.txt'
    
    # ---- Hyperparameters
    nnIsTrained = True
    nnClassCount = 15
    trBatchSize = 16
    trMaxEpoch = 100
    imgtransCrop = 384
    
    # ---- Model paths
    pathModel = 'CheXNet/Trainedmodel/chexnetmodel.pth'
    checkpoint = pathModel  # Resume from this checkpoint
    
    # Get start epoch
    if os.path.exists(checkpoint):
        import torch
        ckpt = torch.load(checkpoint, map_location='cpu')
        start_epoch = ckpt.get('epoch', 0)
        print(f"\nResuming from epoch {start_epoch}")
    else:
        print(f"\nCheckpoint not found: {checkpoint}")
        return
    
    # Resume training
    ChexnetTrainer.train(
        pathDirData, pathFileTrain, pathFileVal,
        nnIsTrained, nnClassCount,
        trBatchSize, trMaxEpoch,
        imgtransCrop, pathModel,
        checkpoint=checkpoint,
        start_epoch=start_epoch
    )
    
    print("\n[DONE] Training completed!")

def runTest():
    """Test trained model"""
    print("\n[MODE] Testing")
    
    # ---- Paths
    pathDirData = 'CheXNet/Database'
    pathFileTest = 'CheXNet/Dataset/test_list.txt'
    pathModel = 'CheXNet/Trainedmodel/chexnetmodel.pth'
    
    # ---- Parameters
    nnClassCount = 15
    trBatchSize = 32  # Larger batch for testing
    imgtransCrop = 384
    
    if not os.path.exists(pathModel):
        print(f"\nError: Model not found at {pathModel}")
        print("Please train the model first.")
        return
    
    print(f"\nTesting model: {pathModel}")
    
    # Test
    aurocMean, aurocIndividual, allPreds, allTargets = ChexnetTrainer.test(
        pathDirData, pathFileTest, pathModel,
        nnClassCount, trBatchSize, imgtransCrop
    )
    
    # Visualize results
    try:
        visualize = input("\nGenerate visualization plots? [y/n]: ").strip().lower()
        if visualize == 'y':
            from Models.visualize import plot_results
            plot_results(allTargets, allPreds, aurocIndividual, 
                        ChexnetTrainer.CLASS_NAMES[:nnClassCount])
    except ImportError:
        print("Visualization module not found. Skipping plots.")
    
    print("\n[DONE] Testing completed!")

if __name__ == '__main__':
    main()
