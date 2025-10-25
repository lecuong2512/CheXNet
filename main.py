"""
CheXNet Enhanced - Multi-Architecture Training & Testing
Supports: DenseNet, ConvNeXtV2, EfficientNetV2, Swin Transformer
Auto-detects GPUs and displays real-time training progress
"""

from Models import ChexnetTrainer

# ============================================================================
# TRAINING EXAMPLES
# ============================================================================

def train_densenet121():
    """Train with DenseNet121 (lightweight, fast)"""
    ChexnetTrainer.train(
        pathDirData='./Database',
        pathFileTrain='./Dataset/train_list.txt',
        pathFileVal='./Dataset/val_list.txt',
        nnIsTrained=True,
        nnClassCount=15,
        trBatchSize=64,
        trMaxEpoch=100,
        transCrop=224,
        pathModel='./Trainedmodel/densenet121_chexnet.pth',
        model_type='densenet121'
    )


def train_convnextv2_large():
    """Train with ConvNeXtV2-Large (high accuracy)"""
    ChexnetTrainer.train(
        pathDirData='./Database',
        pathFileTrain='./Dataset/train_list.txt',
        pathFileVal='./Dataset/val_list.txt',
        nnIsTrained=True,
        nnClassCount=15,
        trBatchSize=32,  # Smaller batch for larger model
        trMaxEpoch=100,
        transCrop=224,
        pathModel='./Trainedmodel/convnextv2_large_chexnet.pth',
        model_type='convnextv2_large'
    )


def train_efficientnetv2_m():
    """Train with EfficientNetV2-Medium (balanced performance)"""
    ChexnetTrainer.train(
        pathDirData='./Database',
        pathFileTrain='./Dataset/train_list.txt',
        pathFileVal='./Dataset/val_list.txt',
        nnIsTrained=True,
        nnClassCount=15,
        trBatchSize=48,
        trMaxEpoch=100,
        transCrop=224,
        pathModel='./Trainedmodel/efficientnetv2_m_chexnet.pth',
        model_type='efficientnetv2_m'
    )


def train_swin_small():
    """Train with Swin Transformer Small (vision transformer)"""
    ChexnetTrainer.train(
        pathDirData='./Database',
        pathFileTrain='./Dataset/train_list.txt',
        pathFileVal='./Dataset/val_list.txt',
        nnIsTrained=True,
        nnClassCount=15,
        trBatchSize=32,
        trMaxEpoch=100,
        transCrop=224,
        pathModel='./Trainedmodel/swin_small_chexnet.pth',
        model_type='swin_small'
    )


def train_with_resume():
    """Resume training from checkpoint"""
    ChexnetTrainer.train(
        pathDirData='./Database',
        pathFileTrain='./Dataset/train_list.txt',
        pathFileVal='./Dataset/val_list.txt',
        nnIsTrained=True,
        nnClassCount=15,
        trBatchSize=32,
        trMaxEpoch=100,
        transCrop=224,
        pathModel='./Trainedmodel/convnextv2_large_chexnet.pth',
        checkpoint='./Trainedmodel/convnextv2_large_chexnet.pth',  # Resume from here
        model_type='convnextv2_large'
    )


def train_with_custom_hyperparameters():
    """Train with custom learning rate and batch size"""
    ChexnetTrainer.train(
        pathDirData='./Database',
        pathFileTrain='./Dataset/train_list.txt',
        pathFileVal='./Dataset/val_list.txt',
        nnIsTrained=True,
        nnClassCount=15,
        trBatchSize=64,
        trMaxEpoch=100,
        transCrop=224,
        pathModel='./Trainedmodel/custom_hyperparams.pth',
        model_type='efficientnetv2_l',
        custom_lr=3e-5,           # Custom learning rate
        custom_batch_size=24      # Custom batch size
    )


# ============================================================================
# TESTING EXAMPLES
# ============================================================================

def test_model():
    """Test any trained model (auto-detects model type)"""
    ChexnetTrainer.test(
        pathDirData='./Database',
        pathFileTest='./Dataset/test_list.txt',
        pathModel='./Trainedmodel/convnextv2_large_chexnet.pth',
        nnClassCount=15,
        trBatchSize=64,
        transCrop=224
        # model_type will be auto-detected from checkpoint
    )


def test_specific_model():
    """Test with explicit model type specification"""
    ChexnetTrainer.test(
        pathDirData='./Database',
        pathFileTest='./Dataset/test_list.txt',
        pathModel='./Trainedmodel/swin_small_chexnet.pth',
        nnClassCount=15,
        trBatchSize=32,
        transCrop=224,
        model_type='swin_small'  # Explicitly specify
    )


def compare_all_models():
    """Compare performance of different architectures"""
    models_to_test = [
        ('densenet121', './Trainedmodel/densenet121_chexnet.pth', 64),
        ('convnextv2_large', './Trainedmodel/convnextv2_large_chexnet.pth', 32),
        ('efficientnetv2_m', './Trainedmodel/efficientnetv2_m_chexnet.pth', 48),
        ('swin_small', './Trainedmodel/swin_small_chexnet.pth', 32),
    ]
    
    results = {}
    
    for model_type, model_path, batch_size in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing {model_type}")
        print(f"{'='*80}")
        
        try:
            mean_auroc, individual_auroc = ChexnetTrainer.test(
                pathDirData='./Database',
                pathFileTest='./Dataset/test_list.txt',
                pathModel=model_path,
                nnClassCount=15,
                trBatchSize=batch_size,
                transCrop=224,
                model_type=model_type
            )
            results[model_type] = mean_auroc
        except FileNotFoundError:
            print(f"⚠️  Model not found: {model_path}")
        except Exception as e:
            print(f"❌ Error testing {model_type}: {e}")
    
    # Print comparison
    print("\n" + "="*80)
    print("🏆 MODEL COMPARISON - Mean AUROC")
    print("="*80)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (model_name, auroc) in enumerate(sorted_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank}. {model_name:25s}: {auroc:.4f}")
    print("="*80 + "\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def show_available_models():
    """Display all supported model architectures"""
    from Models.Model import MultiModelArchitecture
    
    print("\n" + "="*80)
    print("📦 SUPPORTED MODEL ARCHITECTURES")
    print("="*80)
    
    models = MultiModelArchitecture.SUPPORTED_MODELS
    
    categories = {
        'DenseNet': [k for k in models.keys() if k.startswith('densenet')],
        'ConvNeXtV2': [k for k in models.keys() if k.startswith('convnextv2')],
        'EfficientNetV2': [k for k in models.keys() if k.startswith('efficientnet')],
        'Swin Transformer': [k for k in models.keys() if k.startswith('swin')],
    }
    
    for category, model_list in categories.items():
        print(f"\n{category}:")
        for model_name in model_list:
            lr = MultiModelArchitecture.get_recommended_lr(model_name)
            bs = MultiModelArchitecture.get_recommended_batch_size(model_name, 16)
            print(f"  • {model_name:25s} (LR: {lr:.0e}, BS: {bs})")
    
    print("\n" + "="*80 + "\n")


def check_gpu_info():
    """Check available GPU resources"""
    device, gpu_count, gpu_info = ChexnetTrainer.detect_gpus()
    
    if gpu_count > 0:
        print("\n💡 RECOMMENDED CONFIGURATIONS:")
        print("-" * 40)
        
        total_memory = sum(info['memory_gb'] for info in gpu_info)
        
        if total_memory >= 40:
            print("✅ High-end setup - can train largest models:")
            print("   • convnextv2_huge, efficientnetv2_l, swin_base")
            print("   • Batch size: 16-32")
        elif total_memory >= 24:
            print("✅ Mid-high setup - good for most models:")
            print("   • convnextv2_large, efficientnetv2_m, swin_small")
            print("   • Batch size: 24-48")
        elif total_memory >= 12:
            print("✅ Standard setup:")
            print("   • densenet121/169, convnextv2_base, efficientnetv2_s")
            print("   • Batch size: 32-64")
        else:
            print("⚠️  Limited GPU memory:")
            print("   • densenet121, smaller batch sizes recommended")
            print("   • Batch size: 8-16")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Show available models and GPU info
    show_available_models()
    check_gpu_info()
    
    # Choose what to run
    print("\n" + "="*80)
    print("🚀 CHEXNET TRAINING/TESTING")
    print("="*80)
    print("\nSelect operation:")
    print("  1. Train with DenseNet121 (fast, lightweight)")
    print("  2. Train with ConvNeXtV2-Large (high accuracy)")
    print("  3. Train with EfficientNetV2-Medium (balanced)")
    print("  4. Train with Swin Transformer (transformer-based)")
    print("  5. Test trained model")
    print("  6. Compare all models")
    print("  7. Resume training from checkpoint")
    print("  8. Train with custom hyperparameters")
    print()
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Enter choice (1-8): ").strip()
    
    try:
        if choice == '1':
            print("\n🏋️ Starting DenseNet121 training...")
            train_densenet121()
        elif choice == '2':
            print("\n🏋️ Starting ConvNeXtV2-Large training...")
            train_convnextv2_large()
        elif choice == '3':
            print("\n🏋️ Starting EfficientNetV2-Medium training...")
            train_efficientnetv2_m()
        elif choice == '4':
            print("\n🏋️ Starting Swin Transformer training...")
            train_swin_small()
        elif choice == '5':
            print("\n🔍 Starting model testing...")
            test_model()
        elif choice == '6':
            print("\n📊 Comparing all models...")
            compare_all_models()
        elif choice == '7':
            print("\n🔄 Resuming training...")
            train_with_resume()
        elif choice == '8':
            print("\n⚙️  Training with custom hyperparameters...")
            train_with_custom_hyperparameters()
        else:
            print("❌ Invalid choice!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
