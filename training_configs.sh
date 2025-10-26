#!/bin/bash

# ============================================================================
# CheXNet Advanced Training Configurations
# ============================================================================

# Common paths
IMAGE_ROOT="CheXNet/Database"
DATASET_DIR="CheXNet/Dataset"
SAVE_DIR="CheXNet/Trainedmodel"
RESULTS_DIR="CheXNet/results"

# ============================================================================
# CONFIG 1: CONSERVATIVE (Safest, Best for small datasets)
# ============================================================================
# Đặc điểm:
# - Freeze lâu (5 epochs)
# - Warmup đầy đủ (5 epochs)
# - Gradual unfreezing chậm (interval=4)
# - High label smoothing
# - Low learning rate
# 
# Phù hợp với:
# - Dataset < 30k images
# - Noisy labels
# - Limited compute budget
# ============================================================================

train_conservative() {
    echo "🛡️  CONSERVATIVE STRATEGY"
    python chexnet_advanced_main.py \
        --mode train \
        --image_root $IMAGE_ROOT \
        --dataset_dir $DATASET_DIR \
        --save_dir "${SAVE_DIR}_conservative" \
        --epochs 50 \
        --batch_size 8 \
        --lr 5e-5 \
        --weight_decay 1e-4 \
        --dropout 0.15 \
        --freeze_epochs 5 \
        --warmup_epochs 5 \
        --gradual_unfreeze \
        --unfreeze_interval 4 \
        --label_smoothing 0.15 \
        --scheduler cosine \
        --num_workers 2
}


# ============================================================================
# CONFIG 2: BALANCED (Recommended default)
# ============================================================================
# Đặc điểm:
# - Freeze vừa phải (3 epochs)
# - Warmup standard (5 epochs)
# - Gradual unfreezing cân bằng (interval=3)
# - Moderate label smoothing
# - Standard learning rate
#
# Phù hợp với:
# - Dataset 30k-80k images (NIH ChestX-ray)
# - Clean/semi-clean labels
# - Standard use case
# ============================================================================

train_balanced() {
    echo "⚖️  BALANCED STRATEGY (RECOMMENDED)"
    python chexnet_advanced_main.py \
        --mode train \
        --image_root $IMAGE_ROOT \
        --dataset_dir $DATASET_DIR \
        --save_dir "${SAVE_DIR}_balanced" \
        --epochs 50 \
        --batch_size 8 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --dropout 0.1 \
        --freeze_epochs 3 \
        --warmup_epochs 5 \
        --gradual_unfreeze \
        --unfreeze_interval 3 \
        --label_smoothing 0.1 \
        --scheduler cosine \
        --num_workers 2
}


# ============================================================================
# CONFIG 3: AGGRESSIVE (Fastest, for large datasets)
# ============================================================================
# Đặc điểm:
# - Freeze ngắn (2 epochs)
# - Warmup ngắn (3 epochs)
# - Gradual unfreezing nhanh (interval=2)
# - Low label smoothing
# - Higher learning rate
#
# Phù hợp với:
# - Dataset > 80k images
# - Clean labels
# - Strong compute resources
# - Tight deadline
# ============================================================================

train_aggressive() {
    echo "🚀 AGGRESSIVE STRATEGY"
    python chexnet_advanced_main.py \
        --mode train \
        --image_root $IMAGE_ROOT \
        --dataset_dir $DATASET_DIR \
        --save_dir "${SAVE_DIR}_aggressive" \
        --epochs 40 \
        --batch_size 16 \
        --lr 2e-4 \
        --weight_decay 5e-5 \
        --dropout 0.05 \
        --freeze_epochs 2 \
        --warmup_epochs 3 \
        --gradual_unfreeze \
        --unfreeze_interval 2 \
        --label_smoothing 0.05 \
        --scheduler cosine \
        --num_workers 4
}


# ============================================================================
# CONFIG 4: FOCAL LOSS (For extreme class imbalance)
# ============================================================================
# Đặc điểm:
# - Sử dụng Focal Loss thay vì BCE
# - Freeze dài để ổn định
# - No label smoothing (Focal Loss đã handle)
# - Standard learning rate
#
# Phù hợp với:
# - Extreme class imbalance (rare diseases)
# - Hard negative mining
# - Classes với very few positive samples
# ============================================================================

train_focal() {
    echo "🎯 FOCAL LOSS STRATEGY"
    python chexnet_advanced_main.py \
        --mode train \
        --image_root $IMAGE_ROOT \
        --dataset_dir $DATASET_DIR \
        --save_dir "${SAVE_DIR}_focal" \
        --epochs 50 \
        --batch_size 8 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --dropout 0.1 \
        --freeze_epochs 5 \
        --warmup_epochs 5 \
        --gradual_unfreeze \
        --unfreeze_interval 3 \
        --use_focal_loss \
        --label_smoothing 0.0 \
        --scheduler cosine \
        --num_workers 2
}


# ============================================================================
# CONFIG 5: ONECYCLE (Fast convergence)
# ============================================================================
# Đặc điểm:
# - OneCycle LR scheduler
# - Không dùng gradual unfreezing (unfreeze hết sớm)
# - Epochs ít hơn (OneCycle converge nhanh)
# - Moderate settings
#
# Phù hợp với:
# - Quick experiments
# - Budget training (few epochs)
# - Need fast results
# ============================================================================

train_onecycle() {
    echo "🔄 ONECYCLE STRATEGY"
    python chexnet_advanced_main.py \
        --mode train \
        --image_root $IMAGE_ROOT \
        --dataset_dir $DATASET_DIR \
        --save_dir "${SAVE_DIR}_onecycle" \
        --epochs 30 \
        --batch_size 8 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --dropout 0.1 \
        --freeze_epochs 3 \
        --warmup_epochs 0 \
        --no_gradual_unfreeze \
        --label_smoothing 0.1 \
        --scheduler onecycle \
        --num_workers 2
}


# ============================================================================
# CONFIG 6: NO FREEZE (Baseline comparison)
# ============================================================================
# Đặc điểm:
# - Không freeze backbone
# - Train tất cả từ đầu
# - Chỉ dùng warmup
# - Baseline để so sánh
#
# Phù hợp với:
# - Benchmark baseline
# - So sánh hiệu quả của freezing
# ⚠️  NOT RECOMMENDED for production
# ============================================================================

train_no_freeze() {
    echo "❌ NO FREEZE STRATEGY (BASELINE)"
    python chexnet_advanced_main.py \
        --mode train \
        --image_root $IMAGE_ROOT \
        --dataset_dir $DATASET_DIR \
        --save_dir "${SAVE_DIR}_no_freeze" \
        --epochs 50 \
        --batch_size 8 \
        --lr 5e-5 \
        --weight_decay 1e-4 \
        --dropout 0.1 \
        --freeze_epochs 0 \
        --warmup_epochs 7 \
        --no_gradual_unfreeze \
        --label_smoothing 0.1 \
        --scheduler cosine \
        --num_workers 2
}


# ============================================================================
# CONFIG 7: MEMORY OPTIMIZED (For limited VRAM)
# ============================================================================
# Đặc điểm:
# - Batch size nhỏ (4)
# - Gradient accumulation simulation
# - Lower memory footprint
#
# Phù hợp với:
# - GPU < 12GB VRAM
# - Tesla T4, GTX 1080 Ti
# ============================================================================

train_memory_optimized() {
    echo "💾 MEMORY OPTIMIZED STRATEGY"
    python chexnet_advanced_main.py \
        --mode train \
        --image_root $IMAGE_ROOT \
        --dataset_dir $DATASET_DIR \
        --save_dir "${SAVE_DIR}_memory_opt" \
        --epochs 50 \
        --batch_size 4 \
        --lr 5e-5 \
        --weight_decay 1e-4 \
        --dropout 0.1 \
        --freeze_epochs 3 \
        --warmup_epochs 5 \
        --gradual_unfreeze \
        --unfreeze_interval 3 \
        --label_smoothing 0.1 \
        --scheduler cosine \
        --num_workers 2
}


# ============================================================================
# CONFIG 8: HIGH PERFORMANCE (For multiple GPUs)
# ============================================================================
# Đặc điểm:
# - Large batch size
# - Higher learning rate (linear scaling)
# - More workers
# - Fast training
#
# Phù hợp với:
# - Multi-GPU setup (2-4 GPUs)
# - A100, V100 GPUs
# ============================================================================

train_high_performance() {
    echo "⚡ HIGH PERFORMANCE STRATEGY"
    python chexnet_advanced_main.py \
        --mode train \
        --image_root $IMAGE_ROOT \
        --dataset_dir $DATASET_DIR \
        --save_dir "${SAVE_DIR}_high_perf" \
        --epochs 40 \
        --batch_size 32 \
        --lr 4e-4 \
        --weight_decay 1e-4 \
        --dropout 0.1 \
        --freeze_epochs 3 \
        --warmup_epochs 5 \
        --gradual_unfreeze \
        --unfreeze_interval 3 \
        --label_smoothing 0.1 \
        --scheduler cosine \
        --num_workers 8
}


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

test_model() {
    local model_dir=$1
    local results_dir=$2
    
    echo "🧪 TESTING MODEL"
    python chexnet_advanced_main.py \
        --mode test \
        --image_root $IMAGE_ROOT \
        --dataset_dir $DATASET_DIR \
        --test_checkpoint "${model_dir}/chexnetmodel.pth" \
        --results_dir $results_dir \
        --batch_size 16 \
        --num_workers 2
}

test_conservative() {
    test_model "${SAVE_DIR}_conservative" "${RESULTS_DIR}_conservative"
}

test_balanced() {
    test_model "${SAVE_DIR}_balanced" "${RESULTS_DIR}_balanced"
}

test_aggressive() {
    test_model "${SAVE_DIR}_aggressive" "${RESULTS_DIR}_aggressive"
}

test_focal() {
    test_model "${SAVE_DIR}_focal" "${RESULTS_DIR}_focal"
}


# ============================================================================
# RESUME TRAINING
# ============================================================================

resume_training() {
    local config_name=$1
    local save_dir="${SAVE_DIR}_${config_name}"
    
    echo "▶️  RESUMING TRAINING: $config_name"
    python chexnet_advanced_main.py \
        --mode train \
        --image_root $IMAGE_ROOT \
        --dataset_dir $DATASET_DIR \
        --save_dir $save_dir \
        --resume "${save_dir}/last_checkpoint.pth" \
        --epochs 50 \
        --num_workers 2
}


# ============================================================================
# COMPARISON EXPERIMENT
# ============================================================================

run_comparison_experiment() {
    echo "🔬 RUNNING COMPARISON EXPERIMENT"
    echo "This will train 4 different strategies and compare results"
    echo ""
    
    # 1. No Freeze (Baseline)
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "EXPERIMENT 1/4: NO FREEZE BASELINE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    train_no_freeze
    test_model "${SAVE_DIR}_no_freeze" "${RESULTS_DIR}_no_freeze"
    
    # 2. Conservative
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "EXPERIMENT 2/4: CONSERVATIVE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    train_conservative
    test_model "${SAVE_DIR}_conservative" "${RESULTS_DIR}_conservative"
    
    # 3. Balanced
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "EXPERIMENT 3/4: BALANCED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    train_balanced
    test_model "${SAVE_DIR}_balanced" "${RESULTS_DIR}_balanced"
    
    # 4. Focal Loss
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "EXPERIMENT 4/4: FOCAL LOSS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    train_focal
    test_model "${SAVE_DIR}_focal" "${RESULTS_DIR}_focal"
    
    # Generate comparison report
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ COMPARISON EXPERIMENT COMPLETED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Results saved in:"
    echo "  - ${RESULTS_DIR}_no_freeze"
    echo "  - ${RESULTS_DIR}_conservative"
    echo "  - ${RESULTS_DIR}_balanced"
    echo "  - ${RESULTS_DIR}_focal"
    echo ""
    echo "Compare test_results.json in each directory to find best strategy"
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

show_menu() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         CheXNet Advanced Training Configuration Menu          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "TRAINING STRATEGIES:"
    echo "  1) Conservative      - Safest, best for small datasets"
    echo "  2) Balanced         - Recommended default (NIH ChestX-ray)"
    echo "  3) Aggressive       - Fastest, for large datasets"
    echo "  4) Focal Loss       - For extreme class imbalance"
    echo "  5) OneCycle         - Fast convergence"
    echo "  6) No Freeze        - Baseline (not recommended)"
    echo "  7) Memory Optimized - For limited VRAM (<12GB)"
    echo "  8) High Performance - For multi-GPU setup"
    echo ""
    echo "TESTING:"
    echo "  11) Test Conservative"
    echo "  12) Test Balanced"
    echo "  13) Test Aggressive"
    echo "  14) Test Focal"
    echo ""
    echo "EXPERIMENTS:"
    echo "  20) Run Comparison Experiment (trains 4 strategies)"
    echo ""
    echo "UTILITIES:"
    echo "  30) Resume Training (specify config name)"
    echo "  31) Show GPU Info"
    echo ""
    echo "  0) Exit"
    echo ""
}

show_gpu_info() {
    echo "🖥️  GPU INFORMATION"
    python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA Available: Yes')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'    Total Memory: {mem_total:.2f} GB')
    capability = torch.cuda.get_device_capability(0)
    print(f'  Compute Capability: {capability[0]}.{capability[1]}')
    print(f'  Tensor Cores: {'Yes' if capability >= (7, 0) else 'No'}')
else:
    print('CUDA Available: No')
    print('WARNING: Training on CPU will be extremely slow!')
"
}


# ============================================================================
# MAIN MENU LOOP
# ============================================================================

main() {
    while true; do
        show_menu
        read -p "Select option: " choice
        
        case $choice in
            1) train_conservative ;;
            2) train_balanced ;;
            3) train_aggressive ;;
            4) train_focal ;;
            5) train_onecycle ;;
            6) train_no_freeze ;;
            7) train_memory_optimized ;;
            8) train_high_performance ;;
            11) test_conservative ;;
            12) test_balanced ;;
            13) test_aggressive ;;
            14) test_focal ;;
            20) run_comparison_experiment ;;
            30) 
                read -p "Enter config name (conservative/balanced/aggressive/focal): " config
                resume_training $config
                ;;
            31) show_gpu_info ;;
            0) 
                echo "Goodbye!"
                exit 0
                ;;
            *)
                echo "❌ Invalid option. Please try again."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    if [ $# -eq 0 ]; then
        # No arguments - show interactive menu
        main
    else
        # Arguments provided - run specific function
        case $1 in
            conservative) train_conservative ;;
            balanced) train_balanced ;;
            aggressive) train_aggressive ;;
            focal) train_focal ;;
            onecycle) train_onecycle ;;
            no_freeze) train_no_freeze ;;
            memory_opt) train_memory_optimized ;;
            high_perf) train_high_performance ;;
            test)
                if [ -z "$2" ]; then
                    echo "Usage: $0 test <config_name>"
                    exit 1
                fi
                test_model "${SAVE_DIR}_$2" "${RESULTS_DIR}_$2"
                ;;
            compare) run_comparison_experiment ;;
            gpu) show_gpu_info ;;
            *)
                echo "Usage: $0 [conservative|balanced|aggressive|focal|onecycle|no_freeze|memory_opt|high_perf|test|compare|gpu]"
                echo "Or run without arguments for interactive menu"
                exit 1
                ;;
        esac
    fi
else
    # Script is being sourced - make functions available
    echo "✅ Training configs loaded. Available functions:"
    echo "  - train_conservative"
    echo "  - train_balanced"
    echo "  - train_aggressive"
    echo "  - train_focal"
    echo "  - train_onecycle"
    echo "  - train_no_freeze"
    echo "  - train_memory_optimized"
    echo "  - train_high_performance"
    echo "  - test_conservative/balanced/aggressive/focal"
    echo "  - run_comparison_experiment"
    echo "  - show_gpu_info"
fi
