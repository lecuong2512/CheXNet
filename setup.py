import subprocess
import sys

def run_cmd(command, critical=True):
    """Chạy 1 lệnh shell và dừng nếu lỗi (nếu critical=True)."""
    print(f"\n🔧 Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Lỗi khi chạy lệnh: {command}")
        if critical:
            sys.exit(result.returncode)
        else:
            print("⚠️ Bỏ qua lỗi (non-critical).")
    else:
        print("✅ Hoàn thành.")

def main():
    print("=" * 70)
    print("🚀 CÀI ĐẶT MÔI TRƯỜNG CHEXNET (PYTORCH + TENSORFLOW, SSH SAFE)")
    print("=" * 70)

    # 1️⃣ Cập nhật pip
    run_cmd("python3 -m pip install --upgrade pip")

    # 2️⃣ Cài đặt PyTorch (CUDA 12.1 — nếu máy bạn CUDA 11.x thì đổi thành cu118)
    print("\n📦 Cài đặt PyTorch + torchvision + torchaudio (CUDA 12.1)...")
    run_cmd("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    # 3️⃣ Cài TensorFlow (bản GPU nếu có GPU)
    print("\n📦 Cài đặt TensorFlow (GPU nếu tương thích)...")
    run_cmd("pip install tensorflow==2.16.1")

    # 4️⃣ Cài các thư viện phụ trợ
    libs = [
        "tqdm",
        "scikit-learn",
        "numpy",
        "pandas",
        "matplotlib",
        "opencv-python",
        "Pillow",
        "timm"
    ]
    for lib in libs:
        run_cmd(f"pip install {lib}")

    # 5️⃣ Kiểm tra PyTorch & TensorFlow
    try:
        import torch
        print("\n🧠 PyTorch:")
        print(f"  - Phiên bản: {torch.__version__}")
        print(f"  - CUDA khả dụng: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ Lỗi kiểm tra PyTorch: {e}")

    try:
        import tensorflow as tf
        print("\n🧠 TensorFlow:")
        print(f"  - Phiên bản: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  - GPU khả dụng: {gpus[0].name}")
        else:
            print("  ⚠️ TensorFlow không phát hiện GPU.")
    except Exception as e:
        print(f"❌ Lỗi kiểm tra TensorFlow: {e}")

    print("\n✅ Hoàn tất cài đặt môi trường CheXNet (PyTorch + TensorFlow)!")

if __name__ == "__main__":
    main()
