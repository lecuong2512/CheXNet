import subprocess
import sys
import os
import platform

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
    return result.returncode == 0

def detect_cuda_version():
    """Tự động phát hiện CUDA version nếu có."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            # Tìm dòng chứa "release"
            for line in output.split('\n'):
                if 'release' in line.lower():
                    # Extract version như 12.1
                    import re
                    match = re.search(r'release (\d+\.\d+)', line)
                    if match:
                        version = match.group(1)
                        major = version.split('.')[0]
                        print(f"✅ Detected CUDA version: {version}")
                        return major
    except FileNotFoundError:
        pass
    
    print("⚠️ CUDA không được phát hiện hoặc nvcc không có trong PATH")
    return None

def get_pytorch_install_command(cuda_version=None):
    """Lấy lệnh cài đặt PyTorch phù hợp."""
    if cuda_version is None:
        print("📦 Cài đặt PyTorch CPU version...")
        return "pip install torch torchvision torchaudio"
    
    cuda_major = str(cuda_version)
    
    # Map CUDA version to PyTorch CUDA version
    cuda_map = {
        '12': 'cu121',
        '11': 'cu118',
    }
    
    cu_version = cuda_map.get(cuda_major, 'cu121')
    print(f"📦 Cài đặt PyTorch với CUDA {cuda_major}.x ({cu_version})...")
    
    return f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cu_version}"

def check_installation():
    """Kiểm tra các thư viện đã cài đặt."""
    print("\n" + "="*70)
    print("🔍 KIỂM TRA CÀI ĐẶT")
    print("="*70)
    
    # Check PyTorch
    try:
        import torch
        print("\n✅ PyTorch:")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024**3)
                print(f"           Memory: {mem_gb:.1f} GB")
    except ImportError:
        print("❌ PyTorch chưa được cài đặt!")
    except Exception as e:
        print(f"⚠️ Lỗi kiểm tra PyTorch: {e}")
    
    # Check TensorFlow (optional)
    try:
        import tensorflow as tf
        print("\n✅ TensorFlow:")
        print(f"   Version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   GPU available: Yes ({len(gpus)} device(s))")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("   GPU available: No (CPU only)")
    except ImportError:
        print("⚠️ TensorFlow không được cài đặt (optional)")
    except Exception as e:
        print(f"⚠️ Lỗi kiểm tra TensorFlow: {e}")
    
    # Check timm
    try:
        import timm
        print(f"\n✅ timm: {timm.__version__}")
    except ImportError:
        print("❌ timm chưa được cài đặt!")
    
    # Check other libraries
    required_libs = [
        'numpy', 'pandas', 'sklearn', 'tqdm', 
        'matplotlib', 'cv2', 'PIL'
    ]
    
    print("\n📚 Các thư viện khác:")
    for lib_name in required_libs:
        try:
            if lib_name == 'sklearn':
                import sklearn
                lib = sklearn
            elif lib_name == 'cv2':
                import cv2
                lib = cv2
            elif lib_name == 'PIL':
                import PIL
                lib = PIL
            else:
                lib = __import__(lib_name)
            
            version = getattr(lib, '__version__', 'unknown')
            print(f"   ✅ {lib_name}: {version}")
        except ImportError:
            print(f"   ❌ {lib_name}: Not installed")

def main():
    print("="*70)
    print("🚀 CÀI ĐẶT MÔI TRƯỜNG CHEXNET")
    print("="*70)
    
    # System info
    print(f"\n💻 System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Cần Python 3.8 trở lên!")
        sys.exit(1)
    
    # Ask for installation mode
    print("\n" + "="*70)
    print("Chọn chế độ cài đặt:")
    print("  1. Auto (tự động phát hiện CUDA)")
    print("  2. CUDA 12.x")
    print("  3. CUDA 11.x")
    print("  4. CPU only (không có GPU)")
    print("  5. Skip PyTorch (đã cài rồi)")
    print("="*70)
    
    choice = input("\nNhập lựa chọn (1-5) [mặc định: 1]: ").strip() or '1'
    
    # 1️⃣ Upgrade pip
    print("\n" + "="*70)
    print("📦 BƯỚC 1: Nâng cấp pip")
    print("="*70)
    run_cmd(f"{sys.executable} -m pip install --upgrade pip")
    
    # 2️⃣ Install PyTorch
    if choice != '5':
        print("\n" + "="*70)
        print("📦 BƯỚC 2: Cài đặt PyTorch")
        print("="*70)
        
        if choice == '1':
            cuda_version = detect_cuda_version()
            pytorch_cmd = get_pytorch_install_command(cuda_version)
        elif choice == '2':
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        elif choice == '3':
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:  # CPU
            pytorch_cmd = "pip install torch torchvision torchaudio"
        
        run_cmd(pytorch_cmd)
    else:
        print("\n⏭️ Bỏ qua cài đặt PyTorch")
    
    # 3️⃣ Install TensorFlow & Keras (optional)
    print("\n" + "="*70)
    print("📦 BƯỚC 3: Cài đặt TensorFlow & Keras (optional)")
    print("="*70)
    
    tf_choice = input("Cài đặt TensorFlow & Keras? (y/n) [mặc định: n]: ").strip().lower() or 'n'
    
    if tf_choice == 'y':
        # TensorFlow 2.16+ hỗ trợ CUDA 12.x và bao gồm Keras 3
        print("📦 Cài đặt TensorFlow (bao gồm Keras)...")
        run_cmd("pip install tensorflow==2.16.1", critical=False)
        
        # Keras standalone (optional, nếu muốn dùng với backend khác)
        print("📦 Cài đặt Keras 3 (standalone)...")
        run_cmd("pip install keras==3.0.5", critical=False)
    else:
        print("⏭️ Bỏ qua TensorFlow & Keras")
    
    # 4️⃣ Install core libraries
    print("\n" + "="*70)
    print("📦 BƯỚC 4: Cài đặt thư viện cần thiết")
    print("="*70)
    
    core_libs = [
        "timm>=0.9.0",           # Multi-architecture models
        "tqdm>=4.60.0",          # Progress bars
        "scikit-learn>=1.0.0",   # Metrics
        "numpy>=1.21.0",         # Array processing
        "pandas>=1.3.0",         # Data manipulation
    ]
    
    for lib in core_libs:
        run_cmd(f"pip install '{lib}'")
    
    # 5️⃣ Install optional libraries
    print("\n" + "="*70)
    print("📦 BƯỚC 5: Cài đặt thư viện tùy chọn")
    print("="*70)
    
    optional_libs = [
        "matplotlib>=3.3.0",     # Plotting
        "opencv-python>=4.5.0",  # Image processing
        "Pillow>=9.0.0",         # Image I/O
    ]
    
    for lib in optional_libs:
        run_cmd(f"pip install '{lib}'", critical=False)
    
    # 6️⃣ Create directories
    print("\n" + "="*70)
    print("📁 BƯỚC 6: Tạo thư mục cần thiết")
    print("="*70)
    
    dirs = ['Database', 'Dataset', 'Trainedmodel']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"✅ Tạo thư mục: {d}")
        else:
            print(f"📁 Thư mục đã tồn tại: {d}")
    
    # 7️⃣ Final check
    check_installation()
    
    # 8️⃣ Summary
    print("\n" + "="*70)
    print("🎉 HOÀN TẤT CÀI ĐẶT!")
    print("="*70)
    print("\n📝 Các bước tiếp theo:")
    print("  1. Chuẩn bị dữ liệu trong thư mục Database/")
    print("  2. Tạo file train_list.txt, val_list.txt, test_list.txt trong Dataset/")
    print("  3. Chạy: python main.py")
    print("\n💡 Tip: Chạy 'python main.py' để xem các tùy chọn training")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Cài đặt bị hủy bởi người dùng")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
