import os
import sys
import argparse
import time

# Thêm đường dẫn gốc của dự án vào sys.path để import được các module
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
try:
    from Backend.cbir import SimilarCaseFinder
except ImportError as e:
    print(f"Lỗi khi import SimilarCaseFinder: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Xây dựng cơ sở dữ liệu FAISS index cho chức năng CBIR")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng ảnh để kiểm thử")
    parser.add_argument("--checkpoint", type=str, default="Trainedmodel/hybrid_model_light.pth", 
                        help="Đường dẫn đến model checkpoint")
    parser.add_argument("--dataset", type=str, default="Dataset/train_list.csv", 
                        help="Đường dẫn đến file danh sách ảnh CSV")
    parser.add_argument("--image_root", type=str, default=r"D:\NCKH\Database_512", 
                        help="Đường dẫn đến thư mục chứa ảnh")
    args = parser.parse_args()

    checkpoint_path = os.path.join(project_root, args.checkpoint)
    dataset_csv_path = os.path.join(project_root, args.dataset)
    image_root = os.path.join(project_root, args.image_root)
    output_dir = os.path.join(backend_dir, "data")
    
    print("Cấu hình xây dựng index:")
    print(f"- Model checkpoint: {checkpoint_path}")
    print(f"- Dataset CSV: {dataset_csv_path}")
    print(f"- Thư mục ảnh: {image_root}")
    print(f"- Thư mục đầu ra: {output_dir}")
    if args.limit:
        print(f"- Giới hạn số lượng ảnh: {args.limit}")

    # Khởi tạo mô hình
    try:
        import torch
    except ImportError:
        print("Lỗi: Yêu cầu cài đặt PyTorch để tạo index.")
        sys.exit(1)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")
    
    try:
        from Models.Model import HybridCNNViTModel
        print(f"Đang tải mô hình với 15 classes...")
        
        # Hàm hỗ trợ xử lý state_dict
        def clean_state_dict(state_dict):
            from collections import OrderedDict
            cleaned = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                cleaned[name] = v
            return cleaned
            
        if os.path.exists(checkpoint_path):
            print(f"Đang tải trọng số mô hình từ {checkpoint_path}...")
            # Load with mmap=True for safety like main.py
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
            
            # Extract info
            model_size = checkpoint.get('model_size', 'large')
            img_size = checkpoint.get('img_size', 384)
            
            # Init model
            model = HybridCNNViTModel(num_classes=15, model_size=model_size, img_size=img_size, pretrained=False)
            
            # Get state dict and clean it
            state_dict = checkpoint.get('state_dict', checkpoint)
            cleaned_state_dict = clean_state_dict(state_dict)
            
            # Load weights
            model.load_state_dict(cleaned_state_dict, strict=False)
        else:
            print(f"Cảnh báo: Không tìm thấy checkpoint tại {checkpoint_path}. Sẽ sử dụng trọng số khởi tạo ngẫu nhiên.")
            model = HybridCNNViTModel(num_classes=15)
    except Exception as e:
        print(f"Lỗi khi khởi tạo mô hình: {e}")
        print("Đảm bảo HybridCNNViTModel có sẵn trong sys.path hoặc có kiến trúc đúng.")
        sys.exit(1)

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Gọi hàm xây dựng index
    print("Bắt đầu tạo index...")
    SimilarCaseFinder.build_index(
        model=model,
        dataset_csv_path=dataset_csv_path,
        image_root=image_root,
        output_dir=output_dir,
        batch_size=32,
        device=device,
        limit=args.limit
    )
    
if __name__ == "__main__":
    main()
