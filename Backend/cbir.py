import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss-cpu is not installed. CBIR features will be disabled.")

class EmbeddingDataset:
    """Dataset for FAISS feature extraction"""
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import os
        import torch
        from PIL import Image
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            for i in range(1, 13):
                test_path = os.path.join(self.img_dir, f'images_{i:03d}', 'images', img_name)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
                    
        is_valid = True
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # Ảnh lỗi/không tồn tại: KHÔNG âm thầm trả về ảnh đen rồi để nó bị
            # trích xuất thành 1 embedding "thật" và nhét vào index — embedding
            # của ảnh đen không đại diện cho bất kỳ ca bệnh nào, nhưng trước đây
            # vẫn được coi là hợp lệ và lưu vào FAISS index kèm metadata thật của
            # dòng CSV đó -> tra cứu "ca tương tự" có thể trả về 1 ca hoàn toàn
            # không tương ứng ảnh gốc. Đánh dấu is_valid=False để build_index loại
            # bỏ mẫu này khỏi index thay vì fabricate 1 vector giả.
            print(f"[CBIR] Bỏ qua ảnh lỗi (không đọc được): {img_path} — {e}")
            image = torch.zeros((3, 384, 384))
            is_valid = False

        return image, img_name, is_valid

class SimilarCaseFinder:
    """
    Module Tìm kiếm Dựa trên Nội dung Hình ảnh (CBIR) để tìm các ca X-quang ngực
    tương tự bằng cách sử dụng tìm kiếm vector FAISS.
    """
    
    def __init__(self, index_path: str = None, metadata_path: str = None):
        """
        Khởi tạo SimilarCaseFinder. Cố gắng tải index FAISS và metadata.
        
        Args:
            index_path (str, optional): Đường dẫn tới file FAISS index. Mặc định 'Backend/data/chest_xray.faiss'.
            metadata_path (str, optional): Đường dẫn tới file metadata JSON. Mặc định 'Backend/data/chest_xray_meta.json'.
        """
        if index_path is None:
            index_path = 'Backend/data/chest_xray.faiss'
        if metadata_path is None:
            metadata_path = 'Backend/data/chest_xray_meta.json'
            
        self.available = False
        self.index = None
        self.metadata = None
        
        if not FAISS_AVAILABLE:
            print("Cảnh báo: Không thể import faiss. Module sẽ bị vô hiệu hóa.")
            return
            
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                self.available = True
                print(f"Da tai thanh cong co so du lieu CBIR tu {index_path}. Tong so ca: {len(self.metadata)}")
            except Exception as e:
                print(f"Lỗi khi tải dữ liệu CBIR: {e}")
                self.available = False
        else:
            print(f"Khong tim thay file index {index_path}. Bat dau voi co so du lieu CBIR trong (Dynamic Growth).")
            self.index = faiss.IndexFlatIP(1536)
            self.metadata = []
            self.available = True

    def add_case(self, embedding_vector: np.ndarray, metadata: dict) -> bool:
        """
        Thêm một ca bệnh thực tế mới vào cơ sở dữ liệu FAISS và cập nhật JSON.
        
        Args:
            embedding_vector (np.ndarray): Vector đặc trưng hình ảnh
            metadata (dict): Thông tin ca bệnh (VD: {'filename': 'BN01', 'diseases': ['Pneumonia']})
        """
        if not self.available or self.index is None:
            return False
            
        try:
            if len(embedding_vector.shape) == 1:
                query_vector = np.expand_dims(embedding_vector, axis=0)
            else:
                query_vector = embedding_vector.copy()
                
            query_vector = query_vector.astype(np.float32)
            faiss.normalize_L2(query_vector)
            
            # Thêm vào index FAISS đang chạy trên RAM
            self.index.add(query_vector)
            self.metadata.append(metadata)
            
            # Ghi đè file xuống ổ cứng để lưu trữ vĩnh viễn
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                
            print(f"Đã lưu ca bệnh mới vào CBIR. Tổng số ca hiện tại: {len(self.metadata)}")
            return True
        except Exception as e:
            print(f"Lỗi khi thêm ca bệnh vào CBIR: {e}")
            return False
            
    def search(self, embedding_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm k ca X-quang tương tự nhất với vector nhúng đầu vào.
        
        Args:
            embedding_vector (np.ndarray): Vector đặc trưng hình ảnh (shape [1024] hoặc [1, 1024]).
            k (int): Số lượng kết quả cần trả về.
            
        Returns:
            List[Dict]: Danh sách các kết quả tương tự gồm filename, độ tương tự (%) và danh sách bệnh.
        """
        if not self.available or self.index is None or self.metadata is None:
            return []
            
        # Chuẩn bị vector
        if len(embedding_vector.shape) == 1:
            query_vector = np.expand_dims(embedding_vector, axis=0)
        else:
            query_vector = embedding_vector.copy()
            
        # Đảm bảo là float32
        query_vector = query_vector.astype(np.float32)
        
        # Chuẩn hóa L2 cho tìm kiếm cosine similarity (IndexFlatIP + L2 norm = Cosine similarity)
        faiss.normalize_L2(query_vector)
        
        # Tìm kiếm
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: # Không tìm thấy kết quả hợp lệ
                continue
                
            idx = int(idx)
            # Distance của Inner Product đã chuẩn hóa nằm trong khoảng [-1, 1] cho cosine similarity
            # Chuyển giá trị này thành phần trăm [0, 100]%
            sim_score = float(distances[0][i])
            sim_percent = max(0.0, min(100.0, sim_score * 100.0))
            
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({
                    'filename': meta.get('filename', ''),
                    'similarity': round(sim_percent, 2),
                    'diseases': meta.get('diseases', [])
                })
                
        return results

    @staticmethod
    def build_index(model, dataset_csv_path: str, image_root: str, output_dir: str, batch_size: int = 32, device: str = 'cuda', limit: int = None):
        """
        Tạo FAISS index từ tập dữ liệu. Yêu cầu có PyTorch và các thư viện liên quan.
        
        Args:
            model: Mô hình PyTorch đã load. Cần trả về embedding hoặc hỗ trợ tham số return_embedding=True.
            dataset_csv_path (str): Đường dẫn file CSV chứa danh sách ảnh.
            image_root (str): Thư mục gốc chứa ảnh.
            output_dir (str): Thư mục lưu kết quả.
            batch_size (int): Kích thước batch khi trích xuất đặc trưng.
            device (str): Thiết bị tính toán ('cpu' hoặc 'cuda').
            limit (int, optional): Giới hạn số lượng ảnh để xử lý (phục vụ test).
        """
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset
            from torchvision import transforms
            from PIL import Image
            import pandas as pd
            import time
            from tqdm import tqdm
        except ImportError as e:
            print(f"Lỗi: Thiếu thư viện cần thiết để tạo index: {e}")
            return

        if not FAISS_AVAILABLE:
            print("Lỗi: Cần cài đặt faiss-cpu hoặc faiss-gpu để tạo index.")
            return

        print(f"Bắt đầu xây dựng cơ sở dữ liệu CBIR tại {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load CSV
        try:
            df = pd.read_csv(dataset_csv_path)
        except Exception as e:
            print(f"Lỗi khi đọc file CSV {dataset_csv_path}: {e}")
            return
            
        if limit is not None and limit > 0:
            df = df.head(limit)
            print(f"Giới hạn số lượng ảnh xử lý: {limit}")
            
        # Các nhãn bệnh có thể
        class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
                       'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
                       'Fibrosis', 'Pleural_Thickening', 'Hernia']

        # Validation transform chuẩn của mô hình
        transform = transforms.Compose([
            transforms.Resize(438),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = EmbeddingDataset(df, image_root, transform=transform)
        # Chạy 2 luồng (num_workers=2) theo yêu cầu của user để thử tận dụng CPU
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, prefetch_factor=2)

        model = model.to(device)
        model.eval()

        all_embeddings = []
        metadata = []
        
        start_time = time.time()
        print(f"Bắt đầu trích xuất đặc trưng cho {len(dataset)} ảnh...")
        
        skipped_files = []
        with torch.no_grad():
            for i, (images, img_names, valid_flags) in enumerate(tqdm(dataloader, desc="Trích xuất đặc trưng", unit="batch")):
                images = images.to(device)

                # Model.py giờ hỗ trợ return_embedding=True thật sự (trước đây không
                # có tham số này -> luôn rơi vào except -> âm thầm build cả index
                # bằng vector 0 giả, khiến tìm kiếm "ca tương tự" vô nghĩa với MỌI
                # ảnh). Không còn fallback im lặng: nếu forward lỗi, để nó ném lỗi
                # thật ra ngoài — thà dừng build index còn hơn build ra 1 index rác.
                _, _, embeddings = model(images, return_embedding=True)

                # Chuyển embedding sang numpy float32
                embeddings_np = embeddings.cpu().numpy().astype(np.float32)

                # Thu thập metadata trực tiếp từ dataframe cho chính xác, đồng thời
                # LOẠI BỎ các ảnh không đọc được (valid_flags[j]==False) khỏi cả
                # embeddings lẫn metadata thay vì nhét vector ảnh đen giả vào index.
                batch_start_idx = i * batch_size
                for j in range(len(img_names)):
                    idx_in_df = batch_start_idx + j
                    if idx_in_df >= len(df):
                        continue
                    is_valid = bool(valid_flags[j]) if hasattr(valid_flags, '__getitem__') else bool(valid_flags)
                    if not is_valid:
                        skipped_files.append(img_names[j])
                        continue

                    row = df.iloc[idx_in_df]
                    labels_str = row.get('Finding Labels', '')
                    dis = []
                    if pd.notna(labels_str) and labels_str != 'No Finding':
                        dis = labels_str.split('|')

                    all_embeddings.append(embeddings_np[j:j+1])
                    metadata.append({
                        'filename': img_names[j],
                        'diseases': dis
                    })

        if skipped_files:
            print(f"[CBIR] Đã bỏ qua {len(skipped_files)} ảnh lỗi/không đọc được, "
                  f"không đưa vào index (ví dụ: {skipped_files[:5]})")

        if not all_embeddings:
            print("Không thể trích xuất đặc trưng. Dừng xử lý.")
            return

        # Nối tất cả embeddings thành mảng numpy 2D
        embeddings_matrix = np.vstack(all_embeddings)
        print(f"Đã trích xuất xong matrix đặc trưng kích thước: {embeddings_matrix.shape}")
        
        # Chuẩn hóa embeddings cho cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Khởi tạo FAISS index bằng Inner Product
        embedding_dim = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        
        # Thêm dữ liệu vào index
        index.add(embeddings_matrix)
        
        # Lưu index và metadata
        index_file = os.path.join(output_dir, 'chest_xray.faiss')
        meta_file = os.path.join(output_dir, 'chest_xray_meta.json')
        
        faiss.write_index(index, index_file)
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        elapsed = time.time() - start_time
        print(f"Đã hoàn thành sau {elapsed:.2f} giây!")
        print(f"Đã lưu index ({index.ntotal} ảnh) vào {index_file}")
        print(f"Đã lưu metadata vào {meta_file}")