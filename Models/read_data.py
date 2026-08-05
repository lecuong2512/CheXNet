# read_data.py
import os
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image, ImageDraw
import numpy as np
from typing import Callable, Optional, List, Tuple
import pickle
from pathlib import Path
import io
from concurrent.futures import ThreadPoolExecutor
import warnings
import pandas as pd
import torchvision.transforms as transforms
import ast
import random

def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)

# Tên nguồn dữ liệu có bounding box annotation (dùng cho Dice/Tversky loss).
# Định nghĩa 1 lần ở đây và import dùng chung ở các file khác, tránh hardcode
# chuỗi rải rác nhiều nơi - nếu tên nguồn trong CSV đổi trong tương lai (như đã
# từng đổi từ 'VinBigData' -> 'VinDr-CXR'), chỉ cần sửa đúng 1 chỗ duy nhất.
VINDR_SOURCE_NAME = 'VinDr-CXR'

def select_center_crop(crops):
    return crops[2]

class DatasetGenerator(Dataset):
    DISEASE_COLUMNS = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    NUM_CLASSES = len(DISEASE_COLUMNS)

    def __init__(self, 
                 pathImageDirectory: str, 
                 pathDatasetFile: str, 
                 transform: Optional[Callable] = None,
                 cache_images: bool = False,
                 preload_images: bool = False,
                 num_workers_preload: int = 4):
        
        self.pathImageDirectory = pathImageDirectory
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        self.bytes_cache = None  # chỉ được gán dict thật nếu preload_images=True

        print(f"Loading dataset from: {pathDatasetFile}")
        self.listImagePaths, self.listImageLabels, self.listImageMasks, self.sources = self._load_dataset_file(pathDatasetFile)
        print(f"Loaded {len(self.listImagePaths)} images")
        
        if preload_images:
            self._preload_all_images(num_workers_preload)
        
        self._compute_statistics()

        # Precompute source flag: 1 = VinDr-CXR, 0 = other (NIH etc.)
        # Trả về int flag thay vì string để tránh string comparison mỗi batch
        self.source_flags = torch.tensor(
            [1 if s == VINDR_SOURCE_NAME else 0 for s in self.sources],
            dtype=torch.int8
        )

    def _load_dataset_file(self, pathDatasetFile: str):
        file_ext = os.path.splitext(pathDatasetFile)[1].lower()
        if file_ext == '.csv':
            return self._load_csv_file(pathDatasetFile)
        else:
            raise ValueError("Chỉ hỗ trợ file CSV cho kiến trúc có Masking")
    
    def _load_csv_file(self, pathDatasetFile: str):
        """
        Đọc CSV và GỘP NHÓM theo 'Image Index'.

        Lý do bắt buộc phải gộp nhóm: dữ liệu VinDr-CXR thật có thể chứa NHIỀU
        DÒNG cho CÙNG MỘT ẢNH - mỗi dòng là một bbox + nhãn do MỘT radiologist
        gán (cột 'rad_id'), và một ảnh thường được nhiều radiologist đọc độc
        lập. Nếu lặp qua từng dòng như một sample riêng (cách cũ), cùng một ảnh
        sẽ bị nạp lặp lại nhiều lần, mỗi lần chỉ mang 1 bbox/nhãn đơn lẻ thay vì
        toàn bộ thông tin đã được nhiều bác sĩ xác nhận trên ảnh đó.

        Chiến lược gộp (theo yêu cầu):
        - Nhãn bệnh: OR giữa các radiologist - một ảnh có thể có NHIỀU bệnh
          cùng lúc, nên nếu bất kỳ bác sĩ nào đánh dấu bệnh X trên ảnh, ảnh đó
          được gán nhãn X=1 (không phải mâu thuẫn cần chọn 1 trong nhiều).
        - Bounding box: gộp theo TỪNG BỆNH riêng biệt - bbox của bệnh X chỉ
          được gán vào mask của đúng bệnh X (không trộn lẫn bbox của bệnh khác
          vào sai class), phục vụ kiến trúc 15 attention map (1 map/bệnh).
        """
        print("Loading CSV format...")
        df = pd.read_csv(pathDatasetFile)
        df.columns = df.columns.str.strip()
        
        disease_columns = self.DISEASE_COLUMNS
        
        missing_cols = [col for col in disease_columns if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                df[col] = 0

        has_bbox_cols = all(c in df.columns for c in ['x_min', 'y_min', 'x_max', 'y_max'])
        if not has_bbox_cols:
            print("⚠️  CSV không có cột x_min/y_min/x_max/y_max -> sẽ không có ground-truth "
                  "bounding box nào, Dice loss sẽ không học được gì có ý nghĩa.")

        image_paths, image_labels, image_masks, sources = [], [], [], []

        # Gộp nhóm theo Image Index để xử lý đúng trường hợp nhiều radiologist
        for image_index, group in df.groupby('Image Index', sort=False):
            image_path = str(image_index).strip()

            # Source: lấy giá trị xuất hiện trong nhóm (tất cả các dòng cùng 1 ảnh
            # phải cùng Source, lấy dòng đầu tiên cho an toàn)
            source = str(group['Source'].iloc[0]).strip()

            # Nhãn bệnh: OR giữa tất cả radiologist trong nhóm - chỉ cần 1 dòng
            # có nhãn dương cho bệnh X thì ảnh được gán bệnh X=1
            labels = [
                int((group[disease].fillna(0).astype(float) > 0).any())
                for disease in disease_columns
            ]

            # Bbox: gom theo TỪNG BỆNH riêng (dict {disease_idx: [(x1,y1,x2,y2),...]})
            # chỉ áp dụng cho nguồn có annotation bbox (VinDr-CXR)
            mask_coords_per_class = {}
            if has_bbox_cols and source == VINDR_SOURCE_NAME:
                for _, row in group.iterrows():
                    if not pd.notna(row.get('x_min')):
                        continue
                    try:
                        x1, y1, x2, y2 = float(row['x_min']), float(row['y_min']), float(row['x_max']), float(row['y_max'])
                    except (ValueError, TypeError):
                        continue
                    # Xác định bbox này thuộc (các) bệnh nào trên CHÍNH DÒNG đó
                    # (mỗi dòng/radiologist có thể đánh dấu nhiều bệnh cho cùng 1 bbox)
                    for disease_idx, disease in enumerate(disease_columns):
                        val = row.get(disease)
                        if pd.notna(val) and float(val) > 0:
                            mask_coords_per_class.setdefault(disease_idx, []).append((x1, y1, x2, y2))

            image_paths.append(image_path)
            image_labels.append(torch.tensor(labels, dtype=torch.float32))
            image_masks.append(mask_coords_per_class)
            sources.append(source)
            
        return image_paths, image_labels, image_masks, sources

    def _preload_all_images(self, num_workers: int = 4):
        """
        Đọc trước toàn bộ ảnh vào RAM dưới dạng BYTES THÔ (chưa decode PIL).

        Lý do lưu bytes thô thay vì PIL Image đã decode: ảnh X-quang sau khi
        decode (RGB, ~1024x1024+) nặng gấp 10-20 lần so với file nén trên đĩa
        (JPEG/PNG). Với >100k ảnh, decode sẵn hết vào RAM dễ vượt quá RAM máy
        dù dữ liệu gốc chỉ ~50GB. Lưu bytes thô giữ RAM dùng đúng bằng dung
        lượng dữ liệu trên đĩa, decode PIL chỉ thực hiện 1 lần mỗi __getitem__
        (không phải đọc đĩa) -> vẫn nhanh hơn nhiều so với I/O đĩa mỗi epoch.
        """
        total_files = len(self.listImagePaths)
        print(f"\n💾 Preload {total_files} ảnh vào RAM (dùng {num_workers} threads)...")

        def _read_bytes(idx_path):
            idx, rel_path = idx_path
            full_path = os.path.join(self.pathImageDirectory, rel_path)
            try:
                with open(full_path, 'rb') as f:
                    return idx, f.read()
            except Exception:
                return idx, None

        self.bytes_cache = {}
        total_bytes = 0
        items = list(enumerate(self.listImagePaths))

        try:
            from tqdm import tqdm
            iterator = tqdm(total=total_files, desc="Preloading", ncols=100)
        except ImportError:
            iterator = None

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for idx, data in executor.map(_read_bytes, items):
                if data is not None:
                    self.bytes_cache[idx] = data
                    total_bytes += len(data)
                if iterator is not None:
                    iterator.update(1)

        if iterator is not None:
            iterator.close()

        n_failed = total_files - len(self.bytes_cache)
        print(f"✅ Đã preload {len(self.bytes_cache)}/{total_files} ảnh "
              f"(~{total_bytes / 1024**3:.2f} GB vào RAM)")
        if n_failed > 0:
            print(f"⚠️  {n_failed} ảnh không đọc được, sẽ dùng ảnh đen khi __getitem__ gặp lỗi")
        # Lưu ý: trên Linux, DataLoader worker dùng multiprocessing 'fork' (mặc định),
        # nên bytes_cache này được các worker process con CHIA SẺ qua copy-on-write,
        # không bị nhân bản theo số lượng num_workers. Vì vậy preload phải hoàn tất
        # TRƯỚC khi gọi FastDataLoader.create_dataloader(...).

    def _compute_statistics(self):
        labels_array = torch.stack(self.listImageLabels)
        self.class_counts = labels_array.sum(dim=0)
        self.class_frequencies = self.class_counts / len(self.listImageLabels)
        labels_per_image = labels_array.sum(dim=1)
        self.avg_labels_per_image = labels_per_image.mean().item()
        self.max_labels_per_image = labels_per_image.max().item()
        self.min_labels_per_image = labels_per_image.min().item()

    def __getitem__(self, index: int):
        if self.image_cache is not None and index in self.image_cache:
            image = self.image_cache[index]
        elif self.bytes_cache is not None and index in self.bytes_cache:
            # Ảnh đã được preload vào RAM dưới dạng bytes thô -> decode trực tiếp,
            # không cần đọc đĩa, nhanh hơn nhiều khi lặp lại qua nhiều epoch.
            try:
                image = Image.open(io.BytesIO(self.bytes_cache[index])).convert('RGB')
            except Exception:
                image = Image.new('RGB', (224, 224), color='black')
            if self.cache_images and self.image_cache is not None:
                self.image_cache[index] = image
        else:
            imagePath = os.path.join(self.pathImageDirectory, self.listImagePaths[index])
            try:
                with open(imagePath, 'rb') as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                if self.cache_images and self.image_cache is not None:
                    self.image_cache[index] = image
            except Exception:
                image = Image.new('RGB', (224, 224), color='black')
        
        label = self.listImageLabels[index]
        source = self.sources[index]
        mask_coords_per_class = self.listImageMasks[index]  # dict {disease_idx: [(x1,y1,x2,y2),...]}

        # 1. Tạo Binary Mask NHIỀU KÊNH (1 kênh / bệnh) từ coordinates.
        # Mỗi kênh chỉ chứa bbox của ĐÚNG bệnh tương ứng - không trộn lẫn vùng
        # tổn thương giữa các bệnh khác nhau, phục vụ kiến trúc 15 attention map
        # (1 map riêng cho mỗi bệnh trong DISEASE_COLUMNS).
        W_orig, H_orig = image.size
        mask_channels = []
        for disease_idx in range(self.NUM_CLASSES):
            ch_mask = Image.new('L', (W_orig, H_orig), 0)
            boxes = mask_coords_per_class.get(disease_idx) if mask_coords_per_class else None
            if boxes:
                draw = ImageDraw.Draw(ch_mask)
                for box in boxes:
                    draw.rectangle([box[0], box[1], box[2], box[3]], fill=255)
            mask_channels.append(transforms.ToTensor()(ch_mask))
        mask_tensor_multi = torch.cat(mask_channels, dim=0)  # [NUM_CLASSES, H_orig, W_orig]

        # 2. Áp dụng biến đổi chung cho cả Ảnh và Mask để đồng bộ
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.transform:
            image_tensor = transforms.ToTensor()(image)
            
            # Ghép ảnh (3 kênh) + mask đa kênh (NUM_CLASSES kênh) để qua transform
            # hình học cùng lúc (RandomRotation/Affine/Crop phải áp dụng giống hệt
            # nhau cho ảnh và toàn bộ các kênh mask để giữ đồng bộ không gian)
            combined = torch.cat([image_tensor, mask_tensor_multi], dim=0)
            combined = self.transform(combined)
            
            image = combined[:3, :, :]
            mask = combined[3:, :, :]  # [NUM_CLASSES, H, W]
            # Re-binarize mask sau geometric transforms (bilinear interpolation
            # làm blur giá trị 0/1 thành giá trị trung gian — threshold 0.5
            # khôi phục lại binary mask chính xác)
            mask = (mask > 0.5).float()
            image = normalize(image)
        else:
            image = normalize(transforms.ToTensor()(image))
            mask = mask_tensor_multi

        source_flag = self.source_flags[index]
        return image, label, mask, source_flag

    def __len__(self) -> int:
        return len(self.listImagePaths)
    
    def get_class_weights(self, smooth: float = 0.1) -> torch.Tensor:
        class_freqs = self.class_frequencies
        class_weights = 1.0 / (class_freqs + smooth)
        return class_weights / class_weights.sum() * len(class_freqs)
    
    def print_statistics(self):
        print(f"Total samples: {len(self)}")

class FastDataLoader:
    @staticmethod
    def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4,
                          pin_memory: bool = True, prefetch_factor: int = 2,
                          persistent_workers: bool = True, drop_last: bool = False,
                          sampler=None, batch_sampler=None):
        """
        batch_sampler (HybridBatchSampler): khi dùng batch_sampler, các tham số
        batch_size / shuffle / sampler / drop_last bị bỏ qua hoàn toàn theo PyTorch spec.
        """
        if batch_sampler is not None:
            return torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers and num_workers > 0,
                worker_init_fn=worker_init_fn,
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
        )

class HybridBatchSampler(Sampler):
    """Sampler chia tỉ lệ 1:3 (VinDr : Khác) cho Giai đoạn 2"""
    def __init__(self, sources, batch_size):
        self.vin_idx = [i for i, s in enumerate(sources) if s == VINDR_SOURCE_NAME]
        self.nih_idx = [i for i, s in enumerate(sources) if s != VINDR_SOURCE_NAME]
        
        self.batch_size = batch_size
        self.vin_per_batch = int(batch_size * 0.25)
        self.nih_per_batch = batch_size - self.vin_per_batch
        
        # Fallback nếu không có data VinDr
        if len(self.vin_idx) == 0:
            self.vin_per_batch = 0
            self.nih_per_batch = batch_size
            self.num_batches = len(self.nih_idx) // batch_size
        else:
            # Dùng max(vin, nih) để không mất dữ liệu, cả hai đều wrap-around
            self.num_batches = max(
                len(self.nih_idx) // self.nih_per_batch,
                len(self.vin_idx) // self.vin_per_batch if self.vin_per_batch > 0 else 0
            )

    def __iter__(self):
        random.shuffle(self.vin_idx)
        random.shuffle(self.nih_idx)
        
        for i in range(self.num_batches):
            if self.vin_per_batch > 0:
                # Wrap-around an toàn: khi v_start + vin_per_batch vượt qua cuối
                # list, nối phần đầu list để đảm bảo luôn đủ vin_per_batch phần tử
                v_start = (i * self.vin_per_batch) % len(self.vin_idx)
                v_end = v_start + self.vin_per_batch
                if v_end <= len(self.vin_idx):
                    batch_vin = self.vin_idx[v_start:v_end]
                else:
                    batch_vin = self.vin_idx[v_start:] + self.vin_idx[:v_end - len(self.vin_idx)]
            else:
                batch_vin = []
            
            # Wrap-around cho NIH: tránh mất phần dư mỗi epoch
            n_start = (i * self.nih_per_batch) % len(self.nih_idx)
            n_end = n_start + self.nih_per_batch
            if n_end <= len(self.nih_idx):
                batch_nih = self.nih_idx[n_start:n_end]
            else:
                batch_nih = self.nih_idx[n_start:] + self.nih_idx[:n_end - len(self.nih_idx)]
            
            batch = batch_vin + batch_nih
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
