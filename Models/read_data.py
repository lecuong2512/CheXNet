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

def select_center_crop(crops):
    return crops[2]

class DatasetGenerator(Dataset):
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

        print(f"Loading dataset from: {pathDatasetFile}")
        self.listImagePaths, self.listImageLabels, self.listImageMasks, self.sources = self._load_dataset_file(pathDatasetFile)
        print(f"Loaded {len(self.listImagePaths)} images")
        
        if preload_images:
            self._preload_all_images(num_workers_preload)
        
        self._compute_statistics()

    def _load_dataset_file(self, pathDatasetFile: str):
        file_ext = os.path.splitext(pathDatasetFile)[1].lower()
        if file_ext == '.csv':
            return self._load_csv_file(pathDatasetFile)
        else:
            raise ValueError("Chỉ hỗ trợ file CSV cho kiến trúc có Masking")
    
    def _load_csv_file(self, pathDatasetFile: str):
        print("Loading CSV format...")
        df = pd.read_csv(pathDatasetFile)
        df.columns = df.columns.str.strip()
        
        disease_columns = [
            'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        missing_cols = [col for col in disease_columns if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                df[col] = 0
        
        image_paths, image_labels, image_masks, sources = [], [], [], []
        
        for idx, row in df.iterrows():
            if 'Image Index' in df.columns:
                image_path = str(row['Image Index']).strip()
            else:
                raise ValueError("CSV must have 'Image Index' column")
            
            source = str(row.get('Source', 'Unknown')).strip()
            labels = [int(row[disease]) if pd.notna(row[disease]) else 0 for disease in disease_columns]
            
            mask_coords = []
            if source == 'VinBigData' and pd.notna(row.get('x_min')):
                try:
                    x_min_list = ast.literal_eval(str(row['x_min'])) if '[' in str(row['x_min']) else [float(row['x_min'])]
                    y_min_list = ast.literal_eval(str(row['y_min'])) if '[' in str(row['y_min']) else [float(row['y_min'])]
                    x_max_list = ast.literal_eval(str(row['x_max'])) if '[' in str(row['x_max']) else [float(row['x_max'])]
                    y_max_list = ast.literal_eval(str(row['y_max'])) if '[' in str(row['y_max']) else [float(row['y_max'])]
                    
                    for i in range(len(x_min_list)):
                        mask_coords.append((x_min_list[i], y_min_list[i], x_max_list[i], y_max_list[i]))
                except Exception:
                    pass
            
            image_paths.append(image_path)
            image_labels.append(torch.tensor(labels, dtype=torch.float32))
            image_masks.append(mask_coords)
            sources.append(source)
            
        return image_paths, image_labels, image_masks, sources

    def _preload_all_images(self, num_workers: int = 4):
        pass # Rút gọn hàm preload để tập trung logic chính

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
        coords = self.listImageMasks[index]

        # 1. Tạo Binary Mask từ coordinates
        W_orig, H_orig = image.size
        mask = Image.new('L', (W_orig, H_orig), 0)
        if coords:
            draw = ImageDraw.Draw(mask)
            for box in coords:
                draw.rectangle([box[0], box[1], box[2], box[3]], fill=255)

        # 2. Áp dụng biến đổi chung cho cả Ảnh và Mask để đồng bộ
        if self.transform:
            mask_tensor = transforms.ToTensor()(mask)
            image_tensor = transforms.ToTensor()(image)
            
            # Ghép thành 4 kênh để qua transform hình học (nếu có)
            combined = torch.cat([image_tensor, mask_tensor], dim=0)
            combined = self.transform(combined)
            
            image = combined[:3, :, :]
            mask = combined[3:, :, :]
            
            # Normalize chuẩn ImageNet cho 3 kênh RGB
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            image = normalize(image)

        return image, label, mask, source

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
        self.vin_idx = [i for i, s in enumerate(sources) if s == 'VinBigData']
        self.nih_idx = [i for i, s in enumerate(sources) if s != 'VinBigData']
        
        self.batch_size = batch_size
        self.vin_per_batch = int(batch_size * 0.25)
        self.nih_per_batch = batch_size - self.vin_per_batch
        
        # Fallback nếu không có data VinDr
        if len(self.vin_idx) == 0:
            self.vin_per_batch = 0
            self.nih_per_batch = batch_size
            self.num_batches = len(self.nih_idx) // batch_size
        else:
            self.num_batches = len(self.nih_idx) // self.nih_per_batch

    def __iter__(self):
        random.shuffle(self.vin_idx)
        random.shuffle(self.nih_idx)
        
        for i in range(self.num_batches):
            if self.vin_per_batch > 0:
                v_start = (i * self.vin_per_batch) % len(self.vin_idx)
                batch_vin = self.vin_idx[v_start : v_start + self.vin_per_batch]
            else:
                batch_vin = []
                
            batch_nih = self.nih_idx[i * self.nih_per_batch : (i + 1) * self.nih_per_batch]
            
            batch = batch_vin + batch_nih
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
