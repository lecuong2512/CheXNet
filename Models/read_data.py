# read_data.py - Optimized dataset loader with caching and preprocessing
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Callable, Optional, List, Tuple
import pickle
from pathlib import Path
import io
from concurrent.futures import ThreadPoolExecutor
import warnings

class DatasetGenerator(Dataset):
    """
    Optimized dataset for NIH ChestX-ray14 multi-label classification
    with image caching and efficient loading
    """
    def __init__(self, 
                 pathImageDirectory: str, 
                 pathDatasetFile: str, 
                 transform: Optional[Callable] = None,
                 cache_images: bool = False,
                 preload_images: bool = False,
                 num_workers_preload: int = 4):
        """
        Args:
            pathImageDirectory: Root directory containing images
            pathDatasetFile: Path to dataset list file
            transform: Image transformations
            cache_images: Cache decoded images in memory (needs sufficient RAM)
            preload_images: Preload all images at initialization (very fast training)
            num_workers_preload: Number of workers for parallel preloading
        """
        self.pathImageDirectory = pathImageDirectory
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None

        # Load dataset file
        print(f"📂 Loading dataset from: {pathDatasetFile}")
        self.listImagePaths, self.listImageLabels = self._load_dataset_file(pathDatasetFile)
        
        print(f"✓ Loaded {len(self.listImagePaths)} images")
        
        # Preload images if requested
        if preload_images:
            self._preload_all_images(num_workers_preload)
        
        # Compute class statistics
        self._compute_statistics()

    def _load_dataset_file(self, pathDatasetFile: str) -> Tuple[List[str], List[torch.Tensor]]:
        """Load and parse dataset file efficiently"""
        image_paths = []
        image_labels = []
        
        with open(pathDatasetFile, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            items = line.strip().split()
            if not items:
                continue
            
            # Image path (relative)
            imagePath = items[0]
            
            # Labels: 14 diseases + 1 "No Finding"
            imageLabel = torch.tensor([int(x) for x in items[1:]], dtype=torch.float32)
            
            image_paths.append(imagePath)
            image_labels.append(imageLabel)
        
        return image_paths, image_labels

    def _preload_all_images(self, num_workers: int = 4):
        """Preload all images into memory for fastest training"""
        print(f"\n⚡ Preloading all images with {num_workers} workers...")
        print("   This may take a few minutes but will greatly speed up training...")
        
        from tqdm import tqdm
        
        def load_image(idx):
            imagePath = os.path.join(self.pathImageDirectory, self.listImagePaths[idx])
            try:
                with open(imagePath, 'rb') as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                return idx, image
            except Exception as e:
                warnings.warn(f"Error loading {imagePath}: {e}")
                return idx, None
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(load_image, range(len(self))),
                total=len(self),
                desc="Preloading images"
            ))
        
        self.image_cache = {}
        for idx, image in results:
            if image is not None:
                self.image_cache[idx] = image
        
        print(f"✓ Preloaded {len(self.image_cache)}/{len(self)} images into memory")
        
        # Estimate memory usage
        if self.image_cache:
            sample_img = next(iter(self.image_cache.values()))
            img_size_mb = np.prod(sample_img.size) * 3 / (1024 * 1024)  # RGB
            total_mb = img_size_mb * len(self.image_cache)
            print(f"  Estimated memory usage: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")

    def _compute_statistics(self):
        """Compute dataset statistics for monitoring"""
        labels_array = torch.stack(self.listImageLabels)
        self.class_counts = labels_array.sum(dim=0)
        self.class_frequencies = self.class_counts / len(self.listImageLabels)
        
        # Multi-label statistics
        labels_per_image = labels_array.sum(dim=1)
        self.avg_labels_per_image = labels_per_image.mean().item()
        self.max_labels_per_image = labels_per_image.max().item()
        self.min_labels_per_image = labels_per_image.min().item()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item with optimized loading"""
        # Try to get from cache first
        if self.image_cache is not None and index in self.image_cache:
            image = self.image_cache[index]
        else:
            # Load from disk
            imagePath = os.path.join(self.pathImageDirectory, self.listImagePaths[index])
            
            try:
                # Fast image loading using BytesIO
                with open(imagePath, 'rb') as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                
                # Cache if enabled
                if self.cache_images and self.image_cache is not None:
                    self.image_cache[index] = image
                    
            except Exception as e:
                warnings.warn(f"Error loading {imagePath}: {e}")
                # Return a blank image if loading fails
                image = Image.new('RGB', (224, 224), color='black')
        
        label = self.listImageLabels[index]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.listImagePaths)
    
    def get_class_weights(self, smooth: float = 0.1) -> torch.Tensor:
        """
        Get class weights for handling imbalanced data
        
        Args:
            smooth: Smoothing factor to prevent extreme weights
        
        Returns:
            Tensor of class weights
        """
        class_freqs = self.class_frequencies
        class_weights = 1.0 / (class_freqs + smooth)
        class_weights = class_weights / class_weights.sum() * len(class_freqs)
        return class_weights
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("📊 Dataset Statistics")
        print("="*60)
        print(f"Total samples: {len(self)}")
        print(f"Average labels per image: {self.avg_labels_per_image:.2f}")
        print(f"Label range: [{self.min_labels_per_image:.0f}, {self.max_labels_per_image:.0f}]")
        print(f"\nClass distribution:")
        for i, (count, freq) in enumerate(zip(self.class_counts, self.class_frequencies)):
            print(f"  Class {i:2d}: {count:6.0f} samples ({freq*100:5.2f}%)")
        print("="*60 + "\n")


class FastDataLoader:
    """
    Wrapper around DataLoader with optimizations
    """
    @staticmethod
    def create_dataloader(dataset: Dataset,
                          batch_size: int,
                          shuffle: bool = True,
                          num_workers: int = 4,
                          pin_memory: bool = True,
                          prefetch_factor: int = 2,
                          persistent_workers: bool = True,
                          drop_last: bool = False) -> torch.utils.data.DataLoader:
        """
        Create optimized DataLoader
        
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch per worker
            persistent_workers: Keep workers alive between epochs
            drop_last: Drop last incomplete batch
        
        Returns:
            Optimized DataLoader
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=drop_last,
            worker_init_fn=lambda worker_id: np.random.seed(torch.initial_seed() % 2**32)
        )


# Test time augmentation (TTA) dataset wrapper
class TTADataset(Dataset):
    """
    Dataset wrapper for Test-Time Augmentation
    Applies multiple augmentations to each image for ensemble prediction
    """
    def __init__(self, 
                 base_dataset: Dataset,
                 tta_transforms: List[Callable],
                 num_tta: int = 5):
        """
        Args:
            base_dataset: Original dataset
            tta_transforms: List of augmentation transforms for TTA
            num_tta: Number of augmented versions per image
        """
        self.base_dataset = base_dataset
        self.tta_transforms = tta_transforms
        self.num_tta = num_tta

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item with TTA augmentations"""
        # Get index in base dataset
        base_idx = index // self.num_tta
        tta_idx = index % self.num_tta
        
        # Get original image and label
        image, label = self.base_dataset[base_idx]
        
        # Apply TTA transform
        if tta_idx < len(self.tta_transforms):
            image = self.tta_transforms[tta_idx](image)
        
        return image, label

    def __len__(self) -> int:
        return len(self.base_dataset) * self.num_tta


def create_tta_transforms(base_transform: Callable, image_size: int = 384) -> List[Callable]:
    """
    Create TTA transforms for inference
    
    Args:
        base_transform: Base transformation pipeline
        image_size: Target image size
    
    Returns:
        List of TTA transforms
    """
    import torchvision.transforms as transforms
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    tta_list = [
        # Original (center crop)
        transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize
        ]),
        # Slight rotation
        transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            normalize
        ]),
        # Brightness adjustment
        transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            normalize
        ]),
        # Multi-crop (corners + center)
        transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.FiveCrop(image_size),
            transforms.Lambda(lambda crops: crops[2]),  # Use center crop
            transforms.ToTensor(),
            normalize
        ]),
    ]
    
    return tta_list


# Example usage
if __name__ == '__main__':
    import torchvision.transforms as transforms
    
    # Test configuration
    pathDirData = 'CheXNet/Database'
    pathFileTrain = 'CheXNet/Dataset/train_list.txt'
    
    # Create transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize(int(384 * 1.14)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        normalize
    ])
    
    # Test dataset loading
    print("Testing dataset loading...")
    dataset = DatasetGenerator(
        pathDirData, 
        pathFileTrain, 
        transform,
        cache_images=False,
        preload_images=False
    )
    
    dataset.print_statistics()
    
    # Test class weights
    weights = dataset.get_class_weights()
    print(f"\nClass weights: {weights}")
    
    # Test data loader
    print("\nTesting DataLoader...")
    loader = FastDataLoader.create_dataloader(
        dataset,
        batch_size=16,
        num_workers=4
    )
    
    for batch_idx, (images, labels) in enumerate(loader):
        print(f"Batch {batch_idx}: images shape={images.shape}, labels shape={labels.shape}")
        if batch_idx >= 2:
            break
    
    print("\n✓ All tests passed!")
