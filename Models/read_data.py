# read_data.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class DatasetGenerator(Dataset):
    """
    Optimized dataset for NIH ChestX-ray14 multi-label classification
    """
    def __init__(self, pathImageDirectory, pathDatasetFile, transform=None):
        self.pathImageDirectory = pathImageDirectory
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform

        # Read dataset file
        print(f"Loading dataset from: {pathDatasetFile}")
        with open(pathDatasetFile, "r") as f:
            for line in f:
                items = line.strip().split()
                if not items:
                    continue
                
                # Image path (relative)
                imagePath = items[0]
                
                # Labels: 14 diseases + 1 "No Finding"
                imageLabel = torch.tensor([int(x) for x in items[1:]], dtype=torch.float32)
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)
        
        print(f"Loaded {len(self.listImagePaths)} images")

    def __getitem__(self, index):
        # Load image
        imagePath = os.path.join(self.pathImageDirectory, self.listImagePaths[index])
        
        try:
            image = Image.open(imagePath).convert('RGB')
        except Exception as e:
            print(f"Error loading image {imagePath}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.listImageLabels[index]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.listImagePaths)
