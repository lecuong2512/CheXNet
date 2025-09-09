import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class DatasetGenerator(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, transform=None):
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform

        # Đọc file danh sách ảnh,nhãn
        with open(pathDatasetFile, "r") as f:
            for line in f:
                items = line.strip().split()
                if not items:
                    continue
                imagePath = os.path.join(pathImageDirectory, items[0])
                imageLabel = torch.tensor([int(x) for x in items[1:]], dtype=torch.float32)

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)

    def __getitem__(self, index):
        image = Image.open(self.listImagePaths[index]).convert('RGB')
        label = self.listImageLabels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.listImagePaths)
