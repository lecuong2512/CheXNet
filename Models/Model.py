import torch
import torch.nn as nn
import os
try:
    from torchvision.models import densenet121, DenseNet121_Weights
    _HAS_WEIGHTS = True
except Exception:
    from torchvision.models import densenet121
    _HAS_WEIGHTS = False

class DenseNet121(nn.Module):
    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        
        # Đường dẫn đến file weights local cho Kaggle
        local_weight_path = "/kaggle/input/datasets/densenet121-a639ec97.pth"
        
        if isTrained:
            # Kiểm tra xem có file weights local không (cho Kaggle)
            if os.path.exists(local_weight_path):
                print("Loading DenseNet121 from local weights...")
                state_dict = torch.load(local_weight_path, map_location="cpu")
                self.densenet121 = densenet121(weights=None)  # không tải online
                self.densenet121.load_state_dict(state_dict)
            elif _HAS_WEIGHTS:
                # Fallback: tải từ torchvision nếu có weights
                self.densenet121 = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            else:
                # Fallback cho torchvision cũ
                self.densenet121 = densenet121(pretrained=True)
        else:
            self.densenet121 = densenet121(weights=None)

        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(kernelCount, classCount),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x)
