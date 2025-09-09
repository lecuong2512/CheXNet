import torch
import torch.nn as nn
try:
    from torchvision.models import densenet121, DenseNet121_Weights
    _HAS_WEIGHTS = True
except Exception:
    from torchvision.models import densenet121
    _HAS_WEIGHTS = False

class DenseNet121(nn.Module):
    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        
        if _HAS_WEIGHTS:
            if isTrained:
                self.densenet121 = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            else:
                self.densenet121 = densenet121(weights=None)
        else:
            # Fallback for older torchvision
            self.densenet121 = densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(kernelCount, classCount),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x)
