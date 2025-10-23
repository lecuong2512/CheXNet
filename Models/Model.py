import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False
    print("Warning: timm library not found. Please install it: pip install timm")

class ConvNeXtV2Large(nn.Module):
    def __init__(self, classCount, isTrained):
        super(ConvNeXtV2Large, self).__init__()
        
        if not _HAS_TIMM:
            raise ImportError("timm library is required. Install it with: pip install timm")
        
        # Load ConvNeXtV2-Large model
        if isTrained:
            self.model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k', 
                                          pretrained=True)
        else:
            self.model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k', 
                                          pretrained=False)
        
        # Get the number of features from the classifier
        num_features = self.model.head.fc.in_features
        
        # Replace the classifier with custom head
        self.model.head.fc = nn.Sequential(
            nn.Linear(num_features, classCount),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Giữ lại DenseNet121 để tương thích ngược
class DenseNet121(nn.Module):
    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        
        try:
            from torchvision.models import densenet121, DenseNet121_Weights
            if isTrained:
                self.densenet121 = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            else:
                self.densenet121 = densenet121(weights=None)
        except Exception:
            from torchvision.models import densenet121
            self.densenet121 = densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(kernelCount, classCount),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x)