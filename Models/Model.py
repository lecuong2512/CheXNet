# Model.py
import torch
import torch.nn as nn
import timm

class ConvNeXtV2Model(nn.Module):
    """
    ConvNeXtV2-Large backbone for multi-label chest X-ray classification
    """
    def __init__(self, num_classes=15, pretrained=True, dropout_rate=0.2):
        super(ConvNeXtV2Model, self).__init__()
        
        # Load ConvNeXtV2-Large from timm
        self.backbone = timm.create_model(
            'convnextv2_large.fcmae_ft_in22k_in1k_384',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        self.num_features = self.backbone.num_features
        
        # Custom classifier head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features for visualization"""
        return self.backbone(x)
