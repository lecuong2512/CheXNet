import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False
    print("Warning: timm library not found. Install: pip install timm")


class MultiModelArchitecture(nn.Module):
    """
    Unified model class supporting multiple architectures:
    - DenseNet family (121, 169, 201)
    - ConvNeXtV2 family (base, large, huge)
    - EfficientNetV2 family (s, m, l)
    - Swin Transformer family (tiny, small, base)
    """
    
    SUPPORTED_MODELS = {
        # DenseNet family
        'densenet121': 'densenet121',
        'densenet169': 'densenet169', 
        'densenet201': 'densenet201',
        
        # ConvNeXtV2 family
        'convnextv2_base': 'convnextv2_base.fcmae_ft_in22k_in1k',
        'convnextv2_large': 'convnextv2_large.fcmae_ft_in22k_in1k',
        'convnextv2_huge': 'convnextv2_huge.fcmae_ft_in22k_in1k',
        
        # EfficientNetV2 family
        'efficientnetv2_s': 'tf_efficientnetv2_s.in21k_ft_in1k',
        'efficientnetv2_m': 'tf_efficientnetv2_m.in21k_ft_in1k',
        'efficientnetv2_l': 'tf_efficientnetv2_l.in21k_ft_in1k',
        
        # Swin Transformer family
        'swin_tiny': 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
        'swin_small': 'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
        'swin_base': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    }
    
    def __init__(self, model_name, classCount, isTrained=True):
        super(MultiModelArchitecture, self).__init__()
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        
        # Handle DenseNet separately (torchvision)
        if model_name.startswith('densenet'):
            self.model = self._create_densenet(model_name, classCount, isTrained)
        else:
            # Use timm for other models
            if not _HAS_TIMM:
                raise ImportError("timm library required. Install: pip install timm")
            self.model = self._create_timm_model(model_name, classCount, isTrained)
    
    def _create_densenet(self, model_name, classCount, isTrained):
        """Create DenseNet model using torchvision"""
        from torchvision import models
        
        model_map = {
            'densenet121': models.densenet121,
            'densenet169': models.densenet169,
            'densenet201': models.densenet201,
        }
        
        try:
            # Try new torchvision API (0.13+)
            weights = 'IMAGENET1K_V1' if isTrained else None
            model = model_map[model_name](weights=weights)
        except:
            # Fallback to old API
            model = model_map[model_name](pretrained=isTrained)
        
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, classCount),
            nn.Sigmoid()
        )
        return model
    
    def _create_timm_model(self, model_name, classCount, isTrained):
        """Create model using timm library"""
        timm_model_name = self.SUPPORTED_MODELS[model_name]
        
        # Create base model
        model = timm.create_model(timm_model_name, pretrained=isTrained)
        
        # Get classifier layer name and feature count
        if hasattr(model, 'head'):
            if hasattr(model.head, 'fc'):
                num_features = model.head.fc.in_features
                model.head.fc = nn.Sequential(
                    nn.Linear(num_features, classCount),
                    nn.Sigmoid()
                )
            else:
                num_features = model.head.in_features
                model.head = nn.Sequential(
                    nn.Linear(num_features, classCount),
                    nn.Sigmoid()
                )
        elif hasattr(model, 'classifier'):
            num_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_features, classCount),
                nn.Sigmoid()
            )
        else:
            raise AttributeError(f"Cannot find classifier layer for {model_name}")
        
        return model
    
    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def get_recommended_lr(model_name):
        """Get recommended learning rate for each architecture"""
        lr_map = {
            'densenet121': 1e-4,
            'densenet169': 1e-4,
            'densenet201': 1e-4,
            'convnextv2_base': 5e-5,
            'convnextv2_large': 5e-5,
            'convnextv2_huge': 3e-5,
            'efficientnetv2_s': 1e-4,
            'efficientnetv2_m': 8e-5,
            'efficientnetv2_l': 5e-5,
            'swin_tiny': 1e-4,
            'swin_small': 8e-5,
            'swin_base': 5e-5,
        }
        return lr_map.get(model_name, 1e-4)
    
    @staticmethod
    def get_recommended_batch_size(model_name, gpu_memory_gb):
        """Get recommended batch size based on model and GPU memory"""
        # Base batch sizes for 16GB GPU
        base_batch_sizes = {
            'densenet121': 64,
            'densenet169': 48,
            'densenet201': 32,
            'convnextv2_base': 32,
            'convnextv2_large': 16,
            'convnextv2_huge': 8,
            'efficientnetv2_s': 64,
            'efficientnetv2_m': 32,
            'efficientnetv2_l': 16,
            'swin_tiny': 64,
            'swin_small': 32,
            'swin_base': 16,
        }
        
        base_size = base_batch_sizes.get(model_name, 32)
        
        # Scale based on GPU memory
        if gpu_memory_gb >= 24:
            return int(base_size * 1.5)
        elif gpu_memory_gb >= 12:
            return base_size
        elif gpu_memory_gb >= 8:
            return max(4, base_size // 2)
        else:
            return max(2, base_size // 4)


# Keep backward compatibility
class DenseNet121(MultiModelArchitecture):
    def __init__(self, classCount, isTrained=True):
        super().__init__('densenet121', classCount, isTrained)


class ConvNeXtV2Large(MultiModelArchitecture):
    def __init__(self, classCount, isTrained=True):
        super().__init__('convnextv2_large', classCount, isTrained)
