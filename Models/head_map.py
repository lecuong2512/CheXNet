# Head_map.py (Grad-CAM for ConvNeXtV2)
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

class GradCAM:
    """
    Grad-CAM implementation for ConvNeXtV2 visualization
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, class_idx):
        """
        Generate CAM for a specific class
        
        Args:
            input_image: Input tensor (1, C, H, W)
            class_idx: Target class index
        
        Returns:
            cam: Grad-CAM heatmap
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(self, original_image, cam, alpha=0.5):
        """
        Overlay CAM on original image
        
        Args:
            original_image: PIL Image or numpy array
            cam: Grad-CAM heatmap
            alpha: Overlay transparency
        
        Returns:
            Overlaid image
        """
        # Convert to numpy if PIL
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Resize CAM to match image
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlaid = heatmap * alpha + original_image * (1 - alpha)
        overlaid = np.uint8(overlaid)
        
        return overlaid, heatmap


def generate_heatmap_for_image(model_path, image_path, class_idx, 
                                num_classes=15, image_size=384,
                                save_path=None):
    """
    Generate and save Grad-CAM heatmap for a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        class_idx: Target class index
        num_classes: Number of output classes
        image_size: Input image size
        save_path: Path to save visualization
    """
    from Models.Model import ConvNeXtV2Model
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNeXtV2Model(num_classes=num_classes, pretrained=False)
    
    # Load weights
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['state_dict']
    
    # Handle DataParallel
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        state_dict = OrderedDict(
            (k[7:], v) if k.startswith('module.') else (k, v)
            for k, v in state_dict.items()
        )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Get target layer (last conv layer of ConvNeXtV2)
    target_layer = model.backbone.stages[-1]  # Last stage
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Generate CAM
    cam = gradcam.generate_cam(input_tensor, class_idx)
    
    # Visualize
    original_np = np.array(original_image.resize((image_size, image_size)))
    overlaid, heatmap = gradcam.visualize(original_np, cam, alpha=0.5)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlaid)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return cam, overlaid


def generate_multi_class_heatmap(model_path, image_path, class_names,
                                  num_classes=15, image_size=384,
                                  top_k=5, save_dir='CheXNet/Heatmaps'):
    """
    Generate heatmaps for top-k predicted classes
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        class_names: List of class names
        num_classes: Number of output classes
        image_size: Input image size
        top_k: Number of top classes to visualize
        save_dir: Directory to save results
    """
    import os
    from Models.Model import ConvNeXtV2Model
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNeXtV2Model(num_classes=num_classes, pretrained=False)
    
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['state_dict']
    
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        state_dict = OrderedDict(
            (k[7:], v) if k.startswith('module.') else (k, v)
            for k, v in state_dict.items()
        )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Target layer
    target_layer = model.backbone.stages[-1]
    gradcam = GradCAM(model, target_layer)
    
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0]
    
    # Get top-k classes
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # Generate heatmaps for top-k
    original_np = np.array(original_image.resize((image_size, image_size)))
    
    fig, axes = plt.subplots(2, top_k, figsize=(4*top_k, 8))
    
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        idx = idx.item()
        prob = prob.item()
        
        # Generate CAM
        cam = gradcam.generate_cam(input_tensor, idx)
        overlaid, heatmap = gradcam.visualize(original_np, cam, alpha=0.5)
        
        # Plot
        axes[0, i].imshow(heatmap)
        axes[0, i].set_title(f'{class_names[idx]}\n({prob:.3f})', fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(overlaid)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'multi_heatmap_{os.path.basename(image_path)}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# Example usage
if __name__ == '__main__':
    MODEL_PATH = 'CheXNet/Trainedmodel/chexnetmodel.pth'
    IMAGE_PATH = 'CheXNet/Database/images_001/images/00000001_000.png'
    
    CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia', 'No Finding'
    ]
    
    # Generate multi-class heatmap
    generate_multi_class_heatmap(
        MODEL_PATH, IMAGE_PATH, CLASS_NAMES,
        num_classes=15, image_size=384, top_k=5
    )
