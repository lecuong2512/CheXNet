import os
import sys
import torch
import torchvision.transforms as transforms
from xray_validator import validate_chest_xray_image
from collections import OrderedDict
from PIL import Image


class InvalidXRayImageError(Exception):
    """Ảnh upload không phải X-quang ngực — ngoài phạm vi chẩn đoán bệnh phổi."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _validate_with_clip_ai(image: Image.Image) -> float:
    """CLIP zero-shot: ảnh có phải X-quang ngực. Trả về xác suất nhóm chest."""
    is_valid, chest_prob, message = validate_chest_xray_image(image)
    if not is_valid:
        raise InvalidXRayImageError(message)
    return chest_prob


# Thêm thư mục root của dự án vào sys.path để import Models
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from Models.Model import ConvNeXtV2Model

# Danh sách 15 lớp bệnh lý
CLASS_NAMES = [
    'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

class CheXNetInference:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AI Model] Sử dụng thiết bị: {self.device}")
        
        if model_path is None:
            model_path = os.path.join(root_dir, "Trainedmodel", "chexnetmodel.pth")
            
        self.model_path = model_path
        self.model = None
        self.model_version = 'unknown'
        self.image_size = 224  # Default fallback
        
        # Load model first to auto-detect architecture and set correct image size
        self.load_model()
        
        # Then initialize the transform with the correct image size
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        trans_crop = self.image_size
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return transforms.Compose([
            transforms.Resize(int(trans_crop * 1.14)),  # 255 for 224, 437 for 384
            transforms.CenterCrop(trans_crop),          # 224 or 384
            transforms.ToTensor(),
            normalize
        ])

    def load_model(self):
        print(f"[AI Model] Đang tải mô hình từ: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số tại {self.model_path}")
            
        try:
            # Tải checkpoint
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)
            
            # Kiểm tra xem trọng số thuộc về DenseNet-121 hay ConvNeXtV2
            is_densenet = any(k.startswith('densenet121.') for k in state_dict.keys())
            
            if is_densenet:
                print("[AI Model] 🔍 Phát hiện kiến trúc: DenseNet-121 (Classical CheXNet)")
                self.model_version = 'densenet-121'
                self.image_size = 224
                
                import torchvision.models as models
                import torch.nn as nn
                
                class DenseNet121Wrapper(nn.Module):
                    def __init__(self, num_classes=15):
                        super(DenseNet121Wrapper, self).__init__()
                        self.densenet121 = models.densenet121(weights=None)
                        num_ftrs = self.densenet121.classifier.in_features
                        self.densenet121.classifier = nn.Sequential(
                            nn.Linear(num_ftrs, num_classes)
                        )
                    
                    def forward(self, x):
                        return self.densenet121(x)
                
                self.model = DenseNet121Wrapper(num_classes=len(CLASS_NAMES))
                self.model.load_state_dict(state_dict)
            else:
                print("[AI Model] 🔍 Phát hiện kiến trúc: ConvNeXtV2-Large")
                self.model_version = 'convnextv2-large'
                self.image_size = 384
                
                # Khởi tạo kiến trúc ConvNeXtV2Model
                self.model = ConvNeXtV2Model(num_classes=len(CLASS_NAMES), pretrained=False)
                
                # Xử lý prefix từ torch.compile hoặc DataParallel nếu có
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k
                    if k.startswith('_orig_mod.'):
                        name = k.replace('_orig_mod.', '')
                    elif k.startswith('module.'):
                        name = k[7:]
                    new_state_dict[name] = v
                    
                self.model.load_state_dict(new_state_dict)
                
            self.model.to(self.device)
            self.model.eval()
            print(f"[AI Model] ✅ Tải mô hình thành công! (Image Size: {self.image_size}x{self.image_size})")
        except Exception as e:
            print(f"[AI Model] ❌ Lỗi khi tải mô hình: {e}")
            raise e

    def predict(self, image_path):
        """
        Nhận vào đường dẫn ảnh, thực hiện tiền xử lý và dự đoán.
        Trả về danh sách kết quả chứa className và probability của từng bệnh.
        """
        if self.model is None:
            raise RuntimeError("Mô hình chưa được tải thành công")
            
        try:
            image = Image.open(image_path).convert('RGB')
            _validate_with_clip_ai(image)

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.sigmoid(logits)[0]
                
            # Tạo danh sách kết quả dạng JSON-friendly
            predictions = []
            for i, class_name in enumerate(CLASS_NAMES):
                predictions.append({
                    "className": class_name,
                    "probability": float(probabilities[i])
                })
                
            predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
            # Chi tiết log ở [X-Ray AI Validator]

            return predictions
        except InvalidXRayImageError:
            raise
        except Exception as e:
            print(f"[AI Model] ❌ Lỗi khi thực hiện inference: {e}")
            raise e
