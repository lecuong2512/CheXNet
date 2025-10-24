# Models/Model.py
import torch
import torch.nn as nn
import timm

class SwinTransformer(nn.Module):
    """
    Swin Transformer wrapper, sử dụng thư viện 'timm'.
    Lớp này sẽ tải mô hình dựa trên chuỗi 'model_variant'.
    Outputs raw logits (no sigmoid).
    """
    def __init__(self, classCount, isTrained=True, 
                 # THAY ĐỔI: Cập nhật model_variant mặc định
                 model_variant='swin_large_patch4_window7_224.ms_in22k_ft_in1k'):
        """
        Khởi tạo mô hình.
        
        Args:
            classCount (int): Số lượng class output.
            isTrained (bool): True để tải trọng số pretrained.
            model_variant (str): Tên mô hình trong thư viện 'timm'.
                Ví dụ: 'swin_large_patch4_window7_224.ms_in22k_ft_in1k'
        """
        super(SwinTransformer, self).__init__()
        
        self.model_variant = model_variant
        print(f"[Model] Loading: {self.model_variant} (pretrained={isTrained})")

        self.swin = timm.create_model(
            self.model_variant,
            pretrained=isTrained,
            num_classes=classCount  # Tự động thay thế lớp head
        )

    def forward(self, x):
        return self.swin(x)
