# checkpoint_utils.py
# Tiện ích load checkpoint dùng chung, tránh trùng lặp logic giữa nhiều file.
import torch
from collections import OrderedDict


def load_checkpoint_safe(path, device=None):
    """
    Load checkpoint an toàn: thử weights_only=True trước, fallback về False.
    Tự động xử lý numpy scalar safe globals.
    
    Args:
        path: Đường dẫn tới file .pth checkpoint
        device: torch.device để map_location (mặc định: auto detect cuda/cpu)
    
    Returns:
        dict: Nội dung checkpoint (có thể chứa 'state_dict', 'epoch', 'best_auroc', etc.)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        import numpy as _np
        from torch import serialization
        # NumPy 2.x dùng _np._core, NumPy 1.x dùng _np.core
        try:
            _scalar_cls = _np._core.multiarray.scalar
        except AttributeError:
            _scalar_cls = _np.core.multiarray.scalar
        serialization.add_safe_globals([_scalar_cls])
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def extract_state_dict(ckpt, strip_module_prefix=True):
    """
    Trích xuất state_dict từ checkpoint, tự động xử lý:
    - Key 'state_dict' hoặc 'model_state_dict' hoặc trực tiếp là state_dict
    - Tiền tố 'module.' từ DataParallel/DDP
    
    Args:
        ckpt: dict checkpoint đã load
        strip_module_prefix: Bỏ tiền tố 'module.' nếu có (mặc định True)
    
    Returns:
        OrderedDict: state_dict sạch, sẵn sàng cho model.load_state_dict()
    """
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    if strip_module_prefix and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = OrderedDict(
            (k[7:] if k.startswith('module.') else k, v)
            for k, v in state_dict.items()
        )
    
    return state_dict
