"""
Xác thực ảnh: CHỈ phim X-quang ngực (đề tài chẩn đoán bệnh phổi).
1) CLIP zero-shot — quyết định chính; ngưỡng vừa phải
2) Lọc ảnh màu rõ (selfie, cảnh) — chỉ khi CLIP cũng không nhận X-quang
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CHEST_XRAY_PROMPTS = [
    'a chest x-ray radiograph showing lungs ribs and heart silhouette',
    'a frontal posteroanterior PA chest x-ray for pulmonary diagnosis',
    'an anteroposterior AP chest radiograph of the thorax',
    'a grayscale chest X-ray film with clear lung fields',
    'a hospital chest radiograph used to detect lung diseases',
    'a portable chest x-ray image of adult thoracic cavity',
    'a lateral chest x-ray radiograph',
    'a black and white medical radiograph of human chest and lungs',
]

NON_CHEST_PROMPTS = [
    'a color photograph of a young man sitting on a balcony wearing a t-shirt',
    'a color selfie portrait photo of a person indoors',
    'a natural color photograph of people pets or outdoor scenery',
    'a color photo of food restaurant furniture or street',
    'a color landscape with blue sky green trees and sunlight',
    'a webcam or phone camera photo of everyday life',
    'a CT scan axial slice computed tomography',
    'a brain MRI magnetic resonance image',
    'a dental x-ray showing teeth',
    'a bone x-ray of arm leg hand knee or ankle',
    'a mammogram breast scan',
    'an abdominal x-ray not a chest film',
    'a spine lumbar radiograph',
]

# Softmax trên TOÀN BỘ prompt (chuẩn CLIP zero-shot) — cân bằng: chặn ảnh lạ, không chặn X-quang JPG
MIN_CHEST_PROB = 0.30
MIN_MARGIN = 0.04
# CLIP đủ tin → bỏ qua kiểm tra màu (tránh false positive trên phim nén / chụp màn hình)
CLIP_OVERRIDE_PROB = 0.42


def _color_photo_metrics(image: Image.Image) -> dict:
    """Đo độ 'ảnh màu thường' — chỉ dùng kết hợp CLIP, không chặn cứng một mình."""
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    if rgb.size == 0:
        return {'frac_colorful': 0.0, 'mean_sat': 0.0, 'channel_diff': 0.0, 'grayscale_like': True}

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    channel_diff = float(np.mean(np.abs(r - g) + np.abs(g - b)) / 2.0)

    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    saturation = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1e-3)
    mean_sat = float(np.mean(saturation))
    frac_colorful = float(np.mean(saturation > 0.22))

    grayscale_like = mean_sat < 0.14 and frac_colorful < 0.10 and channel_diff < 22.0

    return {
        'frac_colorful': frac_colorful,
        'mean_sat': mean_sat,
        'channel_diff': channel_diff,
        'grayscale_like': grayscale_like,
    }


def _is_obvious_color_photo(metrics: dict) -> bool:
    """Chỉ ảnh màu rõ (selfie, cảnh) — ngưỡng cao để không nhầm X-quang JPG."""
    if metrics.get('grayscale_like'):
        return False
    frac = metrics['frac_colorful']
    sat = metrics['mean_sat']
    diff = metrics['channel_diff']
    if frac > 0.28:
        return True
    if frac > 0.18 and sat > 0.16 and diff > 18.0:
        return True
    if diff > 28.0 and sat > 0.14:
        return True
    return False


class ClipChestXRayValidator:
    def __init__(self, model_name: str = 'ViT-B-32', pretrained: str = 'openai') -> None:
        import open_clip

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[X-Ray AI Validator] Đang tải CLIP {model_name}...')

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            chest_tokens = tokenizer(CHEST_XRAY_PROMPTS).to(self.device)
            non_tokens = tokenizer(NON_CHEST_PROMPTS).to(self.device)
            self.chest_features = self._encode_text(chest_tokens)
            self.non_features = self._encode_text(non_tokens)
            self.logit_scale = self.model.logit_scale.exp()
            self.num_chest = len(CHEST_XRAY_PROMPTS)

        print('[X-Ray AI Validator] ✅ CLIP + lọc ảnh màu — chỉ X-quang ngực')

    def _encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    def _clip_scores(self, image: Image.Image) -> dict:
        tensor = self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.model.encode_image(tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            chest_raw = (self.logit_scale * (img_feat @ self.chest_features.T)).squeeze(0)
            non_raw = (self.logit_scale * (img_feat @ self.non_features.T)).squeeze(0)

            all_logits = torch.cat([chest_raw, non_raw])
            all_probs = F.softmax(all_logits, dim=0)

            chest_prob = float(all_probs[: self.num_chest].sum().item())
            non_prob = float(all_probs[self.num_chest :].sum().item())

            return {
                'chest_prob': chest_prob,
                'non_prob': non_prob,
                'chest_max': float(chest_raw.max().item()),
                'non_max': float(non_raw.max().item()),
                'best_non_idx': int(non_raw.argmax().item()),
            }

    def validate(self, image: Image.Image) -> Tuple[bool, float, str]:
        if min(image.size) < 64:
            return False, 0.0, 'Ảnh quá nhỏ. Vui lòng upload phim X-quang ngực (PA/AP).'

        color_metrics = _color_photo_metrics(image)
        scores = self._clip_scores(image)
        chest_prob = scores['chest_prob']
        margin = chest_prob - scores['non_prob']

        # CLIP tin là X-quang → chấp nhận (kể cả JPG nén, chụp màn hình phim)
        if chest_prob >= CLIP_OVERRIDE_PROB:
            print(
                f'[X-Ray AI Validator] PASS (CLIP override) chest={chest_prob * 100:.1f}% '
                f'gray_like={color_metrics["grayscale_like"]}'
            )
            return True, chest_prob, ''

        obvious_color = _is_obvious_color_photo(color_metrics)
        accepted = chest_prob >= MIN_CHEST_PROB and margin >= MIN_MARGIN

        print(
            f'[X-Ray AI Validator] CLIP chest={chest_prob * 100:.1f}% '
            f'non={scores["non_prob"] * 100:.1f}% margin={margin * 100:.1f}% '
            f'colorful={color_metrics["frac_colorful"] * 100:.1f}% gray_like={color_metrics["grayscale_like"]} '
            f'→ {"PASS" if accepted else "REJECT"}'
        )

        if accepted:
            return True, chest_prob, ''

        if obvious_color and chest_prob < CLIP_OVERRIDE_PROB:
            print('[X-Ray AI Validator] REJECT — ảnh màu + CLIP không nhận X-quang')
            return (
                False,
                chest_prob,
                'AI từ chối: đây là ảnh chụp thường (có màu), không phải phim X-quang ngực. '
                'Vui lòng upload file X-quang phổi PA hoặc AP.',
            )

        return (
            False,
            chest_prob,
            'AI từ chối: không phải phim X-quang ngực. '
            f'Độ tin cậy X-quang {chest_prob * 100:.0f}% (cần ≥ {MIN_CHEST_PROB * 100:.0f}%). '
            'Chỉ upload X-quang lồng ngực — không dùng ảnh thường, CT, hay X-quang vùng khác.',
        )


_validator: Optional[ClipChestXRayValidator] = None
_validator_error: Optional[str] = None


def is_ai_validation_enabled() -> bool:
    return os.environ.get('XRAY_AI_VALIDATE', 'true').strip().lower() not in (
        '0',
        'false',
        'no',
        'off',
    )


def is_strict_validation() -> bool:
    return os.environ.get('XRAY_VALIDATE_STRICT', 'true').strip().lower() not in (
        '0',
        'false',
        'no',
        'off',
    )


def preload_validator() -> bool:
    global _validator, _validator_error

    if not is_ai_validation_enabled():
        print('[X-Ray AI Validator] ⚠️ XRAY_AI_VALIDATE=false — MỌI ảnh đều được nhận!')
        return True

    try:
        if _validator is None:
            _validator = ClipChestXRayValidator()
        _validator_error = None
        return True
    except ImportError as exc:
        _validator_error = f'Thiếu open-clip-torch: {exc}'
        print(f'[X-Ray AI Validator] ❌ {_validator_error}')
        return False
    except Exception as exc:
        _validator_error = str(exc)
        print(f'[X-Ray AI Validator] ❌ {exc}')
        return False


def get_validator_status() -> dict:
    return {
        'enabled': is_ai_validation_enabled(),
        'strict': is_strict_validation(),
        'loaded': _validator is not None,
        'error': _validator_error,
        'method': 'clip-zero-shot + color-photo-guard',
    }


def validate_chest_xray_image(image: Image.Image) -> Tuple[bool, float, str]:
    global _validator

    if not is_ai_validation_enabled():
        return True, 1.0, ''

    if _validator is None and not preload_validator():
        return (
            False,
            0.0,
            'Dịch vụ xác thực ảnh AI chưa sẵn sàng. '
            'Cài: pip install open-clip-torch rồi khởi động lại: py main.py',
        )

    return _validator.validate(image)
