"""Tests for Hybrid localization fallback boxes and original-coordinate projection."""
import importlib.util
import sys
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
for path in [ROOT, ROOT / 'Backend', ROOT / 'Models']:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_SPEC = importlib.util.spec_from_file_location('backend_main_under_test', ROOT / 'Backend' / 'main.py')
hybrid_main = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hybrid_main)
from Backend.yolo_detector import YOLODetector  # noqa: E402


class _TinyFpnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn_merge = nn.Conv2d(3, 4, kernel_size=1)
        self.head = nn.Linear(4, 15)

    def forward(self, x):
        feats = torch.relu(self.fpn_merge(x))
        # Preserve spatial position sensitivity by weighting bottom-right higher for class 14 (Hernia)
        pooled = feats.mean(dim=(2, 3))
        logits = self.head(pooled)
        br_energy = feats[:, :, feats.shape[2] // 2:, feats.shape[3] // 2:].mean(dim=(2, 3))
        logits[:, 14] = br_energy.sum(dim=1)
        return logits, torch.zeros((x.shape[0], 15, feats.shape[2], feats.shape[3]), device=x.device)


class HybridFallbackBoxTests(unittest.TestCase):
    def test_project_crop_bbox_to_original_handles_nonsquare_center_crop(self):
        # 1060x960 -> Resize(437) gives (482, 437), CenterCrop(384) offsets (49, 26)
        projected = hybrid_main._project_crop_bbox_to_original(
            {'x1': 0, 'y1': 0, 'x2': 384, 'y2': 384},
            orig_size=(1060, 960),
            resized_size=(482, 437),
            crop_size=384,
        )
        self.assertEqual(projected, {'x1': 108, 'y1': 57, 'x2': 952, 'y2': 901})
        self.assertIsNone(
            hybrid_main._project_crop_bbox_to_original(
                {'x1': 50, 'y1': 50, 'x2': 50, 'y2': 80},
                orig_size=(1060, 960),
                resized_size=(482, 437),
                crop_size=384,
            )
        )

    def test_compute_fpn_gradcam_maps_does_not_mutate_frozen_params(self):
        model = _TinyFpnModel().eval()
        for param in model.parameters():
            param.requires_grad_(False)
        x = torch.zeros((1, 3, 24, 24))
        x[:, :, 14:22, 14:22] = 2.0
        maps = hybrid_main._compute_fpn_gradcam_maps(model, x, [14, 10])
        self.assertIn(14, maps)
        self.assertIn(10, maps)
        cam_hernia = maps[14]
        self.assertEqual(cam_hernia.shape, (24, 24))
        self.assertGreater(float(cam_hernia[14:22, 14:22].mean()), float(cam_hernia[0:8, 0:8].mean()))
        self.assertTrue(all(param.grad is None for param in model.parameters()))

    def test_select_hybrid_localization_uses_gradcam_for_non_vindr_and_rescues_blank_masks(self):
        blank = np.zeros((24, 24), dtype=np.float32)
        strong = np.zeros((24, 24), dtype=np.float32)
        strong[12:22, 12:22] = 0.9
        lung_mask = np.ones((384, 384), dtype=np.float32)

        # Even if raw attention has noise, GRADCAM_PREFERRED_DISEASES chooses Grad-CAM when available
        vis, boxes, src = hybrid_main._select_hybrid_localization('Hernia', strong * 0.5, blank, lung_mask, 384, strong)
        self.assertEqual(src, 'hybrid-gradcam')
        self.assertIs(vis, strong)
        self.assertGreaterEqual(len(boxes), 1)

        # Standard class: rescues wiped att_filtered using raw att_2d before Grad-CAM
        vis2, boxes2, src2 = hybrid_main._select_hybrid_localization('Effusion', strong, blank, lung_mask, 384, None)
        self.assertEqual(src2, 'hybrid-attention')
        self.assertIs(vis2, strong)
        self.assertGreaterEqual(len(boxes2), 1)

    def test_collect_hybrid_fallback_boxes_filters_tiny_and_caps_count(self):
        raw = [
            {'x1': 200, 'y1': 200, 'x2': 380, 'y2': 380, 'confidence': 0.8},
            {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150, 'confidence': 0.6},
            {'x1': 10, 'y1': 10, 'x2': 80, 'y2': 80, 'confidence': 0.4},
            {'x1': 1, 'y1': 1, 'x2': 3, 'y2': 3, 'confidence': 0.9},  # < 0.2% area, dropped
        ]
        # Re-order so tiny box is skipped and max 2 kept
        collected = hybrid_main._collect_hybrid_fallback_boxes(
            'Hernia', 0.96, [raw[3], raw[0], raw[1], raw[2]], 'hybrid-gradcam',
            orig_size=(447, 447), resized_size=(437, 437), crop_size=384, max_boxes=2,
        )
        self.assertEqual(len(collected), 2)
        self.assertEqual(collected[0]['source'], 'hybrid-gradcam')
        self.assertEqual(collected[0]['disease'], 'Hernia')
        self.assertAlmostEqual(collected[0]['probability'], 0.96)

    def test_yolo_draw_boxes_renders_both_yolo_and_hybrid_entries(self):
        detector = YOLODetector.__new__(YOLODetector)
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        rendered = detector.draw_boxes(canvas, [
            {'display_name': 'Cardiomegaly', 'confidence': 0.8, 'cascade_score': 0.7, 'bbox': {'x1': 20, 'y1': 40, 'x2': 120, 'y2': 160}},
            {'disease': 'Hernia', 'display_name': 'Hernia', 'probability': 0.96, 'confidence': 0.61, 'source': 'hybrid-gradcam', 'bbox': {'x1': 90, 'y1': 90, 'x2': 180, 'y2': 180}},
        ])
        self.assertEqual(rendered.shape, canvas.shape)
        self.assertGreater(int(rendered.sum()), 0)


if __name__ == '__main__':
    unittest.main()
