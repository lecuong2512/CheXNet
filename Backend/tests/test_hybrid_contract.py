"""Contract tests without loading a second copy of the 2 GB checkpoint."""
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException, UploadFile
from PIL import Image


class HybridContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.hybrid = types.ModuleType("Backend.main")
        self.hybrid.model = object()
        self.hybrid.CLASS_NAMES = [f"class_{i}" for i in range(15)]
        self.analysis = {
            "success": True,
            "all_probabilities": [
                {"disease": name, "probability": i / 20}
                for i, name in enumerate(self.hybrid.CLASS_NAMES)
            ],
            "predictions": [{"disease": "class_14", "heatmap": "original-overlay"}],
            "clinical_report": {"summary": "original-report"},
            "yolo_detections": [{"cascade_score": 0.42}],
            "similar_cases": [{"filename": "reference-case"}],
        }
        self.hybrid.predict_image = AsyncMock(return_value=self.analysis)
        self.validator = types.ModuleType("Backend.ai_service.xray_validator")
        self.validator.get_validator_status = Mock(return_value={"enabled": False})
        self.validator.validate_chest_xray_image = Mock(return_value=(False, 0.01, "Not a chest X-ray"))
        self.modules = patch.dict(sys.modules, {
            "Backend.main": self.hybrid,
            "Backend.ai_service.xray_validator": self.validator,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        path = Path(__file__).resolve().parents[1] / "ai_service" / "inference.py"
        spec = importlib.util.spec_from_file_location("hybrid_contract_under_test", path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        self.classifier = adapter.CheXNetInference()

    def upload(self):
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buffer, format="PNG")
        buffer.seek(0)
        return UploadFile(filename="test.png", file=buffer)

    async def test_keeps_all_original_probabilities_and_analysis(self):
        result = await self.classifier.predict(self.upload())
        self.assertIs(result["hybridAnalysis"], self.analysis)
        expected = {p["disease"]: p["probability"] for p in self.analysis["all_probabilities"]}
        self.assertEqual({p["className"]: p["probability"] for p in result["predictions"]}, expected)
        self.assertIsNone(result["isValidXRay"])
        self.assertEqual(result["inferenceMeta"]["contractVersion"], 2)
        self.assertEqual(result["inferenceMeta"]["detectorStatus"], "detections")

    async def test_empty_detector_results_do_not_claim_successful_detection(self):
        self.analysis["yolo_detections"] = []
        result = await self.classifier.predict(self.upload())
        self.assertEqual(result["inferenceMeta"]["detectorStatus"], "unavailable")
        self.hybrid.yolo_detector = types.SimpleNamespace(available=True)
        result = await self.classifier.predict(self.upload())
        self.assertEqual(result["inferenceMeta"]["detectorStatus"], "loaded-unconfirmed")

    async def test_model_failure_never_becomes_a_successful_diagnosis(self):
        self.hybrid.predict_image.return_value = {"error": "model failed"}
        with self.assertRaises(HTTPException) as error:
            await self.classifier.predict(self.upload())
        self.assertEqual(error.exception.status_code, 500)

    async def test_incomplete_probabilities_are_rejected(self):
        self.analysis["all_probabilities"].pop()
        with self.assertRaises(HTTPException) as error:
            await self.classifier.predict(self.upload())
        self.assertEqual(error.exception.status_code, 500)

    async def test_rejected_image_never_reaches_hybrid_model(self):
        self.validator.get_validator_status.return_value = {"enabled": True, "loaded": True}
        with self.assertRaises(HTTPException) as error:
            await self.classifier.predict(self.upload())
        self.assertEqual(error.exception.status_code, 422)
        self.hybrid.predict_image.assert_not_called()

    async def test_dicom_uses_the_same_decoded_pixels_for_validation_and_inference(self):
        import numpy as np
        from test_image_input import dicom_bytes
        from Backend.ai_service.image_input import decode_upload
        data = dicom_bytes(np.arange(16).reshape(4, 4), "MONOCHROME1")
        expected, preview = decode_upload(data, "test.dcm")
        self.validator.get_validator_status.return_value = {"enabled": True, "loaded": True}
        self.validator.validate_chest_xray_image.return_value = (True, 0.9, "")
        received = []

        async def read_upload(upload):
            received.append(await upload.read())
            return self.analysis

        self.hybrid.predict_image.side_effect = read_upload
        result = await self.classifier.predict(UploadFile(filename="test.dcm", file=io.BytesIO(data)))
        self.assertEqual(received, [preview])
        validated_image = self.validator.validate_chest_xray_image.call_args.args[0]
        self.assertEqual(validated_image.tobytes(), expected.tobytes())
        import base64
        self.assertEqual(base64.b64decode(result["previewPng"]), preview)


if __name__ == "__main__":
    unittest.main()
