"""Adapt the inherited web API to the original Hybrid inference pipeline.

Model loading, preprocessing, thresholds and all post-processing stay in
Backend.main. This adapter only validates the upload and reshapes its response.
"""
import io
import base64
import math
import time

from fastapi import HTTPException, UploadFile
from Backend.ai_service.image_input import decode_upload
from Backend import main as hybrid
from Backend.ai_service.xray_validator import get_validator_status, validate_chest_xray_image

MODEL_VERSION = "hybrid-convnextv2-swinv2"


class CheXNetInference:
    async def predict(self, file: UploadFile) -> dict:
        if hybrid.model is None:
            raise HTTPException(503, "Hybrid model is not loaded")
        started = time.perf_counter()
        contents = await file.read()
        await file.seek(0)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(413, "Ảnh vượt quá dung lượng 10 MB")
        image, preview = decode_upload(contents, file.filename or "")
        validation = get_validator_status()
        if validation["enabled"]:
            if not validation["loaded"]:
                raise HTTPException(503, "Chest X-ray validator is not ready: " + str(validation.get("error")))
            valid, _, message = validate_chest_xray_image(image)
            if not valid:
                raise HTTPException(422, message)
        if preview is not None:
            normalized = UploadFile(filename="normalized.png", file=io.BytesIO(preview))
            try:
                analysis = await hybrid.predict_image(normalized)
            finally:
                await normalized.close()
        else:
            analysis = await hybrid.predict_image(file)
        if "error" in analysis or not analysis.get("success"):
            raise HTTPException(500, analysis.get("error", "Hybrid inference failed"))
        probabilities = analysis.get("all_probabilities", [])
        if len(probabilities) != len(hybrid.CLASS_NAMES) or {p["disease"] for p in probabilities} != set(hybrid.CLASS_NAMES):
            raise HTTPException(500, "Hybrid inference returned incomplete probabilities")
        if any(not math.isfinite(p["probability"]) or not 0 <= p["probability"] <= 1 for p in probabilities):
            raise HTTPException(500, "Hybrid inference returned invalid probabilities")
        return {
            "success": True,
            "predictions": [{"className": p["disease"], "probability": p["probability"]} for p in probabilities],
            "processingTime": round(time.perf_counter() - started, 3),
            "modelVersion": MODEL_VERSION,
            "isValidXRay": True if validation["enabled"] else None,
            "validationMethod": "clip-zero-shot" if validation["enabled"] else "disabled",
            "hybridAnalysis": analysis,
            "previewPng": base64.b64encode(preview).decode("ascii") if preview is not None else None,
            "inferenceMeta": {
                "contractVersion": 1,
                "device": str(getattr(hybrid, "device", "unknown")),
                "detectorStatus": (
                    "detections" if analysis.get("yolo_detections") else
                    "loaded-unconfirmed" if getattr(getattr(hybrid, "yolo_detector", None), "available", False)
                    else "unavailable"
                ),
            },
        }
