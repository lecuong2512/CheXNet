"""Run the inherited web contract alongside the unchanged Hybrid FastAPI app."""
import sys
import io
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name(".env"))

from fastapi import File, UploadFile, Form, HTTPException
from Backend import main as hybrid
from Backend.ai_service.inference import CheXNetInference, MODEL_VERSION
from Backend.ai_service.xray_validator import preload_validator, get_validator_status
from Backend.ai_service.runtime import InferenceRunner
from Backend.ai_service.reference_store import ReferenceStore
from Backend.ai_service.image_input import decode_upload
from Backend.ai_service.inference_trace import configure_audit_log

app = hybrid.app
classifier = CheXNetInference()
runner = InferenceRunner()
reference_store = None
reference_error = None


@app.on_event("startup")
def initialize_web_validation():
    global reference_store, reference_error
    configure_audit_log(ROOT / '.runtime/inference-audit.log')
    preload_validator()
    try:
        reference_store = ReferenceStore(hybrid.cbir_finder,
            os.environ.get("CBIR_STORE_ROOT", str(ROOT / "Backend/data/reference-generations")))
        hybrid.cbir_finder = reference_store
    except Exception as exc:
        reference_error = str(exc)
        hybrid.cbir_finder.available = False


@app.on_event("shutdown")
def close_reference_store():
    if reference_store is not None:
        reference_store.close()


async def add_reference(file, patient_id, diseases, strict_id=False):
    if reference_store is None or not reference_store.available:
        raise HTTPException(503, "Kho ca tham chiếu chưa sẵn sàng")
    if strict_id and not re.fullmatch(r"ref_[a-f0-9]{24}", patient_id):
        raise HTTPException(422, "Mã ca tham chiếu không hợp lệ")
    labels = sorted(set(label.strip() for label in diseases.split(",") if label.strip()))
    if strict_id and (not labels or any(label not in hybrid.CLASS_NAMES for label in labels)):
        raise HTTPException(422, "Chọn ít nhất một nhãn xác nhận hợp lệ")
    if strict_id and "No Finding" in labels and len(labels) > 1:
        raise HTTPException(422, "Nhãn No Finding phải được xác nhận riêng")
    existing = reference_store.reference(patient_id) if patient_id.startswith("ref_") else None
    if existing:
        if sorted(existing.get("diseases", [])) != labels:
            raise HTTPException(409, "Ca đã lưu với nhãn xác nhận khác")
        return {"success": True, "referenceId": patient_id, "replayed": True}
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, "Ảnh vượt quá dung lượng 10 MB")
    _, preview = decode_upload(contents, file.filename or "")
    normalized = UploadFile(filename="reference.png" if preview else file.filename,
                            file=io.BytesIO(preview or contents))
    try:
        result = await hybrid.add_reference_case(normalized, patient_id, ",".join(labels))
    finally:
        await normalized.close()
    if not result.get("success"):
        raise HTTPException(503, "Chưa lưu được ca tham chiếu. Vui lòng thử lại.")
    return {**result, "referenceId": patient_id, "replayed": False}


@app.post("/v3/reference-cases")
async def add_web_reference(file: UploadFile = File(...), reference_id: str = Form(...), diseases: str = Form(...)):
    return await runner.run(lambda: add_reference(file, reference_id, diseases, strict_id=True))


@app.get("/v3/reference-cases/{reference_id}")
def get_reference(reference_id: str):
    if reference_store is None:
        raise HTTPException(503, "Kho ca tham chiếu chưa sẵn sàng")
    metadata = reference_store.reference(reference_id)
    if metadata is None:
        raise HTTPException(404, "Chưa lưu ca tham chiếu")
    return {"success": True, "referenceId": reference_id, "diseases": metadata["diseases"]}


# Route all legacy additions through the same writer and inference worker.
app.router.routes[:] = [route for route in app.router.routes if getattr(route, "path", None) != "/api/add_reference_case"]


@app.post("/api/add_reference_case")
async def add_legacy_reference(file: UploadFile = File(...), patient_id: str = Form("Local_Case"), diseases: str = Form("")):
    return await runner.run(lambda: add_reference(file, patient_id, diseases))


@app.post("/v3/predict")
async def predict_xray(file: UploadFile = File(...)):
    return await runner.run(lambda: classifier.predict(file))


app.router.routes[:] = [route for route in app.router.routes if getattr(route, "path", None) != "/predict"]


@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    return await runner.run(lambda: hybrid.predict_image(file))


@app.get("/health")
def health_check():
    original = hybrid.health_check()
    validation = get_validator_status()
    ready = original["model_loaded"] and (not validation["enabled"] or validation["loaded"])
    return {
        **original,
        "status": "healthy" if ready else "degraded",
        "modelLoaded": original["model_loaded"],
        "modelVersion": MODEL_VERSION,
        "imageValidation": validation,
        "inference": runner.status(),
        "cbir": {"available": reference_store is not None and reference_store.available,
                 "count": len(reference_store.metadata) if reference_store is not None else 0,
                 "error": reference_error},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
