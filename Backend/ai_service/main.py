import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Nạp Backend/ai_service/.env (XRAY_AI_VALIDATE=true, …)
# override=True: file .env luôn bật validate (ghi đè XRAY_AI_VALIDATE=false trong terminal)
load_dotenv(Path(__file__).resolve().parent / '.env', override=True)
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import CheXNetInference, InvalidXRayImageError  # noqa: E402

app = FastAPI(
    title="CheXNet AI — Chẩn đoán bệnh phổi từ X-quang ngực",
    description="Chỉ hỗ trợ phim X-quang ngực (PA/AP/Nghiêng). Phân loại 15 bệnh lý phổi bằng CheXNet.",
    version="1.0.0"
)

# Cấu hình CORS để cho phép kết nối nội bộ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_origin_regex=".*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo CheXNet + CLIP validator (chỉ nhận X-quang ngực)
try:
    classifier = CheXNetInference()
except Exception as e:
    print(f"⚠️ Không thể khởi tạo CheXNet: {e}")
    classifier = None

from xray_validator import preload_validator, get_validator_status

if classifier is not None:
    preload_validator()

@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    """
    Nhận file ảnh X-ray, xử lý thông tin chẩn đoán và trả về kết quả
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="AI model is currently not loaded. Please check logs for errors."
        )

    validation = get_validator_status()
    if not validation['enabled']:
        raise HTTPException(
            status_code=503,
            detail='Xác thực ảnh AI đang tắt (XRAY_AI_VALIDATE). Bật lại để chỉ nhận X-quang ngực.',
        )
    if not validation['loaded']:
        raise HTTPException(
            status_code=503,
            detail=(
                'Dịch vụ xác thực ảnh AI chưa sẵn sàng. '
                f"Lỗi: {validation.get('error') or 'unknown'}. "
                'Chạy: pip install open-clip-torch && py main.py'
            ),
        )

    start_time = time.time()
    
    # Tạo file tạm thời để lưu ảnh upload
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
        
    try:
        predictions = classifier.predict(temp_path)
        processing_time = time.time() - start_time

        return {
            "predictions": predictions,
            "processingTime": round(processing_time, 2),
            "modelVersion": classifier.model_version,
            "isValidXRay": True,
            "validationMethod": "clip-zero-shot",
        }
    except InvalidXRayImageError as e:
        raise HTTPException(status_code=422, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running inference: {str(e)}"
        )
    finally:
        # Dọn dẹp file tạm
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
def health_check():
    """
    Endpoint kiểm tra tình trạng dịch vụ
    """
    validation = get_validator_status()
    ai_ready = classifier is not None and (
        not validation['enabled'] or validation['loaded'] or not validation['strict']
    )
    return {
        "status": "healthy" if ai_ready else "degraded",
        "modelLoaded": classifier is not None,
        "modelVersion": classifier.model_version if classifier is not None else "none",
        "device": str(classifier.device) if classifier is not None else "none",
        "service": "CheXNet — Chẩn đoán bệnh phổi từ X-quang ngực",
        "imageValidation": {
            "method": "clip-zero-shot",
            **validation,
        },
    }

if __name__ == "__main__":
    import uvicorn

    print("🚀 CheXNet AI Service — http://0.0.0.0:8000 (Ctrl+C để dừng)")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\n[AI Service] Đã dừng an toàn.")
