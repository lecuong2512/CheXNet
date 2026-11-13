"""Decode supported web images before handing pixels to the original Hybrid pipeline."""
import io

import numpy as np
import pydicom
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian
from pydicom.errors import InvalidDicomError

MAX_PIXELS = 20_000_000
SUPPORTED_SYNTAXES = {ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian}


def decode_upload(contents: bytes, filename: str) -> tuple[Image.Image, bytes | None]:
    """Return RGB pixels and a metadata-free PNG only when the source is DICOM."""
    is_dicom = filename.lower().endswith((".dcm", ".dicom")) or contents[128:132] == b"DICM"
    if not is_dicom:
        try:
            with Image.open(io.BytesIO(contents)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise HTTPException(422, "Ảnh vượt quá giới hạn 20 triệu pixel")
                if source.format not in {"PNG", "JPEG"}:
                    raise HTTPException(422, "Chỉ hỗ trợ PNG, JPEG hoặc DICOM")
                return source.convert("RGB"), None
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, "Không đọc được ảnh") from exc
    try:
        dataset = pydicom.dcmread(io.BytesIO(contents))
        if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
            raise HTTPException(422, "Chỉ hỗ trợ DICOM đơn khung")
        if int(getattr(dataset, "SamplesPerPixel", 1)) != 1 or dataset.PhotometricInterpretation not in {
            "MONOCHROME1", "MONOCHROME2"
        }:
            raise HTTPException(422, "Chỉ hỗ trợ DICOM grayscale MONOCHROME1/2")
        if dataset.file_meta.TransferSyntaxUID not in SUPPORTED_SYNTAXES:
            raise HTTPException(422, "DICOM nén chưa được hỗ trợ. Vui lòng xuất bản không nén.")
        if not 0 < int(dataset.Rows) * int(dataset.Columns) <= MAX_PIXELS:
            raise HTTPException(422, "DICOM vượt quá giới hạn 20 triệu pixel")
        pixels = dataset.pixel_array
        valid = np.ones(pixels.shape, dtype=bool)
        if hasattr(dataset, "PixelPaddingValue"):
            low = min(dataset.PixelPaddingValue, getattr(dataset, "PixelPaddingRangeLimit", dataset.PixelPaddingValue))
            high = max(dataset.PixelPaddingValue, getattr(dataset, "PixelPaddingRangeLimit", dataset.PixelPaddingValue))
            valid = (pixels < low) | (pixels > high)
        values = np.asarray(apply_voi_lut(apply_modality_lut(pixels, dataset), dataset), dtype=np.float64)
        if not valid.any() or not np.isfinite(values).all():
            raise HTTPException(422, "DICOM không có pixel hợp lệ")
        lower, upper = float(values[valid].min()), float(values[valid].max())
        if upper <= lower:
            raise HTTPException(422, "DICOM không có độ tương phản")
        output = np.clip((values - lower) / (upper - lower), 0, 1)
        if dataset.PhotometricInterpretation == "MONOCHROME1":
            output = 1 - output
        output[~valid] = 0
        image = Image.fromarray(np.rint(output * 255).astype(np.uint8)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return image, buffer.getvalue()
    except HTTPException:
        raise
    except (InvalidDicomError, ValueError, AttributeError, KeyError, OSError, RuntimeError, NotImplementedError) as exc:
        raise HTTPException(422, "DICOM không hợp lệ hoặc thiếu dữ liệu điểm ảnh") from exc
