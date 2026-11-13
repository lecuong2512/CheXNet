import io
import unittest

import numpy as np
from fastapi import HTTPException
from PIL import Image
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from Backend.ai_service.image_input import decode_upload


def dicom_bytes(pixels, photometric="MONOCHROME2", **fields):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PatientName = "MUST NOT BE IN PREVIEW"
    dataset.PixelData = np.asarray(pixels, dtype="<u2").tobytes()
    for key, value in fields.items():
        setattr(dataset, key, value)
    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


class ImageInputTests(unittest.TestCase):
    def test_png_keeps_pixels_without_dicom_preview(self):
        image = Image.new("RGB", (8, 8), (45, 70, 30))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        decoded, preview = decode_upload(buffer.getvalue(), "test.png")
        self.assertEqual(decoded.tobytes(), image.tobytes())
        self.assertIsNone(preview)

    def test_monochrome_polarity_and_metadata_free_preview(self):
        pixels = np.array([[0, 100], [200, 300]], dtype=np.uint16)
        image, preview = decode_upload(dicom_bytes(pixels), "test.dcm")
        inverse, _ = decode_upload(dicom_bytes(pixels, "MONOCHROME1"), "test.dcm")
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(image.getpixel((1, 1)), (255, 255, 255))
        self.assertEqual(inverse.getpixel((0, 0)), (255, 255, 255))
        with Image.open(io.BytesIO(preview)) as png:
            self.assertEqual(png.tobytes(), image.tobytes())
            self.assertEqual(png.info, {})
        self.assertNotIn(b"MUST NOT", preview)

    def test_window_saturates_and_padding_is_black(self):
        image, _ = decode_upload(dicom_bytes(
            np.array([[0, 100], [200, 300]]), WindowCenter=150, WindowWidth=100,
            RescaleSlope=1, RescaleIntercept=0, PixelPaddingValue=0,
        ), "window.dcm")
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(image.getpixel((1, 0)), (0, 0, 0))
        self.assertEqual(image.getpixel((0, 1)), (255, 255, 255))

    def test_invalid_and_multiframe_and_constant_are_rejected(self):
        cases = [
            b"invalid",
            dicom_bytes(np.arange(4).reshape(2, 2), NumberOfFrames=2),
            dicom_bytes(np.ones((2, 2))),
        ]
        for data in cases:
            with self.subTest(data_size=len(data)), self.assertRaises(HTTPException) as error:
                decode_upload(data, "bad.dcm")
            self.assertEqual(error.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
