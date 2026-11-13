import { Router } from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { uploadScan, getScansByPatient, getScanPreview } from './scan.controller';
import { verifyAdminOrDoctor } from '@common/middlewares/author.middleware';
import { PRIVATE_UPLOAD_DIR, MAX_FILE_SIZE_MB } from '@config/environment';
import { ApiError } from '@common/utils/ApiError';
import { catchAsync } from '@common/utils/catchAsync';
import ScanModel from './scan.model';
import { Types } from 'mongoose';

// Cấu hình multer disk storage
const storage = multer.diskStorage({
    destination: (_req, _file, cb) => {
        fs.mkdirSync(PRIVATE_UPLOAD_DIR, { recursive: true });
        cb(null, PRIVATE_UPLOAD_DIR);
    },
    filename: (_req, file, cb) => {
        const ext = path.extname(file.originalname);
        cb(null, `scan_${uuidv4()}${ext}`);
    },
});

const fileFilter = (_req: any, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/dicom', 'application/dicom'];
    const dicomName = /\.(dcm|dicom)$/i.test(file.originalname);
    if (allowedTypes.includes(file.mimetype) || (dicomName && ['', 'application/octet-stream'].includes(file.mimetype))) {
        cb(null, true);
    } else {
        cb(new ApiError(422, 'Chỉ chấp nhận phim X-quang ngực (JPEG, PNG hoặc DICOM)'));
    }
};

const upload = multer({
    storage,
    fileFilter,
    limits: { fileSize: MAX_FILE_SIZE_MB * 1024 * 1024 },
});

const router = Router();

router.post('/:patientId/upload', verifyAdminOrDoctor, upload.single('image'), uploadScan);
router.get('/:scanId/preview', getScanPreview);
router.get('/detail/:scanId', catchAsync(async (req, res) => {
    const id = String(req.params.scanId);
    if (!Types.ObjectId.isValid(id)) throw new ApiError(400, 'Mã lần chụp không hợp lệ');
    const [scan] = await ScanModel.aggregate([
        { $match: { _id: new Types.ObjectId(id), status: 'done' } },
        { $lookup: { from: 'diagnoses', localField: '_id', foreignField: 'scanId', as: 'diagnosis' } },
        { $unwind: '$diagnosis' },
    ]);
    if (!scan) throw new ApiError(404, 'Không tìm thấy lần chụp hoàn tất');
    res.json({ success: true, data: scan });
}));
router.get('/:patientId', getScansByPatient);

export default router;
