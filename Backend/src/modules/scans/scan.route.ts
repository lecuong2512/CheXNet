import { Router } from 'express';
import multer from 'multer';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { uploadScan, getScansByPatient } from './scan.controller';
import { verifyAdminOrDoctor } from '@common/middlewares/author.middleware';
import { UPLOAD_DIR, MAX_FILE_SIZE_MB } from '@config/environment';

// Cấu hình multer disk storage
const storage = multer.diskStorage({
    destination: (_req, _file, cb) => {
        cb(null, UPLOAD_DIR);
    },
    filename: (_req, file, cb) => {
        const ext = path.extname(file.originalname);
        cb(null, `scan_${uuidv4()}${ext}`);
    },
});

const fileFilter = (_req: any, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/dicom', 'application/dicom'];
    if (allowedTypes.includes(file.mimetype)) {
        cb(null, true);
    } else {
        cb(new Error('Chỉ chấp nhận phim X-quang ngực (JPEG, PNG hoặc DICOM)'));
    }
};

const upload = multer({
    storage,
    fileFilter,
    limits: { fileSize: MAX_FILE_SIZE_MB * 1024 * 1024 },
});

const router = Router();

router.post('/:patientId/upload', verifyAdminOrDoctor, upload.single('image'), uploadScan);
router.get('/:patientId', getScansByPatient);

export default router;
