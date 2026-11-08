import { Request, Response } from 'express';
import { catchAsync } from '@common/utils/catchAsync';
import { ApiError } from '@common/utils/ApiError';
import { ScanService } from './scan.service';
import path from 'path';
import fs from 'fs';
import { Types } from 'mongoose';
import ScanModel from './scan.model';
import { PRIVATE_UPLOAD_DIR, UPLOAD_DIR } from '@config/environment';
import { ScanStatus } from '@common/utils/enum';

const scanService = new ScanService();

export const getScanPreview = catchAsync(async (req: Request, res: Response) => {
    const id = String(req.params.scanId);
    if (!Types.ObjectId.isValid(id)) throw new ApiError(400, 'Mã lần chụp không hợp lệ');
    const scan = await ScanModel.findOne({ _id: id, status: ScanStatus.DONE });
    if (!scan) throw new ApiError(404, 'Không tìm thấy lần chụp');
    const preview = path.resolve(scan.previewPath || scan.imagePath);
    const roots = [PRIVATE_UPLOAD_DIR, UPLOAD_DIR].map(root => path.resolve(root) + path.sep);
    if (!roots.some(root => preview.startsWith(root)) || !fs.existsSync(preview)) {
        throw new ApiError(404, 'Không tìm thấy ảnh xem lại');
    }
    if (!scan.previewPath && /\.(dcm|dicom)$/i.test(preview)) {
        throw new ApiError(422, 'Lần chụp cũ chưa có bản xem DICOM');
    }
    res.setHeader('Cache-Control', 'private, no-store');
    res.sendFile(preview);
});

export const uploadScan = catchAsync(async (req: Request, res: Response) => {
    if (!req.file) {
        throw new ApiError(400, 'Vui lòng chọn file ảnh X-ray để upload');
    }
    const patientId = String(req.params.patientId);
    const result = await scanService.uploadAndProcess(patientId, req.users.userId, req.file, {
        ...req.body,
        requestKey: req.get('Idempotency-Key') || req.body.requestKey,
    });
    res.status(result.replayed ? 200 : 201).json(result);
});

export const getScansByPatient = catchAsync(async (req: Request, res: Response) => {
    const filter = {
        patientId: String(req.params.patientId),
        page: req.query.page ? parseInt(String(req.query.page), 10) : 1,
        limit: req.query.limit ? parseInt(String(req.query.limit), 10) : 10,
        status: typeof req.query.status === 'string' ? (req.query.status as any) : undefined,
        type: typeof req.query.type === 'string' ? (req.query.type as any) : undefined,
    };
    const result = await scanService.getByPatient(filter);
    res.status(200).json(result);
});
