import { Request, Response } from 'express';
import { catchAsync } from '@common/utils/catchAsync';
import { ApiError } from '@common/utils/ApiError';
import { ScanService } from './scan.service';

const scanService = new ScanService();

export const uploadScan = catchAsync(async (req: Request, res: Response) => {
    if (!req.file) {
        throw new ApiError(400, 'Vui lòng chọn file ảnh X-ray để upload');
    }
    const patientId = String(req.params.patientId);
    const result = await scanService.uploadAndProcess(patientId, req.users.userId, req.file, req.body);
    res.status(201).json(result);
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
