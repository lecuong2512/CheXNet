import { Request, Response } from 'express';
import { catchAsync } from '@common/utils/catchAsync';
import { DiagnosisService } from './diagnosis.service';

const diagnosisService = new DiagnosisService();

export const getDiagnosesByPatient = catchAsync(async (req: Request, res: Response) => {
    const filter = {
        patientId: String(req.params.patientId),
        page: req.query.page ? parseInt(String(req.query.page), 10) : 1,
        limit: req.query.limit ? parseInt(String(req.query.limit), 10) : 10,
        status: typeof req.query.status === 'string' ? (req.query.status as any) : undefined,
    };
    const result = await diagnosisService.getByPatient(filter);
    res.status(200).json(result);
});

export const getDiagnosisById = catchAsync(async (req: Request, res: Response) => {
    const result = await diagnosisService.getById(String(req.params.id));
    res.status(200).json(result);
});

export const verifyDiagnosis = catchAsync(async (req: Request, res: Response) => {
    const { status, notes } = req.body;
    const result = await diagnosisService.verify(String(req.params.id), status, notes, req.users.userId);
    res.status(200).json(result);
});

export const getAllDiagnoses = catchAsync(async (req: Request, res: Response) => {
    const filter = {
        page: req.query.page ? parseInt(String(req.query.page), 10) : 1,
        limit: req.query.limit ? parseInt(String(req.query.limit), 10) : 10,
        status: typeof req.query.status === 'string' ? (req.query.status as any) : undefined,
        keyword: typeof req.query.keyword === 'string' ? req.query.keyword : undefined,
    };
    const result = await diagnosisService.getAll(filter);
    res.status(200).json(result);
});
