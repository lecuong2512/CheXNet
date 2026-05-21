import { Request, Response } from 'express';
import { catchAsync } from '@common/utils/catchAsync';
import { PatientService } from './patient.service';

const patientService = new PatientService();

export const getPatients = catchAsync(async (req: Request, res: Response) => {
    const filter = {
        page: req.query.page ? parseInt(String(req.query.page), 10) : 1,
        limit: req.query.limit ? parseInt(String(req.query.limit), 10) : 10,
        keyword: typeof req.query.keyword === 'string' ? req.query.keyword : undefined,
        gender: typeof req.query.gender === 'string' ? (req.query.gender as any) : undefined,
        department: typeof req.query.department === 'string' ? req.query.department : undefined,
        riskLevel: typeof req.query.riskLevel === 'string' ? (req.query.riskLevel as any) : undefined,
    };
    const result = await patientService.getByFilter(filter);
    res.status(200).json(result);
});

export const getPatientById = catchAsync(async (req: Request, res: Response) => {
    const result = await patientService.getById(String(req.params.id));
    res.status(200).json(result);
});

export const createPatient = catchAsync(async (req: Request, res: Response) => {
    const result = await patientService.create(req.body, req.users.userId);
    res.status(201).json(result);
});

export const updatePatient = catchAsync(async (req: Request, res: Response) => {
    const result = await patientService.update(String(req.params.id), req.body);
    res.status(200).json(result);
});
