import { Router } from 'express';
import { getDiagnosesByPatient, getDiagnosisById, verifyDiagnosis, getAllDiagnoses } from './diagnosis.controller';
import { verifyAdminOrDoctor } from '@common/middlewares/author.middleware';
import { renderDiagnosisPdf } from './diagnosis.pdf';
import { catchAsync } from '@common/utils/catchAsync';
import { saveReferenceCase } from './diagnosis.reference';
import DiagnosisModel from './diagnosis.model';
import { DiagnosisService } from './diagnosis.service';

const router = Router();

router.get('/', getAllDiagnoses);
router.get('/by-patient/:patientId', getDiagnosesByPatient);
router.get('/:id', getDiagnosisById);
router.get('/:id/similar-cases', catchAsync(async (req, res) => {
    const { data } = await new DiagnosisService().getById(String(req.params.id));
    const cases = (data?.hybridAnalysis?.similar_cases || []) as { filename: string; similarity: number; diseases: string[] }[];
    const references = await DiagnosisModel.find({
        'referenceCase.referenceId': { $in: cases.map(item => item.filename) }, 'referenceCase.status': 'ready',
    }).select('scanId patientId referenceCase').populate<{ patientId: { patientCode: string } | null }>('patientId', 'patientCode');
    res.json({ success: true, data: cases.map(item => {
        const reference = references.find(record => record.referenceCase?.referenceId === item.filename);
        return { ...item, sourceCase: reference?.patientId ? {
            patientCode: reference.patientId.patientCode, scanId: String(reference.scanId),
        } : undefined };
    }) });
}));
router.get('/:id/report.pdf', catchAsync(async (req, res) => {
    const result = await renderDiagnosisPdf(String(req.params.id), typeof req.query.revision === 'string' ? req.query.revision : undefined);
    res.setHeader('Cache-Control', 'private, no-store');
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename="chexnet-${req.params.id}-v${result.revision}.pdf"`);
    res.send(result.pdf);
}));
router.patch('/:id/verify', verifyAdminOrDoctor, verifyDiagnosis);
router.post('/:id/reference-case', verifyAdminOrDoctor, catchAsync(async (req, res) => {
    const result = await saveReferenceCase(String(req.params.id), req.body.diseases, req.users.userId);
    res.status(result.replayed ? 200 : 201).json(result);
}));

export default router;
