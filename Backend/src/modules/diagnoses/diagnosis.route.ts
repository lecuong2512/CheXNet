import { Router } from 'express';
import { getDiagnosesByPatient, getDiagnosisById, verifyDiagnosis, getAllDiagnoses } from './diagnosis.controller';
import { verifyAdminOrDoctor } from '@common/middlewares/author.middleware';

const router = Router();

router.get('/', getAllDiagnoses);
router.get('/by-patient/:patientId', getDiagnosesByPatient);
router.get('/:id', getDiagnosisById);
router.patch('/:id/verify', verifyAdminOrDoctor, verifyDiagnosis);

export default router;
