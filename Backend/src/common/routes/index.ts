import { Router } from 'express';
import { authenticationMiddleware } from '@common/middlewares/authen.middleware';
import AuthRoute from '@modules/auth/auth.route';
import UserRoute from '@modules/users/user.route';
import PatientRoute from '@modules/patients/patient.route';
import ScanRoute from '@modules/scans/scan.route';
import DiagnosisRoute from '@modules/diagnoses/diagnosis.route';
import ResearchRoute from '@modules/research/research.route';

const router = Router();

// Public routes (không cần token)
router.use('/auth', AuthRoute);

// Protected routes (tất cả đều qua authenticationMiddleware)
router.use('/users', authenticationMiddleware, UserRoute);
router.use('/patients', authenticationMiddleware, PatientRoute);
router.use('/scans', authenticationMiddleware, ScanRoute);
router.use('/diagnoses', authenticationMiddleware, DiagnosisRoute);
router.use('/research', authenticationMiddleware, ResearchRoute);

export default router;
