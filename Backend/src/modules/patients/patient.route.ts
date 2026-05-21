import { Router } from 'express';
import { getPatients, getPatientById, createPatient, updatePatient } from './patient.controller';
import { validate } from '@common/middlewares/validate.middleware';
import { verifyAdminOrDoctor } from '@common/middlewares/author.middleware';
import { createPatientSchema, updatePatientSchema } from './patient.validation';

const router = Router();

router.get('/', getPatients);
router.get('/:id', getPatientById);
router.post('/', verifyAdminOrDoctor, validate(createPatientSchema), createPatient);
router.patch('/:id', verifyAdminOrDoctor, validate(updatePatientSchema), updatePatient);

export default router;
