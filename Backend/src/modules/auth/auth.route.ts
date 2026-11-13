import { Router } from 'express';
import { login, register, logout, refreshToken } from './auth.controller';
import { validate } from '@common/middlewares/validate.middleware';
import { authenticationMiddleware } from '@common/middlewares/authen.middleware';
import { loginSchema, registerSchema } from './auth.validation';

const router = Router();

router.post('/login', validate(loginSchema), login);
router.post('/register', validate(registerSchema), register);
router.post('/refresh-token', refreshToken);
router.post('/logout', authenticationMiddleware, logout);

export default router;

