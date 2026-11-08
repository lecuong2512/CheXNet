import { Router } from 'express';
import { getUsers, createUser, updateUser, deleteUser } from './user.controller';
import { validate } from '@common/middlewares/validate.middleware';
import { verifyAdmin } from '@common/middlewares/author.middleware';
import { createUserSchema, updateUserSchema } from './user.validation';

const router = Router();

// Tất cả users routes yêu cầu quyền Admin
router.get('/', verifyAdmin, getUsers);
router.post('/', verifyAdmin, validate(createUserSchema), createUser);
router.patch('/:id', verifyAdmin, validate(updateUserSchema), updateUser);
router.delete('/:id', verifyAdmin, deleteUser);

export default router;
