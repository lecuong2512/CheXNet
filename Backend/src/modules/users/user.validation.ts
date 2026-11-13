import Joi from 'joi';
import { UserRole } from '@common/utils/enum';

export const createUserSchema = Joi.object({
    name: Joi.string().min(2).max(100).required().messages({
        'string.min': 'Tên phải có ít nhất 2 ký tự',
        'string.empty': 'Tên không được để trống',
        'any.required': 'Tên là bắt buộc',
    }),
    email: Joi.string().email().required().messages({
        'string.email': 'Email không hợp lệ',
        'any.required': 'Email là bắt buộc',
    }),
    password: Joi.string().min(6).required().messages({
        'string.min': 'Mật khẩu phải có ít nhất 6 ký tự',
        'any.required': 'Mật khẩu là bắt buộc',
    }),
    role: Joi.string()
        .valid(...Object.values(UserRole))
        .optional(),
    department: Joi.string().optional(),
});

export const updateUserSchema = Joi.object({
    name: Joi.string().min(2).max(100).optional(),
    role: Joi.string()
        .valid(...Object.values(UserRole))
        .optional(),
    department: Joi.string().optional(),
    isActive: Joi.boolean().optional(),
    password: Joi.string().min(6).optional(),
});
