import Joi from 'joi';

export const loginSchema = Joi.object({
    email: Joi.string().email().required().messages({
        'string.email': 'Định dạng email không hợp lệ',
        'string.empty': 'Email không được để trống',
        'any.required': 'Email là bắt buộc',
    }),
    password: Joi.string().min(6).required().messages({
        'string.min': 'Mật khẩu phải có ít nhất 6 ký tự',
        'string.empty': 'Mật khẩu không được để trống',
        'any.required': 'Mật khẩu là bắt buộc',
    }),
});

export const registerSchema = Joi.object({
    name: Joi.string().min(2).max(100).required().messages({
        'string.min': 'Họ tên phải có ít nhất 2 ký tự',
        'string.max': 'Họ tên không được vượt quá 100 ký tự',
        'string.empty': 'Họ tên không được để trống',
        'any.required': 'Họ tên là bắt buộc',
    }),
    email: Joi.string().email().required().messages({
        'string.email': 'Định dạng email không hợp lệ',
        'string.empty': 'Email không được để trống',
        'any.required': 'Email là bắt buộc',
    }),
    password: Joi.string().min(6).required().messages({
        'string.min': 'Mật khẩu phải có ít nhất 6 ký tự',
        'string.empty': 'Mật khẩu không được để trống',
        'any.required': 'Mật khẩu là bắt buộc',
    }),
    department: Joi.string().max(100).optional().allow('').messages({
        'string.max': 'Tên khoa không được vượt quá 100 ký tự',
    }),
});

