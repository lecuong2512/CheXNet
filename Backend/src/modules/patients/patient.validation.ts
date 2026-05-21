import Joi from 'joi';
import { Gender, RiskLevel } from '@common/utils/enum';

export const createPatientSchema = Joi.object({
    patientCode: Joi.string().uppercase().optional(),
    name: Joi.string().min(2).max(150).required().messages({
        'string.empty': 'Tên bệnh nhân không được để trống',
        'any.required': 'Tên bệnh nhân là bắt buộc',
    }),
    gender: Joi.string()
        .valid(...Object.values(Gender))
        .required()
        .messages({
            'any.only': `Giới tính phải là một trong: ${Object.values(Gender).join(', ')}`,
            'any.required': 'Giới tính là bắt buộc',
        }),
    age: Joi.number().integer().min(0).max(150).required().messages({
        'number.base': 'Tuổi phải là số',
        'any.required': 'Tuổi là bắt buộc',
    }),
    bloodType: Joi.string().optional(),
    department: Joi.string().optional(),
    phone: Joi.string().optional(),
    address: Joi.string().optional(),
    vitals: Joi.object({
        heartRate: Joi.number().optional(),
        bloodPressure: Joi.string().optional(),
        spo2: Joi.number().min(0).max(100).optional(),
        temperature: Joi.number().optional(),
        lungIndex: Joi.number().optional(),
    }).optional(),
    riskLevel: Joi.string()
        .valid(...Object.values(RiskLevel))
        .optional(),
});

export const updatePatientSchema = Joi.object({
    name: Joi.string().min(2).max(150).optional(),
    gender: Joi.string()
        .valid(...Object.values(Gender))
        .optional(),
    age: Joi.number().integer().min(0).max(150).optional(),
    bloodType: Joi.string().optional(),
    department: Joi.string().optional(),
    phone: Joi.string().optional(),
    address: Joi.string().optional(),
    vitals: Joi.object({
        heartRate: Joi.number().optional(),
        bloodPressure: Joi.string().optional(),
        spo2: Joi.number().min(0).max(100).optional(),
        temperature: Joi.number().optional(),
        lungIndex: Joi.number().optional(),
    }).optional(),
    riskLevel: Joi.string()
        .valid(...Object.values(RiskLevel))
        .optional(),
});
