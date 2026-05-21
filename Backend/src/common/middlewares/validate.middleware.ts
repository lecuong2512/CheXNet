import { Request, Response, NextFunction } from 'express';
import Joi from 'joi';
import { ApiError } from '@common/utils/ApiError';

/**
 * Joi validation middleware.
 * Lưu ý: Luôn bảo toàn userId, patientId trong req.body để middleware phân quyền phía sau sử dụng.
 */
export const validate = (schema: Joi.ObjectSchema) => {
    return (req: Request, _res: Response, next: NextFunction): void => {
        const { userId, patientId } = req.body;

        const { error, value } = schema.validate(
            {
                ...req.body,
                ...(userId ? { userId: String(userId) } : {}),
                ...(patientId ? { patientId: String(patientId) } : {}),
            },
            {
                abortEarly: false,
                stripUnknown: true,
            },
        );

        if (error) {
            const errorMessage = error.details.map((d) => d.message).join(', ');
            return next(new ApiError(400, errorMessage));
        }

        req.body = {
            ...value,
            ...(userId ? { userId } : {}),
            ...(patientId ? { patientId } : {}),
        };

        next();
    };
};
