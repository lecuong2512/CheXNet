import { Request, Response, NextFunction } from 'express';
import { ApiError } from '@common/utils/ApiError';
import { logger } from '@common/utils/logger';

/**
 * Global error handler - bắt cuối cùng trong middleware chain.
 * ApiError → trả status code + message, lỗi khác → 500
 */
export const errorHandler = (err: Error | ApiError, req: Request, res: Response, _next: NextFunction): void => {
    let statusCode = 500;
    let message = 'Lỗi máy chủ nội bộ';

    if (err instanceof ApiError) {
        statusCode = err.statusCode;
        message = err.message;
    } else {
        logger.error(`Unrecognized Error [${req.method} ${req.originalUrl}]:`, err);
    }

    res.status(statusCode).json({
        success: false,
        status: statusCode,
        message,
    });
};
