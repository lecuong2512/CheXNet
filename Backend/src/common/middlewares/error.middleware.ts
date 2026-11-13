import { Request, Response, NextFunction } from 'express';
import { ApiError } from '@common/utils/ApiError';
import { logger } from '@common/utils/logger';
import { MulterError } from 'multer';

/**
 * Global error handler - bắt cuối cùng trong middleware chain.
 * ApiError → trả status code + message, lỗi khác → 500
 */
export const errorHandler = (err: Error | ApiError, req: Request, res: Response, _next: NextFunction): void => {
    let statusCode = 500;
    let message = 'Lỗi máy chủ nội bộ';

    if (err instanceof MulterError) {
        statusCode = err.code === 'LIMIT_FILE_SIZE' ? 413 : 400;
        message = err.code === 'LIMIT_FILE_SIZE' ? 'Ảnh vượt quá dung lượng cho phép' : 'Dữ liệu upload không hợp lệ';
    } else if (err instanceof ApiError) {
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
