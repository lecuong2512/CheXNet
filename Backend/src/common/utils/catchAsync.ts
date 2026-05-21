import { Request, Response, NextFunction } from 'express';

type AsyncHandler = (req: Request, res: Response, next: NextFunction) => Promise<void>;

/**
 * Bọc async handler, tự động forward lỗi vào middleware error handler.
 * Sử dụng: export const myHandler = catchAsync(async (req, res) => { ... })
 */
export const catchAsync = (fn: AsyncHandler) => {
    return (req: Request, res: Response, next: NextFunction): void => {
        Promise.resolve(fn(req, res, next)).catch((err) => next(err));
    };
};
