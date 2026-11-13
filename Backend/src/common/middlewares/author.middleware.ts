import { Request, Response, NextFunction } from 'express';
import { ApiError } from '@common/utils/ApiError';
import { UserRole } from '@common/utils/enum';

/**
 * Middleware kiểm tra phân quyền theo role.
 * Sử dụng: authorizeRoles(UserRole.ADMIN, UserRole.DOCTOR)
 */
export const authorizeRoles = (...roles: UserRole[]) => {
    return (req: Request, _res: Response, next: NextFunction): void => {
        if (!req.users || !roles.includes(req.users.role)) {
            throw new ApiError(403, 'Bạn không có quyền truy cập tài nguyên này');
        }
        next();
    };
};

/**
 * Middleware chỉ cho phép Admin truy cập
 */
export const verifyAdmin = (req: Request, res: Response, next: NextFunction): void => {
    authorizeRoles(UserRole.ADMIN)(req, res, next);
};

/**
 * Middleware cho phép Admin hoặc Doctor truy cập
 */
export const verifyAdminOrDoctor = (req: Request, res: Response, next: NextFunction): void => {
    authorizeRoles(UserRole.ADMIN, UserRole.DOCTOR)(req, res, next);
};
