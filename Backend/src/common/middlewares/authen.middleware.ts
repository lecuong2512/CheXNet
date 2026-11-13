import { Request, Response, NextFunction } from 'express';
import { verifyToken } from '@common/utils/token.utils';
import { ApiError } from '@common/utils/ApiError';
import UserModel from '@modules/users/user.model';

/**
 * Xác thực JWT token và kiểm tra tokenVersion (chống dùng token sau khi đổi mật khẩu/logout).
 * Gắn req.users với thông tin user đã xác thực.
 */
export const authenticationMiddleware = async (req: Request, _res: Response, next: NextFunction): Promise<void> => {
    try {
        const authHeader = req.headers.authorization;
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            throw new ApiError(401, 'Không có token xác thực');
        }

        const token = authHeader.split(' ')[1];
        const decoded = verifyToken(token);

        // Kiểm tra tokenVersion trong DB để phát hiện logout/đổi mật khẩu
        const user = await UserModel.findById(decoded.userId).select('tokenVersion role isActive');
        if (!user) {
            throw new ApiError(401, 'Tài khoản không tồn tại');
        }
        if (!user.isActive) {
            throw new ApiError(401, 'Tài khoản đã bị vô hiệu hóa');
        }
        if (user.tokenVersion !== decoded.tokenVersion) {
            throw new ApiError(401, 'Token đã hết hiệu lực, vui lòng đăng nhập lại');
        }

        req.users = {
            userId: decoded.userId,
            role: decoded.role as any,
            tokenVersion: decoded.tokenVersion,
        };

        next();
    } catch (err) {
        next(err);
    }
};
