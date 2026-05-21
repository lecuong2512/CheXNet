import jwt from 'jsonwebtoken';
import { JWT_PRIVATE_KEY, JWT_EXPIRES_IN, JWT_REFRESH_EXPIRES_IN } from '@config/environment';
import { ApiError } from './ApiError';

export interface ITokenPayload {
    userId: string;
    role: string;
    tokenVersion: number;
}

/**
 * Sinh access token (15m) và refresh token (7d)
 */
export const generateTokens = (payload: ITokenPayload): { accessToken: string; refreshToken: string } => {
    const accessToken = jwt.sign(payload, JWT_PRIVATE_KEY, { expiresIn: JWT_EXPIRES_IN } as jwt.SignOptions);
    const refreshToken = jwt.sign(payload, JWT_PRIVATE_KEY, { expiresIn: JWT_REFRESH_EXPIRES_IN } as jwt.SignOptions);
    return { accessToken, refreshToken };
};

/**
 * Xác thực và giải mã token
 */
export const verifyToken = (token: string): ITokenPayload => {
    try {
        return jwt.verify(token, JWT_PRIVATE_KEY) as ITokenPayload;
    } catch (err) {
        throw new ApiError(401, 'Token không hợp lệ hoặc đã hết hạn');
    }
};
