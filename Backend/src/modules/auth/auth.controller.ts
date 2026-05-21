import { Request, Response } from 'express';
import { catchAsync } from '@common/utils/catchAsync';
import { verifyToken } from '@common/utils/token.utils';
import { ApiError } from '@common/utils/ApiError';
import { AuthService } from './auth.service';

const authService = new AuthService();

export const login = catchAsync(async (req: Request, res: Response) => {
    const { email, password } = req.body;
    const result = await authService.login(email, password);
    res.status(200).json(result);
});

export const register = catchAsync(async (req: Request, res: Response) => {
    const { name, email, password, department } = req.body;
    const result = await authService.register({ name, email, password, department });
    res.status(201).json(result);
});

export const logout = catchAsync(async (req: Request, res: Response) => {
    const result = await authService.logout(req.users.userId);
    res.status(200).json(result);
});

export const refreshToken = catchAsync(async (req: Request, res: Response) => {
    const { refreshToken: token } = req.body;
    if (!token) throw new ApiError(400, 'Refresh token không được để trống');
    const decoded = verifyToken(token);
    const result = await authService.refreshToken(decoded.userId, decoded.tokenVersion);
    res.status(200).json(result);
});

