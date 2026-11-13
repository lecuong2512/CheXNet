import { IApiResponse } from '@common/interfaces/response.interface';
import { ApiError } from '@common/utils/ApiError';
import { comparePassword, hashPassword } from '@common/utils/hashPassword.utils';
import { generateTokens } from '@common/utils/token.utils';
import UserModel from '@modules/users/user.model';
import { IUser } from '@common/interfaces/user.interface';
import { UserRole } from '@common/utils/enum';
import { RESERVED_SYSTEM_EMAILS } from '@common/migration/seedData';

export class AuthService {
    /**
     * Đăng nhập với email/password, trả về JWT access + refresh token
     */
    public async login(
        email: string,
        password: string,
    ): Promise<IApiResponse<{ accessToken: string; refreshToken: string; user: Partial<IUser> }>> {
        const normalizedEmail = email.trim().toLowerCase();
        const user = await UserModel.findOne({ email: normalizedEmail, isActive: true });
        if (!user) {
            throw new ApiError(401, 'Email hoặc mật khẩu không đúng');
        }

        const isMatch = await comparePassword(password, user.passwordHash);
        if (!isMatch) {
            throw new ApiError(401, 'Email hoặc mật khẩu không đúng');
        }

        const { accessToken, refreshToken } = generateTokens({
            userId: user._id.toString(),
            role: user.role,
            tokenVersion: user.tokenVersion,
        });

        return {
            success: true,
            message: 'Đăng nhập thành công',
            data: {
                accessToken,
                refreshToken,
                user: {
                    _id: user._id,
                    name: user.name,
                    email: user.email,
                    role: user.role,
                    department: user.department,
                    avatar: user.avatar,
                },
            },
        };
    }

    /**
     * Đăng ký tài khoản Bác sĩ mới
     */
    public async register(payload: {
        name: string;
        email: string;
        password: string;
        department?: string;
    }): Promise<IApiResponse<{ accessToken: string; refreshToken: string; user: Partial<IUser> }>> {
        const normalizedEmail = payload.email.trim().toLowerCase();

        if ((RESERVED_SYSTEM_EMAILS as readonly string[]).includes(normalizedEmail)) {
            throw new ApiError(
                403,
                'Email hệ thống (admin@chexnet.vn) không thể đăng ký mới. Vui lòng dùng trang Đăng nhập.',
            );
        }

        const existing = await UserModel.findOne({ email: normalizedEmail });
        if (existing) {
            throw new ApiError(400, 'Email đã được sử dụng bởi tài khoản khác');
        }

        const passwordHash = await hashPassword(payload.password);
        const user = await UserModel.create({
            name: payload.name,
            email: normalizedEmail,
            passwordHash,
            role: UserRole.DOCTOR,
            department: payload.department || '',
            tokenVersion: 0,
            isActive: true,
        });

        const { accessToken, refreshToken } = generateTokens({
            userId: user._id.toString(),
            role: user.role,
            tokenVersion: user.tokenVersion,
        });

        return {
            success: true,
            message: 'Đăng ký tài khoản thành công',
            data: {
                accessToken,
                refreshToken,
                user: {
                    _id: user._id,
                    name: user.name,
                    email: user.email,
                    role: user.role,
                    department: user.department,
                    avatar: user.avatar,
                },
            },
        };
    }

    /**
     * Đăng xuất: tăng tokenVersion để vô hiệu hóa token cũ
     */
    public async logout(userId: string): Promise<IApiResponse<null>> {
        await UserModel.findByIdAndUpdate(userId, { $inc: { tokenVersion: 1 } });
        return { success: true, message: 'Đăng xuất thành công' };
    }

    /**
     * Refresh token: xác thực refresh token và sinh cặp token mới
     */
    public async refreshToken(
        userId: string,
        tokenVersion: number,
    ): Promise<IApiResponse<{ accessToken: string; refreshToken: string }>> {
        const user = await UserModel.findById(userId).select('tokenVersion role isActive');
        if (!user || !user.isActive) {
            throw new ApiError(401, 'Tài khoản không tồn tại hoặc đã bị vô hiệu hóa');
        }
        if (user.tokenVersion !== tokenVersion) {
            throw new ApiError(401, 'Refresh token không còn hợp lệ');
        }

        const { accessToken, refreshToken } = generateTokens({
            userId: user._id.toString(),
            role: user.role,
            tokenVersion: user.tokenVersion,
        });

        return {
            success: true,
            message: 'Làm mới token thành công',
            data: { accessToken, refreshToken },
        };
    }
}

