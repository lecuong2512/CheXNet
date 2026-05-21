import { logger } from '@common/utils/logger';
import UserModel from '@modules/users/user.model';
import { hashPassword } from '@common/utils/hashPassword.utils';
import { UserRole } from '@common/utils/enum';

/** Tài khoản hệ thống — luôn được phép đăng nhập với đúng vai trò */
const SYSTEM_ACCOUNTS = [
    {
        email: 'admin@chexnet.vn',
        name: 'Quản trị viên CheXNet',
        password: 'Admin@123456',
        role: UserRole.ADMIN,
        department: 'Ban Quản trị',
    },
    {
        email: 'bacsi@chexnet.vn',
        name: 'BS. Nguyễn Văn Khoa',
        password: 'Doctor@123456',
        role: UserRole.DOCTOR,
        department: 'Nội Hô Hấp',
    },
] as const;

export const RESERVED_SYSTEM_EMAILS = SYSTEM_ACCOUNTS.map((a) => a.email);

export const runSeedData = async (): Promise<void> => {
    logger.info('[Seed] Kiểm tra và khởi tạo tài khoản hệ thống...');

    for (const account of SYSTEM_ACCOUNTS) {
        const existing = await UserModel.findOne({ email: account.email });

        if (!existing) {
            await UserModel.create({
                name: account.name,
                email: account.email,
                passwordHash: await hashPassword(account.password),
                role: account.role,
                department: account.department,
                tokenVersion: 0,
                isActive: true,
            });
            logger.info(
                `[Seed] ✅ Tạo ${account.role}: ${account.email} / ${account.password}`,
            );
            continue;
        }

        const roleWasWrong = existing.role !== account.role;
        const update: Record<string, unknown> = {
            role: account.role,
            isActive: true,
            name: account.name,
            department: account.department,
        };
        if (roleWasWrong) {
            update.passwordHash = await hashPassword(account.password);
        }

        await UserModel.updateOne({ email: account.email }, { $set: update });
        logger.info(
            roleWasWrong
                ? `[Seed] ✅ Khôi phục ${account.role} + mật khẩu mặc định: ${account.email}`
                : `[Seed] ✅ Đảm bảo quyền ${account.role}: ${account.email}`,
        );
    }

    logger.info('[Seed] ✅ Hoàn thành khởi tạo tài khoản hệ thống');
};
