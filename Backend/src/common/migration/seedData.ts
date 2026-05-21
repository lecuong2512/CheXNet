import { logger } from '@common/utils/logger';
import UserModel from '@modules/users/user.model';
import PatientModel from '@modules/patients/patient.model';
import ScanModel from '@modules/scans/scan.model';
import DiagnosisModel from '@modules/diagnoses/diagnosis.model';
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

/**
 * Thực hiện dọn dẹp sạch sẽ toàn bộ 15 bệnh án mẫu đã gieo hạt trước đó
 */
const cleanupSeededClinicalData = async (): Promise<void> => {
    const seededPatientCodes = [
        'PT-MANH45', 'PT-BINH32', 'PT-NAM60', 'PT-THU28', 'PT-DUY55',
        'PT-HOAINAM67', 'PT-LAN41', 'PT-BAO50', 'PT-TRANG36', 'PT-THANH72',
        'PT-THU58', 'PT-HUY19', 'PT-MAIANH25', 'PT-SAM63', 'PT-CHI48'
    ];

    try {
        // Tìm các bệnh nhân thuộc dữ liệu mẫu
        const seededPatients = await PatientModel.find({ patientCode: { $in: seededPatientCodes } });
        const patientIds = seededPatients.map((p) => p._id);

        if (patientIds.length > 0) {
            logger.info(`[Cleanup] Phát hiện ${patientIds.length} bệnh án mẫu. Bắt đầu dọn dẹp surgically...`);
            
            // Xóa các chẩn đoán liên quan
            const diagDelete = await DiagnosisModel.deleteMany({ patientId: { $in: patientIds } });
            
            // Xóa các phim chụp liên quan
            const scanDelete = await ScanModel.deleteMany({ patientId: { $in: patientIds } });
            
            // Xóa các hồ sơ bệnh nhân mẫu
            const patientDelete = await PatientModel.deleteMany({ _id: { $in: patientIds } });

            logger.info(
                `[Cleanup] ✅ Đã xóa sạch sẽ dữ liệu mẫu khỏi DB: ` +
                `${patientDelete.deletedCount} Bệnh nhân, ` +
                `${scanDelete.deletedCount} Phim chụp, ` +
                `${diagDelete.deletedCount} Chẩn đoán AI.`
            );
        } else {
            logger.info('[Cleanup] CSDL lâm sàng đã sạch sẽ. Không phát hiện dữ liệu mẫu.');
        }
    } catch (error) {
        logger.error('[Cleanup] ❌ Gặp lỗi khi dọn dẹp dữ liệu mẫu:', error);
    }
};

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

    // Dọn dẹp dữ liệu mẫu theo yêu cầu của bác sĩ
    await cleanupSeededClinicalData();
};
