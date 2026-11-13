import cron from 'node-cron';
import { logger } from '@common/utils/logger';
import path from 'path';
import fs from 'fs';
import { UPLOAD_DIR } from '@config/environment';
import ScanModel from '@modules/scans/scan.model';
import { ScanStatus } from '@common/utils/enum';

export const initCronJobs = (): void => {
    // Dọn file upload thất bại sau 24h (chạy mỗi đêm lúc 2:00 AM)
    cron.schedule('0 2 * * *', async () => {
        logger.info('[Cron] Đang dọn dẹp file scan thất bại...');
        try {
            const oneDayAgo = new Date(Date.now() - 24 * 3600 * 1000);
            const failedScans = await ScanModel.find({
                status: ScanStatus.FAILED,
                createdAt: { $lt: oneDayAgo },
            });

            let cleaned = 0;
            for (const scan of failedScans) {
                const fullPath = path.resolve(scan.imagePath);
                if (fs.existsSync(fullPath)) {
                    fs.unlinkSync(fullPath);
                    cleaned++;
                }
                await ScanModel.findByIdAndDelete(scan._id);
            }

            logger.info(`[Cron] Đã dọn ${cleaned} file scan thất bại`);
        } catch (err) {
            logger.error('[Cron] Lỗi dọn file:', err);
        }
    });

    // Reset scans bị kẹt ở trạng thái PROCESSING > 5 phút (chạy mỗi 5 phút)
    cron.schedule('*/5 * * * *', async () => {
        try {
            const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000);
            const stuckScans = await ScanModel.updateMany(
                {
                    status: ScanStatus.PROCESSING,
                    processingStartedAt: { $lt: fiveMinAgo },
                },
                { status: ScanStatus.FAILED },
            );

            if (stuckScans.modifiedCount > 0) {
                logger.warn(`[Cron] Đặt lại ${stuckScans.modifiedCount} scan bị kẹt → FAILED`);
            }
        } catch (err) {
            logger.error('[Cron] Lỗi reset stuck scans:', err);
        }
    });

    logger.info('✅ Cron jobs đã khởi động');
};
