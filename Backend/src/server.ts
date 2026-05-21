import http from 'http';
import app from './app';
import { PORT } from '@config/environment';
import { connectDatabase } from '@config/database';
import { connectRedis } from '@config/redis';
import { socketService } from '@socket/socket.service';
import { initCronJobs } from '@worker/cron/cron.manager';
import { runSeedData } from '@common/migration/seedData';
import { logger } from '@common/utils/logger';

const bootstrap = async (): Promise<void> => {
    try {
        // 1. Kết nối Database (MongoDB)
        await connectDatabase();

        // 2. Kết nối Caching (Redis)
        await connectRedis();

        // 3. Khởi tạo HTTP Server
        const server = http.createServer(app);

        // 4. Khởi tạo Socket.IO
        socketService.initialize(server);

        // 5. Khởi chạy Cron Jobs tác vụ nền
        initCronJobs();

        // 6. Khởi tạo tài khoản hệ thống (nếu chưa có)
        await runSeedData();

        // 7. Lắng nghe cổng kết nối
        server.listen(PORT, () => {
            logger.info(`🚀 Máy chủ CheXNet V3 đang chạy tại cổng http://localhost:${PORT}`);
        });

        // Xử lý đóng kết nối an toàn khi nhận tín hiệu tắt máy (Graceful Shutdown)
        const gracefulShutdown = (signal: string) => {
            logger.warn(`⚠️ Nhận tín hiệu ${signal}. Đang đóng máy chủ gracefully...`);
            server.close(() => {
                logger.info('👋 Đã đóng máy chủ HTTP.');
                process.exit(0);
            });
        };

        process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
        process.on('SIGINT', () => gracefulShutdown('SIGINT'));

    } catch (error) {
        logger.error('❌ Lỗi nghiêm trọng khi khởi động hệ thống:', error);
        process.exit(1);
    }
};

bootstrap();
