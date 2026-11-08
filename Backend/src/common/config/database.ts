import mongoose from 'mongoose';
import { MONGODB_URI } from './environment';
import { logger } from '@common/utils/logger';

export const connectDatabase = async (): Promise<void> => {
    try {
        await mongoose.connect(MONGODB_URI, {
            serverSelectionTimeoutMS: 5000,
            retryWrites: false,
            directConnection: true,
        });
        logger.info(`✅ MongoDB kết nối thành công: ${MONGODB_URI}`);

        mongoose.connection.on('disconnected', () => {
            logger.warn('⚠️  MongoDB bị ngắt kết nối');
        });

        mongoose.connection.on('error', (err) => {
            logger.error('❌ Lỗi MongoDB:', err);
        });
    } catch (error) {
        logger.error('❌ Không thể kết nối MongoDB:', error);
        process.exit(1);
    }
};
