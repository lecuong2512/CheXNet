import Redis from 'ioredis';
import { REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_PREFIX } from './environment';
import { logger } from '@common/utils/logger';

let redisClient: Redis | null = null;

export const connectRedis = async (): Promise<void> => {
    try {
        redisClient = new Redis({
            host: REDIS_HOST,
            port: REDIS_PORT,
            password: REDIS_PASSWORD || undefined,
            keyPrefix: REDIS_PREFIX,
            lazyConnect: true,
            enableOfflineQueue: false,
        });

        await redisClient.connect();
        logger.info(`✅ Redis kết nối thành công: ${REDIS_HOST}:${REDIS_PORT}`);

        redisClient.on('error', (err) => {
            logger.error('❌ Lỗi Redis:', err);
        });

        redisClient.on('close', () => {
            logger.warn('⚠️  Redis bị ngắt kết nối');
        });
    } catch (error) {
        logger.warn('⚠️  Redis không khả dụng, tiếp tục mà không có cache');
        redisClient = null;
    }
};

/**
 * RedisAdapter - Wrapper an toàn, tự xử lý khi Redis không khả dụng
 */
export const RedisAdapter = {
    get: async (key: string): Promise<string | null> => {
        if (!redisClient) return null;
        return redisClient.get(key);
    },
    set: async (key: string, value: string, ttlSeconds?: number): Promise<void> => {
        if (!redisClient) return;
        if (ttlSeconds) {
            await redisClient.setex(key, ttlSeconds, value);
        } else {
            await redisClient.set(key, value);
        }
    },
    setnx: async (key: string, value: string): Promise<number> => {
        if (!redisClient) return 0;
        return redisClient.setnx(key, value);
    },
    del: async (key: string): Promise<void> => {
        if (!redisClient) return;
        await redisClient.del(key);
    },
    sadd: async (key: string, ...members: string[]): Promise<void> => {
        if (!redisClient) return;
        await redisClient.sadd(key, ...members);
    },
    smembers: async (key: string): Promise<string[]> => {
        if (!redisClient) return [];
        return redisClient.smembers(key);
    },
    sismember: async (key: string, member: string): Promise<boolean> => {
        if (!redisClient) return false;
        const result = await redisClient.sismember(key, member);
        return result === 1;
    },
    expire: async (key: string, ttlSeconds: number): Promise<void> => {
        if (!redisClient) return;
        await redisClient.expire(key, ttlSeconds);
    },
    isConnected: (): boolean => redisClient !== null,
};
