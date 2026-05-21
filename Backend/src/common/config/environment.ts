import 'dotenv/config';

export const PORT: number = parseInt(process.env.PORT || '3000', 10);
export const NODE_ENV: string = process.env.NODE_ENV || 'development';
/** MongoDB standalone (Windows dev) không hỗ trợ retryable writes — bắt buộc tắt */
const defaultMongoUri = 'mongodb://localhost:27017/chexnet_v3?retryWrites=false';
const rawMongoUri = process.env.MONGODB_URI || defaultMongoUri;

export const MONGODB_URI: string = rawMongoUri.includes('retryWrites=')
    ? rawMongoUri
    : `${rawMongoUri}${rawMongoUri.includes('?') ? '&' : '?'}retryWrites=false`;
export const REDIS_HOST: string = process.env.REDIS_HOST || 'localhost';
export const REDIS_PORT: number = parseInt(process.env.REDIS_PORT || '6379', 10);
export const REDIS_PASSWORD: string = process.env.REDIS_PASSWORD || '';
export const REDIS_PREFIX: string = process.env.REDIS_PREFIX || 'chexnet::';
export const JWT_PRIVATE_KEY: string = process.env.JWT_PRIVATE_KEY || 'fallback-secret-key';
export const JWT_EXPIRES_IN: string = process.env.JWT_EXPIRES_IN || '15m';
export const JWT_REFRESH_EXPIRES_IN: string = process.env.JWT_REFRESH_EXPIRES_IN || '7d';
export const CRYPTO_SECRET: string = process.env.CRYPTO_SECRET || 'fallback-crypto-secret';
export const AI_SERVICE_URL: string = process.env.AI_SERVICE_URL || 'http://localhost:8000';
export const UPLOAD_DIR: string = process.env.UPLOAD_DIR || './uploads';
export const MAX_FILE_SIZE_MB: number = parseInt(process.env.MAX_FILE_SIZE_MB || '10', 10);
export const ALLOWED_ORIGINS: string[] = (process.env.ALLOWED_ORIGINS || 'http://localhost:5173').split(',');
export const LOG_LEVEL: string = process.env.LOG_LEVEL || 'debug';
