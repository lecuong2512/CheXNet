import crypto from 'crypto';
import { CRYPTO_SECRET } from '@config/environment';
import { logger } from './logger';

/**
 * Mã hóa AES-256-CBC - tương thích với Frontend crypto-js
 * Format: iv:encryptedContent (hex)
 */
export const encrypt = (plainText: string): string => {
    const iv = crypto.randomBytes(16);
    const key = crypto.createHash('sha256').update(CRYPTO_SECRET).digest();
    const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
    let encrypted = cipher.update(Buffer.from(plainText));
    encrypted = Buffer.concat([encrypted, cipher.final()]);
    return `${iv.toString('hex')}:${encrypted.toString('hex')}`;
};

/**
 * Giải mã AES-256-CBC
 * Input format: iv:encryptedContent (hex)
 */
export const decrypt = (encryptedData: string): string | null => {
    try {
        const parts = encryptedData.split(':');
        if (parts.length !== 2) return null;
        const iv = Buffer.from(parts[0], 'hex');
        const encryptedText = Buffer.from(parts[1], 'hex');
        const key = crypto.createHash('sha256').update(CRYPTO_SECRET).digest();
        const decipher = crypto.createDecipheriv('aes-256-cbc', Buffer.from(key), iv);
        let decrypted = decipher.update(encryptedText);
        decrypted = Buffer.concat([decrypted, decipher.final()]);
        return decrypted.toString();
    } catch (error) {
        logger.error('Decryption failed:', error);
        return null;
    }
};
