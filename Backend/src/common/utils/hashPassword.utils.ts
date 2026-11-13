import bcrypt from 'bcrypt';

const SALT_ROUNDS = 12;

/**
 * Hash mật khẩu bằng bcrypt
 */
export const hashPassword = async (plainPassword: string): Promise<string> => {
    return bcrypt.hash(plainPassword, SALT_ROUNDS);
};

/**
 * So sánh mật khẩu plaintext với hash
 */
export const comparePassword = async (plainPassword: string, hash: string): Promise<boolean> => {
    return bcrypt.compare(plainPassword, hash);
};
