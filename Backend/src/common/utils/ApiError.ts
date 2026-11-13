/**
 * Lớp lỗi tùy chỉnh mang theo HTTP status code.
 * Sử dụng: throw new ApiError(400, 'Thông báo lỗi tiếng Việt')
 */
export class ApiError extends Error {
    public statusCode: number;

    constructor(statusCode: number, message: string) {
        super(message);
        this.statusCode = statusCode;
        this.name = 'ApiError';
        Object.setPrototypeOf(this, ApiError.prototype);
    }
}
