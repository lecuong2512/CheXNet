import { Request, Response, NextFunction } from 'express';

/**
 * Tự động trim tất cả string trong req.body trước khi xử lý
 */
const trimValue = (value: unknown): unknown => {
    if (typeof value === 'string') return value.trim();
    if (Array.isArray(value)) return value.map(trimValue);
    if (typeof value === 'object' && value !== null) {
        return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, trimValue(v)]));
    }
    return value;
};

export const trimRequest = (req: Request, _res: Response, next: NextFunction): void => {
    if (req.body && typeof req.body === 'object') {
        req.body = trimValue(req.body);
    }
    next();
};
