import { Document, Types } from 'mongoose';
import { UserRole } from '@common/utils/enum';

export interface IUser extends Document {
    _id: Types.ObjectId;
    name: string;
    email: string;
    passwordHash: string;
    role: UserRole;
    department?: string;
    avatar?: string;
    tokenVersion: number;
    isActive: boolean;
    createdAt: Date;
    updatedAt: Date;
}

export interface IUserPayload {
    name?: string;
    email?: string;
    password?: string;
    role?: UserRole;
    department?: string;
    avatar?: string;
    isActive?: boolean;
}

export interface IUserFilter {
    name?: string;
    email?: string;
    role?: UserRole;
    department?: string;
    isActive?: boolean;
    page?: number;
    limit?: number;
    keyword?: string;
}

// Gắn thêm vào Express Request
export interface IRequestUser {
    userId: string;
    role: UserRole;
    tokenVersion: number;
    filter?: Record<string, unknown>;
    scope?: string;
}

// Extend Express Request
declare global {
    namespace Express {
        interface Request {
            users: IRequestUser;
        }
    }
}
