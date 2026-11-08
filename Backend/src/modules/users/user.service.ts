import { PipelineStage, QueryWithHelpers, Types } from 'mongoose';
import { IApiResponse, IPaginatedResponse } from '@common/interfaces/response.interface';
import { IUser, IUserFilter, IUserPayload } from '@common/interfaces/user.interface';
import { ApiError } from '@common/utils/ApiError';
import { hashPassword } from '@common/utils/hashPassword.utils';
import { UserRole } from '@common/utils/enum';
import UserModel from './user.model';

export class UserService {
    /**
     * Lấy danh sách người dùng với phân trang và tìm kiếm
     */
    public async getByFilter(filter: IUserFilter): Promise<IPaginatedResponse<IUser>> {
        const page = filter.page || 1;
        const limit = filter.limit || 10;
        const query: QueryWithHelpers<IUser[], IUser, object> | Record<string, unknown> = { isActive: true };

        if (filter.role) (query as Record<string, unknown>).role = filter.role;
        if (filter.department) (query as Record<string, unknown>).department = filter.department;
        if (filter.keyword) {
            (query as Record<string, unknown>).$or = [
                { name: { $regex: filter.keyword, $options: 'i' } },
                { email: { $regex: filter.keyword, $options: 'i' } },
            ];
        }

        const aggregateQuery: PipelineStage[] = [
            { $match: query },
            { $sort: { createdAt: -1 } as Record<string, 1 | -1> },
        ];

        const [records, counter] = await Promise.all([
            UserModel.aggregate([
                ...aggregateQuery,
                { $skip: (page - 1) * limit },
                { $limit: limit },
                { $project: { passwordHash: 0 } },
            ]),
            UserModel.aggregate([...aggregateQuery, { $count: 'count' }]),
        ]);

        const count = counter[0]?.count || 0;
        return {
            success: true,
            message: 'Lấy danh sách người dùng thành công',
            data: records,
            meta: {
                totalItems: count,
                itemCount: records.length,
                itemsPerPage: limit,
                totalPages: Math.ceil(count / limit),
                currentPage: page,
            },
        };
    }

    /**
     * Tạo tài khoản mới (Admin only)
     */
    public async create(payload: IUserPayload & { password: string }): Promise<IApiResponse<Partial<IUser>>> {
        const existing = await UserModel.findOne({ email: payload.email?.toLowerCase() });
        if (existing) {
            throw new ApiError(400, 'Email đã được sử dụng bởi tài khoản khác');
        }

        const passwordHash = await hashPassword(payload.password);
        const user = await UserModel.create({
            ...payload,
            email: payload.email?.toLowerCase(),
            passwordHash,
        });

        const { passwordHash: _, ...userWithoutPassword } = user.toObject();
        return {
            success: true,
            message: 'Tạo tài khoản thành công',
            data: userWithoutPassword,
        };
    }

    /**
     * Cập nhật thông tin người dùng
     */
    public async update(id: string, payload: IUserPayload): Promise<IApiResponse<Partial<IUser>>> {
        if (!Types.ObjectId.isValid(id)) {
            throw new ApiError(400, 'ID người dùng không hợp lệ');
        }

        const updateData: Record<string, unknown> = { ...payload };
        if ((payload as any).password) {
            updateData.passwordHash = await hashPassword((payload as any).password);
            updateData.tokenVersion = 1; // invalidate existing tokens
            delete updateData.password;
        }

        const user = await UserModel.findByIdAndUpdate(id, updateData, { new: true }).select('-passwordHash');
        if (!user) {
            throw new ApiError(404, 'Không tìm thấy tài khoản');
        }

        return { success: true, message: 'Cập nhật tài khoản thành công', data: user.toObject() };
    }

    /**
     * Vô hiệu hóa tài khoản (soft delete)
     */
    public async remove(id: string): Promise<IApiResponse<null>> {
        if (!Types.ObjectId.isValid(id)) {
            throw new ApiError(400, 'ID người dùng không hợp lệ');
        }
        const user = await UserModel.findByIdAndUpdate(id, { isActive: false, $inc: { tokenVersion: 1 } });
        if (!user) {
            throw new ApiError(404, 'Không tìm thấy tài khoản');
        }
        return { success: true, message: 'Vô hiệu hóa tài khoản thành công' };
    }
}
