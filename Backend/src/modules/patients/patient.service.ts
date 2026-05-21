import { PipelineStage, Types } from 'mongoose';
import { IApiResponse, IPaginatedResponse } from '@common/interfaces/response.interface';
import { IPatient, IPatientFilter, IPatientPayload } from '@common/interfaces/patient.interface';
import { ApiError } from '@common/utils/ApiError';
import PatientModel from './patient.model';
import { v4 as uuidv4 } from 'uuid';

export class PatientService {
    /**
     * Sinh mã bệnh nhân tự động: PT-XXXXXX
     */
    private generatePatientCode(): string {
        return `PT-${uuidv4().substring(0, 6).toUpperCase()}`;
    }

    /**
     * Lấy danh sách bệnh nhân với phân trang + tìm kiếm tiếng Việt
     */
    public async getByFilter(filter: IPatientFilter): Promise<IPaginatedResponse<IPatient>> {
        const page = filter.page || 1;
        const limit = filter.limit || 10;
        const query: Record<string, unknown> = { isActive: true };

        if (filter.gender) query.gender = filter.gender;
        if (filter.department) query.department = filter.department;
        if (filter.riskLevel) query.riskLevel = filter.riskLevel;
        if (filter.createdBy && Types.ObjectId.isValid(String(filter.createdBy))) {
            query.createdBy = new Types.ObjectId(String(filter.createdBy));
        }
        if (filter.keyword) {
            query.$or = [
                { name: { $regex: filter.keyword, $options: 'i' } },
                { patientCode: { $regex: filter.keyword, $options: 'i' } },
                { phone: { $regex: filter.keyword, $options: 'i' } },
            ];
        }

        const baseAgg: PipelineStage[] = [
            { $match: query },
            { $sort: { createdAt: -1 } as Record<string, 1 | -1> },
        ];

        const [records, counter] = await Promise.all([
            PatientModel.aggregate([
                ...baseAgg,
                { $skip: (page - 1) * limit },
                { $limit: limit },
                {
                    $lookup: {
                        from: 'users',
                        localField: 'createdBy',
                        foreignField: '_id',
                        as: 'createdByInfo',
                        pipeline: [{ $project: { name: 1, email: 1 } }],
                    },
                },
                { $unwind: { path: '$createdByInfo', preserveNullAndEmptyArrays: true } },
                {
                    $lookup: {
                        from: 'scans',
                        localField: '_id',
                        foreignField: 'patientId',
                        as: 'scans',
                        pipeline: [
                            { $match: { status: { $ne: 'failed' } } },
                            { $sort: { createdAt: -1 } },
                            { $limit: 10 },
                        ],
                    },
                },
                {
                    $lookup: {
                        from: 'diagnoses',
                        localField: '_id',
                        foreignField: 'patientId',
                        as: 'diagnoses',
                        pipeline: [{ $sort: { createdAt: -1 } }, { $limit: 10 }],
                    },
                },
            ]),
            PatientModel.aggregate([...baseAgg, { $count: 'count' }]),
        ]);

        const count = counter[0]?.count || 0;
        return {
            success: true,
            message: 'Lấy danh sách bệnh nhân thành công',
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
     * Lấy chi tiết một bệnh nhân kèm scans và diagnoses mới nhất
     */
    public async getById(id: string): Promise<IApiResponse<IPatient>> {
        // Hỗ trợ tìm kiếm bằng cả ObjectId lẫn patientCode (VD: PT-307640)
        let matchStage: Record<string, unknown>;
        if (Types.ObjectId.isValid(id) && id.length === 24) {
            matchStage = { _id: new Types.ObjectId(id), isActive: true };
        } else {
            matchStage = { patientCode: id.toUpperCase(), isActive: true };
        }

        const [patient] = await PatientModel.aggregate([
            { $match: matchStage },
            {
                $lookup: {
                    from: 'scans',
                    localField: '_id',
                    foreignField: 'patientId',
                    as: 'scans',
                    pipeline: [
                        { $match: { status: { $ne: 'failed' } } },
                        { $sort: { createdAt: -1 } },
                        { $limit: 10 },
                    ],
                },
            },
            {
                $lookup: {
                    from: 'diagnoses',
                    localField: '_id',
                    foreignField: 'patientId',
                    as: 'diagnoses',
                    pipeline: [{ $sort: { createdAt: -1 } }, { $limit: 5 }],
                },
            },
        ]);

        if (!patient) {
            throw new ApiError(404, 'Không tìm thấy hồ sơ bệnh nhân');
        }

        return { success: true, message: 'Lấy hồ sơ bệnh nhân thành công', data: patient };
    }

    /**
     * Tạo hồ sơ bệnh nhân mới
     */
    public async create(payload: IPatientPayload, createdBy: string): Promise<IApiResponse<IPatient>> {
        const patientCode = payload.patientCode || this.generatePatientCode();

        const existing = await PatientModel.findOne({ patientCode: patientCode.toUpperCase() });
        if (existing) {
            throw new ApiError(400, `Mã bệnh nhân ${patientCode} đã tồn tại trong hệ thống`);
        }

        const patient = await PatientModel.create({
            ...payload,
            patientCode: patientCode.toUpperCase(),
            createdBy: new Types.ObjectId(createdBy),
        });

        return { success: true, message: 'Thêm hồ sơ bệnh nhân thành công', data: patient };
    }

    /**
     * Cập nhật hồ sơ và vitals của bệnh nhân
     */
    public async update(id: string, payload: IPatientPayload): Promise<IApiResponse<IPatient>> {
        if (!Types.ObjectId.isValid(id)) {
            throw new ApiError(400, 'ID bệnh nhân không hợp lệ');
        }

        const patient = await PatientModel.findByIdAndUpdate(
            id,
            { $set: payload },
            { new: true, runValidators: true },
        );

        if (!patient) {
            throw new ApiError(404, 'Không tìm thấy hồ sơ bệnh nhân');
        }

        return { success: true, message: 'Cập nhật hồ sơ bệnh nhân thành công', data: patient };
    }
}
