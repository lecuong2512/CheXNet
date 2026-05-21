import { PipelineStage, Types } from 'mongoose';
import { IApiResponse, IPaginatedResponse } from '@common/interfaces/response.interface';
import { IDiagnosis, IDiagnosisFilter } from '@common/interfaces/diagnosis.interface';
import { DiagnosisStatus } from '@common/utils/enum';
import { ApiError } from '@common/utils/ApiError';
import DiagnosisModel from './diagnosis.model';

export class DiagnosisService {
    /**
     * Lấy danh sách chẩn đoán của một bệnh nhân kèm thông tin scan
     */
    public async getByPatient(filter: IDiagnosisFilter): Promise<IPaginatedResponse<IDiagnosis>> {
        const page = filter.page || 1;
        const limit = filter.limit || 10;
        const query: Record<string, unknown> = {};

        if (filter.patientId && Types.ObjectId.isValid(String(filter.patientId))) {
            query.patientId = new Types.ObjectId(String(filter.patientId));
        }
        if (filter.status) query.status = filter.status;

        const baseAgg: PipelineStage[] = [
            { $match: query },
            { $sort: { createdAt: -1 } as Record<string, 1 | -1> },
        ];

        const [records, counter] = await Promise.all([
            DiagnosisModel.aggregate([
                ...baseAgg,
                { $skip: (page - 1) * limit },
                { $limit: limit },
                {
                    $lookup: {
                        from: 'scans',
                        localField: 'scanId',
                        foreignField: '_id',
                        as: 'scan',
                    },
                },
                { $unwind: { path: '$scan', preserveNullAndEmptyArrays: true } },
                {
                    $lookup: {
                        from: 'patients',
                        localField: 'patientId',
                        foreignField: '_id',
                        as: 'patient',
                        pipeline: [{ $project: { name: 1, patientCode: 1, gender: 1, age: 1 } }],
                    },
                },
                { $unwind: { path: '$patient', preserveNullAndEmptyArrays: true } },
            ]),
            DiagnosisModel.aggregate([...baseAgg, { $count: 'count' }]),
        ]);

        const count = counter[0]?.count || 0;
        return {
            success: true,
            message: 'Lấy lịch sử chẩn đoán thành công',
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
     * Lấy chi tiết một kết quả chẩn đoán theo ID
     */
    public async getById(id: string): Promise<IApiResponse<IDiagnosis>> {
        if (!Types.ObjectId.isValid(id)) {
            throw new ApiError(400, 'ID chẩn đoán không hợp lệ');
        }

        const [diagnosis] = await DiagnosisModel.aggregate([
            { $match: { _id: new Types.ObjectId(id) } },
            {
                $lookup: {
                    from: 'scans',
                    localField: 'scanId',
                    foreignField: '_id',
                    as: 'scan',
                },
            },
            { $unwind: { path: '$scan', preserveNullAndEmptyArrays: true } },
            {
                $lookup: {
                    from: 'patients',
                    localField: 'patientId',
                    foreignField: '_id',
                    as: 'patient',
                },
            },
            { $unwind: { path: '$patient', preserveNullAndEmptyArrays: true } },
        ]);

        if (!diagnosis) {
            throw new ApiError(404, 'Không tìm thấy kết quả chẩn đoán');
        }

        return { success: true, message: 'Lấy kết quả chẩn đoán thành công', data: diagnosis };
    }

    /**
     * Bác sĩ xác minh hoặc đánh dấu kết quả chẩn đoán AI
     */
    public async verify(
        id: string,
        status: DiagnosisStatus,
        notes: string | undefined,
        verifiedBy: string,
    ): Promise<IApiResponse<IDiagnosis>> {
        if (!Types.ObjectId.isValid(id)) {
            throw new ApiError(400, 'ID chẩn đoán không hợp lệ');
        }
        if (![DiagnosisStatus.VERIFIED, DiagnosisStatus.FLAGGED].includes(status)) {
            throw new ApiError(400, 'Trạng thái xác minh không hợp lệ');
        }

        const diagnosis = await DiagnosisModel.findByIdAndUpdate(
            id,
            {
                status,
                notes,
                verifiedBy: new Types.ObjectId(verifiedBy),
                verifiedAt: new Date(),
            },
            { new: true },
        );

        if (!diagnosis) {
            throw new ApiError(404, 'Không tìm thấy kết quả chẩn đoán');
        }

        const label = status === DiagnosisStatus.VERIFIED ? 'Đã xác minh' : 'Đánh dấu';
        return { success: true, message: `Kết quả chẩn đoán đã được ${label}`, data: diagnosis };
    }

    /**
     * Lấy tất cả chẩn đoán cho bảng lịch sử (có filter đa dạng)
     */
    public async getAll(filter: IDiagnosisFilter & { keyword?: string }): Promise<IPaginatedResponse<IDiagnosis>> {
        const page = filter.page || 1;
        const limit = filter.limit || 10;
        const query: Record<string, unknown> = {};

        if (filter.status) query.status = filter.status;
        if (filter.verifiedBy && Types.ObjectId.isValid(String(filter.verifiedBy))) {
            query.verifiedBy = new Types.ObjectId(String(filter.verifiedBy));
        }

        const baseAgg: PipelineStage[] = [
            { $match: query },
            { $sort: { createdAt: -1 } as Record<string, 1 | -1> },
        ];

        const [records, counter] = await Promise.all([
            DiagnosisModel.aggregate([
                ...baseAgg,
                { $skip: (page - 1) * limit },
                { $limit: limit },
                {
                    $lookup: {
                        from: 'scans',
                        localField: 'scanId',
                        foreignField: '_id',
                        as: 'scan',
                    },
                },
                { $unwind: { path: '$scan', preserveNullAndEmptyArrays: true } },
                {
                    $lookup: {
                        from: 'patients',
                        localField: 'patientId',
                        foreignField: '_id',
                        as: 'patient',
                        pipeline: [{ $project: { name: 1, patientCode: 1, age: 1, gender: 1 } }],
                    },
                },
                { $unwind: { path: '$patient', preserveNullAndEmptyArrays: true } },
            ]),
            DiagnosisModel.aggregate([...baseAgg, { $count: 'count' }]),
        ]);

        const count = counter[0]?.count || 0;
        return {
            success: true,
            message: 'Lấy lịch sử chẩn đoán thành công',
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
}
