import fs from 'fs';
import axios, { isAxiosError } from 'axios';
import FormData from 'form-data';
import { PipelineStage, Types } from 'mongoose';
import { IApiResponse, IPaginatedResponse } from '@common/interfaces/response.interface';
import { IScan, IScanFilter, IScanPayload } from '@common/interfaces/scan.interface';
import { ApiError } from '@common/utils/ApiError';
import { ScanStatus, ScanType } from '@common/utils/enum';
import { AI_SERVICE_URL } from '@config/environment';
import { logger } from '@common/utils/logger';
import ScanModel from './scan.model';
import DiagnosisModel from '@modules/diagnoses/diagnosis.model';
import { socketService } from '@socket/socket.service';

export class ScanService {
    /**
     * Upload ảnh X-quang: gọi AI trước, chỉ lưu DB + file khi phân tích thành công.
     */
    public async uploadAndProcess(
        patientId: string,
        uploadedBy: string,
        file: Express.Multer.File,
        payload: IScanPayload,
    ): Promise<IApiResponse<IScan & { diagnosis?: unknown }>> {
        if (!Types.ObjectId.isValid(patientId)) {
            this.deleteUploadedFile(file.path);
            throw new ApiError(400, 'ID bệnh nhân không hợp lệ');
        }

        const scanType = payload.type || ScanType.PA;
        if (!Object.values(ScanType).includes(scanType as ScanType)) {
            this.deleteUploadedFile(file.path);
            throw new ApiError(
                400,
                'Loại phim không hợp lệ. Hệ thống chỉ hỗ trợ X-quang ngực (PA / AP / Nghiêng).',
            );
        }

        const filePath = file.path;
        const imageUrl = `/uploads/${file.filename}`;
        const startedAt = new Date();

        try {
            const { predictions, processingTime, aiModel } = await this.callAIService(filePath);

            const scan = await ScanModel.create({
                patientId: new Types.ObjectId(patientId),
                imagePath: filePath,
                imageUrl,
                type: scanType,
                notes: payload.notes,
                status: ScanStatus.DONE,
                uploadedBy: new Types.ObjectId(uploadedBy),
                processingStartedAt: startedAt,
                processingCompletedAt: new Date(),
            });

            const diagnosis = await DiagnosisModel.create({
                scanId: scan._id,
                patientId: new Types.ObjectId(patientId),
                predictions,
                aiModel,
                processingTime,
            });

            const scanId = scan._id.toString();
            socketService.sendToRoom(`patient_${patientId}`, 'scan:completed', {
                scanId,
                patientId,
                predictions,
                processingTime,
            });

            logger.info(`✅ Upload + AI hoàn tất: scan ${scanId}`);

            return {
                success: true,
                message: 'Phân tích X-quang ngực thành công',
                data: {
                    ...scan.toObject(),
                    diagnosis: diagnosis.toObject(),
                },
            };
        } catch (error) {
            this.deleteUploadedFile(filePath);

            if (error instanceof ApiError) {
                throw error;
            }

            const failureMessage =
                error instanceof Error ? error.message : 'Phân tích AI thất bại';
            logger.error(`❌ Upload thất bại — không lưu scan: ${failureMessage}`);
            throw new ApiError(500, failureMessage);
        }
    }

    private async callAIService(filePath: string): Promise<{
        predictions: { className: string; probability: number }[];
        processingTime: number;
        aiModel: string;
    }> {
        const startTime = Date.now();
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));

        try {
            const response = await axios.post(`${AI_SERVICE_URL}/predict`, form, {
                headers: form.getHeaders(),
                timeout: 120000,
            });

            return {
                predictions: response.data.predictions || [],
                processingTime:
                    response.data.processingTime || (Date.now() - startTime) / 1000,
                aiModel: response.data.modelVersion || 'chexnet-unknown',
            };
        } catch (aiError) {
            if (isAxiosError(aiError) && aiError.response?.status === 422) {
                const detail = aiError.response.data?.detail;
                const message =
                    typeof detail === 'string'
                        ? detail
                        : 'Ảnh không phải phim X-quang phổi hợp lệ';
                throw new ApiError(422, message);
            }
            logger.error('AI service không khả dụng:', aiError);
            throw new ApiError(
                503,
                'AI service không khả dụng. Kiểm tra dịch vụ AI (port 8000) và thử lại.',
            );
        }
    }

    private deleteUploadedFile(filePath: string): void {
        try {
            if (filePath && fs.existsSync(filePath)) {
                fs.unlinkSync(filePath);
                logger.info(`Đã xóa file upload thất bại: ${filePath}`);
            }
        } catch (err) {
            logger.warn(`Không xóa được file ${filePath}:`, err);
        }
    }

    /**
     * Lấy danh sách scans của một bệnh nhân
     */
    public async getByPatient(filter: IScanFilter): Promise<IPaginatedResponse<IScan>> {
        const page = filter.page || 1;
        const limit = filter.limit || 10;
        const query: Record<string, unknown> = {
            status: { $ne: ScanStatus.FAILED },
        };

        if (filter.patientId && Types.ObjectId.isValid(String(filter.patientId))) {
            query.patientId = new Types.ObjectId(String(filter.patientId));
        }
        if (filter.status) {
            query.status = filter.status;
        }
        if (filter.type) query.type = filter.type;

        const baseAgg: PipelineStage[] = [
            { $match: query },
            { $sort: { createdAt: -1 } as Record<string, 1 | -1> },
        ];

        const [records, counter] = await Promise.all([
            ScanModel.aggregate([
                ...baseAgg,
                { $skip: (page - 1) * limit },
                { $limit: limit },
                {
                    $lookup: {
                        from: 'diagnoses',
                        localField: '_id',
                        foreignField: 'scanId',
                        as: 'diagnosis',
                    },
                },
                { $unwind: { path: '$diagnosis', preserveNullAndEmptyArrays: true } },
            ]),
            ScanModel.aggregate([...baseAgg, { $count: 'count' }]),
        ]);

        const count = counter[0]?.count || 0;
        return {
            success: true,
            message: 'Lấy danh sách ảnh X-ray thành công',
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
