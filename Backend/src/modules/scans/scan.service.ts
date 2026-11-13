import fs from 'fs';
import { createHash, randomUUID } from 'crypto';
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
import PatientModel from '@modules/patients/patient.model';
import { socketService } from '@socket/socket.service';

export class ScanService {
    /** Run before accepting requests, when no inference from this process is active. */
    public async recoverInterruptedUploads(): Promise<void> {
        const interrupted = await ScanModel.find({ status: { $in: [ScanStatus.PROCESSING, ScanStatus.FAILED] } });
        for (const scan of interrupted) {
            const diagnosis = await DiagnosisModel.exists({ scanId: scan._id });
            const filesExist = fs.existsSync(scan.imagePath) && (!scan.previewPath || fs.existsSync(scan.previewPath));
            if (scan.status === ScanStatus.PROCESSING && diagnosis && filesExist) {
                await ScanModel.updateOne({ _id: scan._id }, {
                    status: ScanStatus.DONE, processingCompletedAt: new Date(),
                });
            } else {
                await DiagnosisModel.deleteOne({ scanId: scan._id });
                await ScanModel.updateOne({ _id: scan._id }, {
                    status: ScanStatus.FAILED, failureMessage: 'Phân tích bị gián đoạn. Vui lòng gửi lại ảnh.',
                });
                this.deleteUploadedFile(scan.imagePath);
                if (scan.previewPath) this.deleteUploadedFile(scan.previewPath);
                this.deleteUploadedFile(scan.imagePath + '.preview.png');
            }
        }
    }

    /**
     * Commit scan and diagnosis before publishing the completed analysis.
     */
    public async uploadAndProcess(
        patientId: string,
        uploadedBy: string,
        file: Express.Multer.File,
        payload: IScanPayload,
    ): Promise<IApiResponse<Record<string, unknown>> & { replayed?: boolean }> {
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
        const requestKey = payload.requestKey || randomUUID();
        if (!/^[A-Za-z0-9_-]{8,128}$/.test(requestKey)) {
            this.deleteUploadedFile(filePath);
            throw new ApiError(400, 'Mã yêu cầu upload không hợp lệ');
        }
        const digest = createHash('sha256');
        for await (const chunk of fs.createReadStream(filePath)) digest.update(chunk);
        const requestHash = digest.update(JSON.stringify({ scanType, notes: payload.notes || '' })).digest('hex');
        let ownedScanId: Types.ObjectId | undefined;
        let committed = false;

        try {
            if (!await PatientModel.exists({ _id: patientId, isActive: true })) {
                throw new ApiError(404, 'Không tìm thấy bệnh nhân');
            }
            let scan;
            try {
                scan = await ScanModel.create({
                    patientId: new Types.ObjectId(patientId),
                    imagePath: filePath,
                    imageUrl,
                    type: scanType,
                    notes: payload.notes,
                    status: ScanStatus.PROCESSING,
                    uploadedBy: new Types.ObjectId(uploadedBy),
                    processingStartedAt: startedAt,
                    requestKey,
                    requestHash,
                });
            } catch (error) {
                if ((error as { code?: number }).code !== 11000) throw error;
                const existing = await ScanModel.findOne({ patientId, uploadedBy, requestKey });
                if (!existing || existing.requestHash !== requestHash) {
                    throw new ApiError(409, 'Mã yêu cầu đã dùng cho nội dung khác');
                }
                if (existing.status !== ScanStatus.DONE) {
                    throw new ApiError(409, existing.status === ScanStatus.FAILED
                        ? 'Yêu cầu trước đã thất bại. Vui lòng tạo lần phân tích mới.'
                        : 'Ảnh đang được phân tích. Vui lòng thử lại sau.');
                }
                const existingDiagnosis = await DiagnosisModel.findOne({ scanId: existing._id });
                if (!existingDiagnosis) throw new ApiError(409, 'Kết quả đang được đối soát');
                this.deleteUploadedFile(filePath);
                return { success: true, replayed: true, message: 'Đã lấy kết quả phân tích đã lưu',
                    data: { ...existing.toObject(), diagnosis: existingDiagnosis.toObject() } };
            }
            ownedScanId = scan._id;
            const { predictions, processingTime, aiModel, hybridAnalysis, inferenceMeta, previewPng } = await this.callAIService(filePath);
            if (previewPng) {
                const preview = Buffer.from(previewPng, 'base64');
                if (preview.length > 60 * 1024 * 1024 || preview.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a') {
                    throw new ApiError(500, 'Dịch vụ AI trả ảnh xem không hợp lệ');
                }
                scan.previewPath = filePath + '.preview.png';
                await fs.promises.writeFile(scan.previewPath, preview, { flag: 'wx' });
            }
            scan.imageUrl = `/api/v1/scans/${scan._id}/preview`;
            await scan.save();

            const diagnosis = await DiagnosisModel.create({
                scanId: scan._id,
                patientId: new Types.ObjectId(patientId),
                predictions,
                aiModel,
                processingTime,
                hybridAnalysis,
                inferenceMeta,
            });

            scan.status = ScanStatus.DONE;
            scan.processingCompletedAt = new Date();
            await scan.save();
            committed = true;
            const scanId = scan._id.toString();
            try {
                socketService.sendToRoom(`patient_${patientId}`, 'scan:completed', {
                    scanId,
                    patientId,
                    predictions,
                    processingTime,
                });
            } catch (notificationError) {
                logger.warn('Không gửi được thông báo cho ca đã lưu', notificationError);
            }

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
            if (!committed) {
                if (ownedScanId) {
                    // Persist failure before deleting files. Startup recovery handles a DB outage.
                    await ScanModel.updateOne({ _id: ownedScanId }, {
                        status: ScanStatus.FAILED,
                        failureMessage: error instanceof Error ? error.message : 'Phân tích thất bại',
                    });
                    await DiagnosisModel.deleteOne({ scanId: ownedScanId });
                }
                this.deleteUploadedFile(filePath);
                this.deleteUploadedFile(filePath + '.preview.png');
            }

            if (error instanceof ApiError) {
                throw error;
            }

            const failureMessage =
                error instanceof Error ? error.message : 'Phân tích AI thất bại';
            logger.error(`Upload thất bại: ${failureMessage}`);
            throw new ApiError(500, failureMessage);
        }
    }

    private async callAIService(filePath: string): Promise<{
        predictions: { className: string; probability: number }[];
        processingTime: number;
        aiModel: string;
        hybridAnalysis: Record<string, unknown>;
        inferenceMeta?: Record<string, unknown>;
        previewPng?: string;
    }> {
        const startTime = Date.now();
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));

        try {
            const response = await axios.post(`${AI_SERVICE_URL}/v3/predict`, form, {
                headers: form.getHeaders(),
                timeout: 120000,
            });

            if (!response.data.success || response.data.predictions?.length !== 15) {
                throw new ApiError(500, 'Dịch vụ AI trả kết quả không đủ 15 xác suất');
            }
            return {
                predictions: response.data.predictions,
                processingTime:
                    response.data.processingTime || (Date.now() - startTime) / 1000,
                aiModel: response.data.modelVersion,
                hybridAnalysis: response.data.hybridAnalysis,
                inferenceMeta: response.data.inferenceMeta,
                previewPng: response.data.previewPng,
            };
        } catch (aiError) {
            if (aiError instanceof ApiError) throw aiError;
            if (isAxiosError(aiError) && [413, 422].includes(aiError.response?.status || 0)) {
                const detail = aiError.response?.data?.detail;
                const message =
                    typeof detail === 'string'
                        ? detail
                        : 'Ảnh không phải phim X-quang phổi hợp lệ';
                throw new ApiError(aiError.response!.status, message);
            }
            if (isAxiosError(aiError) && aiError.code === 'ECONNABORTED') {
                throw new ApiError(504, 'Phân tích quá thời gian chờ. Vui lòng thử lại.');
            }
            if (isAxiosError(aiError) && aiError.response && aiError.response.status >= 500) {
                throw new ApiError(aiError.response.status === 503 ? 503 : 500,
                    'Dịch vụ AI chưa hoàn tất phân tích. Vui lòng kiểm tra trạng thái và thử lại.');
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
            status: ScanStatus.DONE,
        };

        if (filter.patientId && Types.ObjectId.isValid(String(filter.patientId))) {
            query.patientId = new Types.ObjectId(String(filter.patientId));
        }
        if (filter.status && filter.status === ScanStatus.DONE) {
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
