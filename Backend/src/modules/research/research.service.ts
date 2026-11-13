import { PipelineStage } from 'mongoose';
import { IApiResponse } from '@common/interfaces/response.interface';
import DiagnosisModel from '@modules/diagnoses/diagnosis.model';
import PatientModel from '@modules/patients/patient.model';
import { CLASS_NAMES } from '@common/utils/enum';

const DAY = 86400000;
const diseases = CLASS_NAMES.filter(name => name !== 'No Finding');

function completed(since?: Date, until?: Date): PipelineStage[] {
    return [
        ...(since ? [{ $match: { createdAt: { $gte: since, ...(until ? { $lt: until } : {}) } } }] : []),
        { $lookup: { from: 'scans', localField: 'scanId', foreignField: '_id', as: 'completedScan',
            pipeline: [{ $match: { status: 'done' } }, { $project: { _id: 1 } }] } },
        { $match: { 'completedScan.0': { $exists: true } } },
        { $unset: 'completedScan' },
    ];
}

function detections(): PipelineStage[] {
    return [
        { $project: { scanId: 1, createdAt: 1, detected: {
            $cond: [{ $isArray: '$hybridAnalysis.predictions' }, '$hybridAnalysis.predictions', []],
        } } },
        { $unwind: '$detected' },
        { $match: { 'detected.disease': { $in: diseases } } },
        { $group: {
            _id: { scanId: '$scanId', disease: '$detected.disease' },
            createdAt: { $first: '$createdAt' },
            probability: { $max: '$detected.probability' },
            location: { $first: '$detected.lesion_location' },
        } },
    ];
}

export class ResearchService {
    public async getStats(weeks = 8): Promise<IApiResponse<Record<string, unknown>>> {
        const now = new Date();
        const since = new Date(now.getTime() - weeks * 7 * DAY);
        const previous = new Date(since.getTime() - weeks * 7 * DAY);
        const [totalPatients, [summary], [prior]] = await Promise.all([
            PatientModel.countDocuments({ isActive: true }),
            DiagnosisModel.aggregate([
                ...completed(since, now),
                { $facet: {
                    totals: [{ $group: { _id: null, total: { $sum: 1 }, patients: { $addToSet: '$patientId' },
                        verified: { $sum: { $cond: [{ $eq: ['$status', 'verified'] }, 1, 0] } },
                        avgTime: { $avg: { $cond: [{ $gt: ['$processingTime', 0] }, '$processingTime', null] } },
                        missingDetectionData: { $sum: { $cond: [{ $isArray: '$hybridAnalysis.predictions' }, 0, 1] } },
                    } }],
                    alerts: [
                        { $match: { 'hybridAnalysis.clinical_report.warnings.0': { $exists: true } } },
                        { $sort: { createdAt: -1 } }, { $limit: 5 },
                        { $project: { _id: 1, warnings: '$hybridAnalysis.clinical_report.warnings' } },
                    ],
                } },
            ]),
            DiagnosisModel.aggregate([...completed(previous, since), { $count: 'total' }]),
        ]);
        const totals = summary?.totals?.[0];
        const count = totals?.total ?? 0;
        return { success: true, message: 'Thống kê các lần chụp hoàn tất', data: {
            totalPatients, patientsInPeriod: totals?.patients?.length ?? 0,
            totalScans: count, totalDiagnoses: count, verifiedDiagnoses: totals?.verified ?? 0,
            recentScans: count,
            growthRate: prior?.total ? Math.round((count - prior.total) / prior.total * 1000) / 10 : null,
            avgProcessingTime: totals?.avgTime == null ? null : Math.round(totals.avgTime * 1000) / 1000,
            missingDetectionData: totals?.missingDetectionData ?? 0,
            modelVersion: 'hybrid-convnextv2-swinv2',
            period: { weeks, since: since.toISOString(), until: now.toISOString() },
            benchmarks: [
                { dataset: 'NIH · README', auroc: 0.8895, source: 'README.md mục đánh giá NIH' },
                { dataset: 'NIH + VinDr · README', auroc: 0.8758, source: 'README.md mục đánh giá Hybrid' },
                { dataset: 'NIH · Báo cáo tiến độ', auroc: 0.8546, source: 'Mẫu 10 ngày 31/10/2026, commit 5f6c424' },
            ],
            clinicalAlerts: (summary?.alerts || []).map((item: { _id: unknown; warnings: string[] }) => ({
                id: String(item._id), type: 'warning', title: 'Cảnh báo từ báo cáo AI', message: item.warnings.join(' '),
            })),
        } };
    }

    public async getTrends(weeks = 8): Promise<IApiResponse<unknown[]>> {
        const end = new Date();
        const start = new Date(end.getTime() - weeks * 7 * DAY);
        const rows = await DiagnosisModel.aggregate([
            ...completed(start, end), ...detections(),
            { $group: {
                _id: { week: { $floor: { $divide: [{ $subtract: ['$createdAt', start] }, 7 * DAY] } }, disease: '$_id.disease' },
                count: { $sum: 1 }, avgProbability: { $avg: '$probability' },
            } },
            { $sort: { '_id.week': 1, '_id.disease': 1 } },
        ]);
        return { success: true, message: 'Số lần chụp phát hiện nhãn vượt ngưỡng Hybrid', data:
            Array.from({ length: weeks }, (_, index) => ({
                week: 'T' + (index + 1),
                weekStart: new Date(start.getTime() + index * 7 * DAY).toISOString(),
                weekEnd: new Date(start.getTime() + (index + 1) * 7 * DAY).toISOString(),
                data: rows.filter(row => row._id.week === index).map(row => ({
                    _id: row._id.disease, count: row.count, avgProbability: row.avgProbability,
                })),
            })),
        };
    }

    public async getHeatmap(weeks = 8): Promise<IApiResponse<unknown[]>> {
        const end = new Date();
        const rows = await DiagnosisModel.aggregate([
            ...completed(new Date(end.getTime() - weeks * 7 * DAY), end), ...detections(),
            { $match: { location: { $nin: [null, '', 'Không xác định'] } } },
            { $group: { _id: { location: '$location', disease: '$_id.disease' }, count: { $sum: 1 } } },
            { $sort: { count: -1, '_id.location': 1, '_id.disease': 1 } },
            { $project: { _id: 0, location: '$_id.location', disease: '$_id.disease', count: 1 } },
        ]);
        return { success: true, message: 'Vị trí do pipeline ghi nhận trong kết quả từng lần chụp', data: rows };
    }
}
