import fs from 'fs';
import axios from 'axios';
import FormData from 'form-data';
import { Types } from 'mongoose';
import { AI_SERVICE_URL } from '@config/environment';
import { CLASS_NAMES, DiagnosisStatus, ScanStatus } from '@common/utils/enum';
import { ApiError } from '@common/utils/ApiError';
import DiagnosisModel from './diagnosis.model';
import ScanModel from '../scans/scan.model';

export async function saveReferenceCase(id: string, diseases: unknown, savedBy: string) {
    if (!Types.ObjectId.isValid(id)) throw new ApiError(400, 'Mã chẩn đoán không hợp lệ');
    if (!Array.isArray(diseases) || diseases.length === 0 ||
        diseases.some(label => typeof label !== 'string' || !(CLASS_NAMES as readonly string[]).includes(label))) {
        throw new ApiError(422, 'Chọn nhãn xác nhận hợp lệ');
    }
    const labels = [...new Set(diseases as string[])].sort();
    if (labels.includes('No Finding') && labels.length > 1) {
        throw new ApiError(422, 'No Finding phải được xác nhận riêng');
    }
    let diagnosis = await DiagnosisModel.findById(id);
    if (!diagnosis) throw new ApiError(404, 'Không tìm thấy chẩn đoán');
    if (diagnosis.status !== DiagnosisStatus.VERIFIED) {
        throw new ApiError(409, 'Bác sĩ cần xác minh phiếu trước khi lưu ca tham chiếu');
    }
    const scan = await ScanModel.findOne({ _id: diagnosis.scanId, status: ScanStatus.DONE });
    if (!scan) throw new ApiError(404, 'Không tìm thấy lần chụp hoàn tất');
    const referenceId = 'ref_' + id.toLowerCase();
    await DiagnosisModel.updateOne({
        _id: id, status: DiagnosisStatus.VERIFIED, referenceCase: { $exists: false },
        $or: [{ revision: diagnosis.revision ?? 0 }, ...(diagnosis.revision ? [] : [{ revision: { $exists: false } }])],
    }, { $set: { referenceCase: { referenceId, diseases: labels, status: 'pending',
        revisionAtSave: diagnosis.revision ?? 0, savedBy } } });
    diagnosis = await DiagnosisModel.findById(id);
    const reference = diagnosis?.referenceCase;
    if (!reference || JSON.stringify(reference.diseases) !== JSON.stringify(labels)) {
        throw new ApiError(409, 'Ca đã được lưu hoặc đang lưu với nhãn khác');
    }
    if (reference.status === 'ready') return { success: true, referenceId, replayed: true };
    const form = new FormData();
    const imagePath = scan.previewPath || scan.imagePath;
    if (!fs.existsSync(imagePath)) throw new ApiError(404, 'Không tìm thấy ảnh của lần chụp');
    form.append('file', fs.createReadStream(imagePath));
    form.append('reference_id', referenceId);
    form.append('diseases', labels.join(','));
    try {
        const response = await axios.post(`${AI_SERVICE_URL}/v3/reference-cases`, form, {
            headers: form.getHeaders(), timeout: 120000,
        });
        if (response.data.success !== true || response.data.referenceId !== referenceId) {
            throw new ApiError(503, 'Kho ca tham chiếu chưa xác nhận lưu thành công');
        }
        await DiagnosisModel.updateOne({ _id: id, 'referenceCase.referenceId': referenceId }, {
            $set: { 'referenceCase.status': 'ready', 'referenceCase.savedAt': new Date() },
        });
        return response.data;
    } catch (error) {
        if (error instanceof ApiError) throw error;
        if (axios.isAxiosError(error) && error.response?.status === 409) {
            throw new ApiError(409, 'Kho tham chiếu đã lưu ca này với nhãn khác');
        }
        throw new ApiError(503, 'Chưa xác nhận được thao tác lưu. Có thể thử lại cùng các nhãn đã chọn.');
    }
}
