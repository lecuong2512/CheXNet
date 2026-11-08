import { execFile } from 'child_process';
import path from 'path';
import { ApiError } from '@common/utils/ApiError';
import { DiagnosisService } from './diagnosis.service';
import type { IDiagnosis } from '@common/interfaces/diagnosis.interface';
import type { IScan } from '@common/interfaces/scan.interface';
import type { IPatient } from '@common/interfaces/patient.interface';

export async function renderDiagnosisPdf(id: string, revision?: string) {
    const result = await new DiagnosisService().getById(id);
    const diagnosis = result.data as IDiagnosis & { scan: IScan; patient?: IPatient };
    const currentRevision = diagnosis.revision ?? 0;
    if (revision !== undefined && (!/^\d+$/.test(revision) || Number(revision) !== currentRevision)) {
        throw new ApiError(409, 'Phiếu đã thay đổi. Vui lòng mở lại trước khi tải PDF.');
    }
    const timestamp = (value?: Date) => value ? new Date(value).toLocaleString('vi-VN', { timeZone: 'Asia/Ho_Chi_Minh' }) : undefined;
    const input = {
        scanId: String(diagnosis.scanId),
        revision: currentRevision,
        patient: { name: diagnosis.patient?.name, patientCode: diagnosis.patient?.patientCode,
            age: diagnosis.patient?.age, gender: diagnosis.patient?.gender },
        createdAt: timestamp(diagnosis.scan.createdAt),
        scanType: diagnosis.scan.type,
        model: diagnosis.aiModel,
        report: diagnosis.hybridAnalysis?.clinical_report,
        notes: diagnosis.notes,
        statusLabel: diagnosis.status === 'verified' ? 'Đã xác minh' : diagnosis.status === 'flagged' ? 'Cần xem lại' : 'Chưa xác minh',
        verifiedByName: diagnosis.verifiedByName,
        verifiedBy: diagnosis.verifiedBy,
        verifiedAt: timestamp(diagnosis.verifiedAt),
    };
    const python = process.env.PYTHON_EXECUTABLE || path.resolve('../.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    const script = path.resolve('ai_service/report_pdf.py');
    const pdf = await new Promise<Buffer>((resolve, reject) => {
        const child = execFile(python, [script], { encoding: 'buffer', windowsHide: true,
            timeout: 30000, maxBuffer: 10 * 1024 * 1024 }, (error, output) => {
            if (error) reject(new ApiError(500, 'Chưa tạo được PDF. Kiểm tra Python, reportlab và font tiếng Việt.'));
            else if (!output.subarray(0, 5).equals(Buffer.from('%PDF-'))) reject(new ApiError(500, 'PDF không hợp lệ'));
            else resolve(output);
        });
        child.stdin?.on('error', () => { /* execFile callback handles early process exit. */ });
        child.stdin?.end(JSON.stringify(input));
    });
    return { pdf, revision: currentRevision };
}
