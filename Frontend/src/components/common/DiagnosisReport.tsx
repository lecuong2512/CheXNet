import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Alert, Button, Input, Modal, message } from 'antd';
import { useSelector } from 'react-redux';
import type { RootState } from '../../stores/store';
import { deviceLabel, scanTypeLabel, diagnosisStatusLabel } from '../../utils/display';
import { formatAiModelLabel } from '../../utils/aiModel';
import { useVerifyScanMutation } from '../../stores/baseApi';
import type { HybridAnalysis, InferenceMeta } from '../../types/hybrid';
import ReferenceCaseActions from './ReferenceCaseActions';

interface ReportScan {
    id: string;
    diagnosisId?: string;
    type?: string;
    date?: string;
    time?: string;
    status?: string;
    revision?: number;
    notes?: string;
    verifiedAt?: string;
    verifiedBy?: string;
    verifiedByName?: string;
    aiModel?: string;
    inferenceMeta?: InferenceMeta;
    hybridAnalysis?: HybridAnalysis;
    description?: string;
    referenceCase?: { status: string; diseases: string[] };
}

interface ReportPatient {
    id: string;
    name: string;
    age?: number;
    gender?: string;
}

export default function DiagnosisReport({ open, onClose, scan, patient }: {
    open: boolean;
    onClose: () => void;
    scan: ReportScan;
    patient: ReportPatient;
}) {
    const [notes, setNotes] = useState(scan.notes || '');
    const [conflict, setConflict] = useState(false);
    const [editingRevision, setEditingRevision] = useState(scan.revision ?? 0);
    const [downloading, setDownloading] = useState(false);
    const [verify, { isLoading }] = useVerifyScanMutation();
    const role = useSelector((state: RootState) => state.auth.user?.role);
    const token = useSelector((state: RootState) => state.auth.accessToken);
    const canVerify = role === 'admin' || role === 'doctor';
    useEffect(() => {
        if (open) {
            setNotes(scan.notes || '');
            setConflict(false);
            setEditingRevision(scan.revision ?? 0);
        }
    }, [open, scan.id]);
    const changed = notes !== (scan.notes || '');
    const downloadPdf = async () => {
        if (!scan.diagnosisId || !token) return;
        setDownloading(true);
        try {
            const origin = import.meta.env.VITE_BACKEND_ORIGIN || 'http://127.0.0.1:3006';
            const response = await fetch(`${origin}/api/v1/diagnoses/${scan.diagnosisId}/report.pdf?revision=${scan.revision ?? 0}`, {
                headers: { Authorization: 'Bearer ' + token },
            });
            if (!response.ok) {
                const failure = await response.json();
                throw new Error(failure.message || 'Chưa tải được PDF');
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `chexnet-${scan.id}-v${scan.revision ?? 0}.pdf`;
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch (error) {
            message.error(error instanceof Error ? error.message : 'Chưa tải được PDF');
        } finally {
            setDownloading(false);
        }
    };
    const save = async (status: 'Đã xác minh' | 'Đánh dấu') => {
        if (!scan.diagnosisId) return;
        try {
            await verify({ patientId: patient.id, scanId: scan.id, diagnosisId: scan.diagnosisId,
                status, notes, expectedRevision: editingRevision }).unwrap();
            message.success('Đã lưu phiếu của lần chụp đang chọn');
            onClose();
        } catch (error) {
            const failure = error as { status?: number; data?: { message?: string } };
            if (failure.status === 409) setConflict(true);
            message.error(failure.data?.message || 'Chưa lưu được phiếu. Vui lòng thử lại.');
        }
    };
    const report = scan.hybridAnalysis?.clinical_report;
    const body = (
        <article className="space-y-4 text-on-surface dark:text-gray-200">
            <h1 className="text-xl font-bold">Phiếu phân tích X-quang ngực</h1>
            <p>{patient.name} · {patient.id} · {patient.age ?? '—'} tuổi · {patient.gender || '—'}</p>
            <p>Mã lần chụp {scan.id} · {scan.date} {scan.time}</p>
            <p>Trạng thái: {diagnosisStatusLabel(scan.status)} · Phiên bản {scan.revision ?? 0}</p>
            <h2 className="font-bold">Kỹ thuật và dữ liệu phân tích</h2>
            <p>{scanTypeLabel(scan.type)} · {formatAiModelLabel(scan.aiModel)} · {deviceLabel(scan.inferenceMeta?.device)}</p>
            <h2 className="font-bold">Mô tả do AI đề xuất</h2>
            {report?.findings?.length ? report.findings.map((finding, index) =>
                <p key={index}>{finding.finding_text}</p>) : <p>{report?.summary || scan.description || 'Chưa có bản mô tả AI'}</p>}
            <h2 className="font-bold">Kết luận AI</h2>
            <p>{report?.summary || scan.description || 'Chưa có kết luận AI'}</p>
            <h2 className="font-bold">Khuyến nghị do AI đề xuất</h2>
            <ul>{report?.recommendations?.map((recommendation, index) => <li key={index}>{recommendation}</li>)}</ul>
            {report?.warnings?.map((warning, index) => <p key={index}>{warning}</p>)}
            <h2 className="font-bold">Kết luận bác sĩ đã lưu</h2>
            <p className="whitespace-pre-wrap">{scan.notes || 'Chưa có kết luận bác sĩ'}</p>
            {scan.verifiedAt && <p>Người cập nhật {scan.verifiedByName || scan.verifiedBy || 'Chưa ghi nhận'} · {new Date(scan.verifiedAt).toLocaleString('vi-VN')}</p>}
        </article>
    );
    return <>
        <Modal open={open} onCancel={onClose} closable={isLoading ? false : { 'aria-label': 'Đóng phiếu kết quả' }} keyboard={!isLoading} mask={{ closable: !isLoading }}
            width={850} title="Phiếu kết quả X-quang" footer={null} destroyOnHidden>
            {body}
            {conflict && <Alert className="my-4" type="warning" showIcon title="Phiếu đã thay đổi. Đóng và mở lại để xem bản mới trước khi sửa." />}
            {canVerify && <div className="my-4 space-y-2">
                <label htmlFor="doctor-conclusion">Nhập kết luận bác sĩ</label>
                <Input.TextArea id="doctor-conclusion" value={notes} maxLength={20000} rows={5}
                    onChange={event => setNotes(event.target.value)} disabled={isLoading || conflict} />
            </div>}
            <div className="flex gap-2 flex-wrap mt-4">
                {canVerify && <>
                    <Button disabled={!scan.diagnosisId || conflict} loading={isLoading} onClick={() => save('Đánh dấu')}>Đánh dấu xem lại</Button>
                    <Button type="primary" disabled={!scan.diagnosisId || conflict} loading={isLoading} onClick={() => save('Đã xác minh')}>Lưu và xác minh</Button>
                </>}
                <Button disabled={changed || isLoading} title={changed ? 'Lưu kết luận trước khi in' : undefined}
                    onClick={() => window.print()}>In phiếu</Button>
                <Button loading={downloading} disabled={changed || isLoading || !scan.diagnosisId}
                    onClick={downloadPdf}>Tải PDF</Button>
            </div>
            {canVerify && <ReferenceCaseActions key={scan.id + String(open)} diagnosisId={scan.diagnosisId}
                status={scan.status} probabilities={scan.hybridAnalysis?.all_probabilities || []} saved={scan.referenceCase}
                disabled={changed || isLoading || conflict} />}
        </Modal>
        {open && createPortal(<div id="clinical-print-root">{body}</div>, document.body)}
    </>;
}
