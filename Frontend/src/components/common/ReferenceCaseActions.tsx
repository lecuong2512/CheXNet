import { useState } from 'react';
import { Button, Select, message } from 'antd';
import { useSaveReferenceCaseMutation } from '../../stores/baseApi';
import { diseaseLabel } from '../../utils/display';

export default function ReferenceCaseActions({ diagnosisId, status, probabilities, saved, disabled }: {
    diagnosisId?: string;
    status?: string;
    probabilities: { disease: string }[];
    saved?: { status: string; diseases: string[] };
    disabled?: boolean;
}) {
    const [labels, setLabels] = useState<string[]>(saved?.diseases || []);
    const [save, { isLoading }] = useSaveReferenceCaseMutation();
    const ready = saved?.status === 'ready';
    const submit = async () => {
        if (!diagnosisId) return;
        try {
            await save({ diagnosisId, diseases: labels }).unwrap();
            message.success('Đã lưu ca tham chiếu');
        } catch (error) {
            const failure = error as { data?: { message?: string } };
            message.error(failure.data?.message || 'Chưa xác nhận được thao tác lưu. Có thể thử lại.');
        }
    };
    return <div className="mt-4 space-y-3">
        <p>{ready ? 'Ca đã lưu làm tham chiếu' : 'Chọn nhãn bác sĩ xác nhận để bổ sung kho ca tương tự.'}</p>
        <Select mode="multiple" aria-label="Nhãn xác nhận cho ca tham chiếu" className="w-full"
            value={ready ? saved?.diseases : labels} onChange={setLabels} disabled={ready || isLoading || disabled}
            placeholder="Chọn nhãn bác sĩ xác nhận"
            options={probabilities.map(item => ({ value: item.disease, label: diseaseLabel(item.disease) }))} />
        <Button onClick={submit} loading={isLoading}
            disabled={!diagnosisId || status !== 'Đã xác minh' || !labels.length || ready || disabled}>
            {ready ? 'Đã lưu ca tham chiếu' : 'Lưu ca tham chiếu'}
        </Button>
        {status !== 'Đã xác minh' && <p>Xác minh phiếu bác sĩ trước khi lưu ca tham chiếu.</p>}
    </div>;
}
