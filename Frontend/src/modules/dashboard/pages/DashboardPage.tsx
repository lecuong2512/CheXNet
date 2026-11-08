import { useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Alert, Button, Empty, Progress, Select, Spin } from 'antd';
import DiagnosisReport from '../../../components/common/DiagnosisReport';
import AiStatus from '../../../components/common/AiStatus';
import HybridAnalysisPanel from '../../../components/common/HybridAnalysisPanel';
import { useGetPatientsQuery, useGetPatientProfileQuery, useGetScanDetailQuery } from '../../../stores/baseApi';
import { formatNumber, scanTypeLabel, scanChoiceLabel, diagnosisStatusLabel } from '../../../utils/display';

export default function DashboardPage() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [reportOpen, setReportOpen] = useState(false);
  const { data: patients, isLoading: loadingPatients, error: patientsError, refetch } = useGetPatientsQuery();
  const options = useMemo(() => [...new Map((patients || []).map(p => [p.patientId, {
    value: p.patientId, label: `${p.patientName} · ${p.patientId}`,
  }])).values()], [patients]);
  const activePatientId = params.get('patient') || options[0]?.value || '';
  const { currentData: patient, isFetching: loadingProfile, error: profileError } = useGetPatientProfileQuery(activePatientId, { skip: !activePatientId });
  const requestedScan = params.get('scan');
  const scanId = requestedScan || patient?.scans?.[0]?.id || '';
  const { currentData: detail, isFetching: loadingDetail, error: detailError } = useGetScanDetailQuery(scanId, { skip: !scanId });
  const summaryScan = patient?.scans?.find((scan: { id: string }) => scan.id === scanId);
  const activeScan = detail?.id === scanId && detail.patientMongoId === patient?._id ? detail : summaryScan;
  const busy = loadingPatients || loadingProfile || loadingDetail;
  const addPatient = () => window.dispatchEvent(new CustomEvent('open-add-patient-modal'));

  return <div className="page-shell space-y-5">
    <header className="page-heading">
      <div><p className="eyebrow">KHÔNG GIAN LÀM VIỆC</p><h1>Phân tích X-quang</h1>
        <p className="page-description">Chọn bệnh nhân và lần chụp để xem ảnh, kết quả AI và phiếu bác sĩ.</p></div>
      <div className="flex flex-wrap gap-3 items-center"><AiStatus /><Button type="primary" onClick={addPatient}>+ Thêm bệnh nhân</Button></div>
    </header>
    <section className="ui-card case-selector" aria-label="Chọn hồ sơ phân tích">
      <label className="field-label">Bệnh nhân
        <Select aria-label="Chọn bệnh nhân" showSearch optionFilterProp="label" value={activePatientId || undefined}
          onChange={patientId => { setReportOpen(false); setParams({ patient: patientId }); }} options={options}
          placeholder="Tìm tên hoặc mã bệnh nhân" loading={loadingPatients} className="w-full" /></label>
      <label className="field-label">Lần chụp
        <Select aria-label="Chọn lần chụp" value={scanId || undefined} disabled={!patient?.scans?.length}
          onChange={id => { setReportOpen(false); setParams({ patient: activePatientId, scan: id }); }}
          options={(patient?.scans || []).map((scan: { id: string; date: string; time: string; type: string }) => ({
            value: scan.id, label: scanChoiceLabel(scan),
          }))} placeholder="Chưa có phim chụp" className="w-full" /></label>
      <Button disabled={!activePatientId} onClick={() => navigate(`/patient-profile/${activePatientId}${scanId ? `?scan=${scanId}` : ''}`)}>Mở hồ sơ bệnh nhân</Button>
    </section>
    {(patientsError || profileError) && <Alert type="error" showIcon title="Chưa tải được hồ sơ bệnh nhân" action={<Button onClick={() => refetch()}>Thử lại</Button>} />}
    {detailError && <Alert type="error" showIcon title="Chưa tải được kết quả của lần chụp này. Vui lòng chọn lại lần chụp hoặc mở lại trang." />}
    {busy ? <div className="ui-card py-16 text-center" role="status"><Spin /><p className="mt-3">Đang tải ảnh và kết quả đã lưu…</p></div> : activeScan ? <>
      <div className="analysis-workspace">
        <div className="min-w-0"><HybridAnalysisPanel key={activeScan.id} analysis={activeScan.hybridAnalysis}
          inferenceMeta={activeScan.inferenceMeta} originalImage={activeScan.image} diagnosisId={activeScan.diagnosisId} /></div>
        <aside className="space-y-4 min-w-0" aria-label="Tóm tắt kết quả">
          <section className="ui-card space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2"><h2 className="section-title">Kết quả AI</h2><span className="status-pill">{diagnosisStatusLabel(activeScan.status)}</span></div>
            <p className="text-sm leading-relaxed text-secondary dark:text-gray-300">{activeScan.description}</p>
            <p className="text-xs text-secondary dark:text-gray-400">Điểm dự đoán cho từng dấu hiệu; không phải độ chính xác của chẩn đoán.</p>
            <div className="space-y-3">{activeScan.predictions?.length ? activeScan.predictions.map((prediction: { name: string; probability: number }) =>
              <div key={prediction.name}><div className="flex justify-between gap-3 text-sm"><span>{prediction.name}</span><strong className="whitespace-nowrap">{formatNumber(prediction.probability, 0)}%</strong></div>
                <Progress percent={prediction.probability} showInfo={false} strokeColor="#2563eb" /></div>) : <p className="text-sm">Không có nhãn bệnh vượt ngưỡng hiển thị.</p>}</div>
            <Button type="primary" block disabled={!activeScan.diagnosisId} onClick={() => setReportOpen(true)}>Mở phiếu kết quả & xác minh</Button>
          </section>
          {activeScan.nextSteps?.length > 0 && <details className="ui-card disclosure"><summary>Khuyến nghị do AI đề xuất</summary>
            <ul className="list-disc pl-5 space-y-2 mt-3 text-sm">{activeScan.nextSteps.map((step: string, index: number) => <li key={index}>{step}</li>)}</ul></details>}
          <p className="text-xs text-secondary dark:text-gray-400 px-1">{scanTypeLabel(activeScan.type)} · {activeScan.date} {activeScan.time}
            {activeScan.processingTime != null && <> · Xử lý trong {formatNumber(activeScan.processingTime, 2)} giây</>}</p>
        </aside>
      </div>
      {patient && <DiagnosisReport key={activeScan.id} open={reportOpen} onClose={() => setReportOpen(false)} scan={activeScan} patient={patient} />}
    </> : !patientsError && !profileError && <section className="ui-card py-14"><Empty description={activePatientId ? 'Bệnh nhân chưa có ảnh X-quang để phân tích.' : 'Chưa có hồ sơ bệnh nhân.'}>
      <Button type="primary" onClick={() => activePatientId ? navigate(`/patient-profile/${activePatientId}`) : addPatient()}>{activePatientId ? 'Mở hồ sơ để tải ảnh' : 'Thêm bệnh nhân đầu tiên'}</Button>
    </Empty></section>}
  </div>;
}
