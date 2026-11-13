import { useState } from 'react';
import { Empty, Select, Tabs, Tag } from 'antd';
import { Link } from 'react-router-dom';
import { useGetSimilarCasesQuery } from '../../stores/baseApi';
import type { HybridAnalysis, InferenceMeta } from '../../types/hybrid';
import { diseaseLabel, formatNumber, formatPercent } from '../../utils/display';
import XrayViewer from './XrayViewer';
import ClassificationDetails from './ClassificationDetails';
import InferenceDetails from './InferenceDetails';

export default function HybridAnalysisPanel({ analysis, inferenceMeta, originalImage, diagnosisId }: {
  analysis?: HybridAnalysis; inferenceMeta?: InferenceMeta; originalImage?: string; diagnosisId?: string;
}) {
  const [selectedDisease, setSelectedDisease] = useState('');
  const { currentData: linkedCases, isFetching: loadingCases } = useGetSimilarCasesQuery(diagnosisId || '', { skip: !diagnosisId || !analysis });
  if (!analysis) return <section className="ui-card space-y-4"><h2 className="section-title">Ảnh X-quang</h2>
    <XrayViewer originalImage={originalImage} /><p>Chưa có dữ liệu phân tích chi tiết cho lần chụp này.</p></section>;
  const hasPathologyFindings = inferenceMeta?.classification
    ? inferenceMeta.classification.findings.length > 0
    : analysis.predictions.some(p => p.disease !== 'No Finding');
  const basePredictions = hasPathologyFindings
    ? analysis.predictions.filter(p => p.disease !== 'No Finding')
    : analysis.predictions;
  const sortedPredictions = [...basePredictions].sort((a, b) => b.probability - a.probability);
  const probByDisease = new Map(analysis.all_probabilities.map(p => [p.disease, p.probability]));
  const extraDiseases = new Set<string>();
  for (const item of inferenceMeta?.classification?.findings || []) {
    if (item.disease !== 'No Finding') extraDiseases.add(item.disease);
  }
  for (const box of analysis.hybrid_boxes || []) {
    if (box.disease !== 'No Finding') extraDiseases.add(box.disease);
  }
  const predictionDiseases = new Set(sortedPredictions.map(p => p.disease));
  const extraPositiveOptions = [...extraDiseases]
    .filter(disease => !predictionDiseases.has(disease))
    .map(disease => {
      const boxProb = (analysis.hybrid_boxes || []).find(b => b.disease === disease)?.probability;
      const prob = probByDisease.get(disease) ?? boxProb ?? 0;
      return { value: disease, probability: prob, label: `${diseaseLabel(disease)} · ${formatPercent(prob)}` };
    })
    .sort((a, b) => b.probability - a.probability);
  const coveredYoloDiseases = new Set([...predictionDiseases, ...extraPositiveOptions.map(o => o.value)]);
  const extraYoloScores = new Map<string, number>();
  for (const det of analysis.yolo_detections || []) {
    if (!det.disease || coveredYoloDiseases.has(det.disease)) continue;
    const score = Math.max(det.cascade_score ?? 0, det.confidence ?? 0);
    extraYoloScores.set(det.disease, Math.max(extraYoloScores.get(det.disease) ?? 0, score));
  }
  const extraYoloOptions = [...extraYoloScores.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([disease, score]) => ({ value: disease, label: `${diseaseLabel(disease)} · Điểm YOLO ${formatPercent(score)}` }));
  const selectableOptions = [
    ...sortedPredictions.map(p => ({ value: p.disease, label: `${diseaseLabel(p.disease)} · ${formatPercent(p.probability)}`, prediction: p })),
    ...extraPositiveOptions.map(({ value, label }) => ({ value, label, prediction: undefined })),
    ...extraYoloOptions.map(({ value, label }) => ({ value, label, prediction: undefined })),
  ];
  const displayedProbabilities = hasPathologyFindings
    ? analysis.all_probabilities.filter(p => p.disease !== 'No Finding')
    : analysis.all_probabilities;
  const selectedOption = selectableOptions.find(o => o.value === selectedDisease) ?? selectableOptions[0];
  const selected = selectedOption?.prediction;
  const report = analysis.clinical_report;
  const similar = linkedCases || analysis.similar_cases;
  return <section className="ui-card analysis-panel">
    <h2 className="section-title mb-1">Ảnh và phân tích</h2>
    <Tabs className="analysis-tabs" tabBarGutter={20} items={[
      { key: 'attention', label: 'Ảnh X-quang', children: <div className="space-y-4">
        {selectedOption && <label className="field-label">Dấu hiệu xem khung & bản đồ chú ý
          <Select aria-label="Chọn dấu hiệu xem khung và bản đồ chú ý" className="w-full" value={selectedOption.value} onChange={setSelectedDisease}
            options={selectableOptions.map(({ value, label }) => ({ value, label }))} /></label>}
        <XrayViewer originalImage={originalImage} attentionImage={selected?.heatmap}
          disease={selectedOption && diseaseLabel(selectedOption.value)} selectedDisease={selectedOption?.value}
          detections={analysis.yolo_detections} hybridBoxes={analysis.hybrid_boxes} detectorStatus={inferenceMeta?.detectorStatus} />
        {selected && <div className="rounded-lg bg-slate-50 dark:bg-slate-800 p-3 text-sm leading-relaxed">
          <strong>Vị trí ước tính từ bản đồ chú ý: </strong>{selected.lesion_location}
        </div>}
        <details className="disclosure technical-details"><summary>Thông số phân tích mở rộng</summary>
          <div className="space-y-3 mt-3 text-sm">
            {selected ? <dl className="metric-list">
              <div><dt>Dấu hiệu</dt><dd>{diseaseLabel(selected.disease)}</dd></div>
              <div><dt>Diện tích vùng chú ý</dt><dd>{formatNumber(selected.lesion_area_percent, 2)}%</dd></div>
              <div><dt>Số vùng từ bản đồ chú ý</dt><dd>{selected.bounding_boxes.length}</dd></div>
              {selected.calibrated_confidence != null && <div><dt>Điểm mô hình cung cấp</dt><dd>{formatPercent(selected.calibrated_confidence)}</dd></div>}
              {selected.uncertainty != null && <div><dt>Độ bất định mô hình cung cấp</dt><dd>{formatNumber(selected.uncertainty, 4)}</dd></div>}
            </dl> : <p>Không có nhãn bệnh vượt ngưỡng hiển thị.</p>}
            {report.findings.filter(f => f.disease === selected?.disease).map(f => <p key={f.disease}>{diseaseLabel(f.disease)}{f.severity_vi ? ` · Mức độ ${f.severity_vi.toLowerCase()}` : ''}</p>)}
            <h3 className="font-semibold">Bộ định vị bổ sung (YOLO11m)</h3>
            {analysis.yolo_detections.length ? analysis.yolo_detections.map((d, i) => <div key={i}>
              <p>{diseaseLabel(d.disease)} · Điểm định vị {formatPercent(d.confidence)} · Điểm kết hợp {formatPercent(d.cascade_score)}</p>
              {d.vindr_class && <p>Lớp dữ liệu VinDr: {diseaseLabel(d.vindr_class)}</p>}
              {d.bbox && <p>Tọa độ vùng: [{d.bbox.x1}, {d.bbox.y1}, {d.bbox.x2}, {d.bbox.y2}]</p>}
            </div>) : <p>{inferenceMeta?.detectorStatus === 'unavailable' ? 'Chưa nạp trọng số YOLO11m cho lần phân tích này.' : inferenceMeta?.detectorStatus === 'loaded-unconfirmed' ? 'Đã nạp YOLO11m và đã quét ảnh, nhưng không phát hiện vùng tổn thương khu trú vượt ngưỡng (YOLO hỗ trợ 14 lớp VinDr).' : 'Chưa ghi nhận trạng thái bộ định vị cho lần chụp này.'}</p>}
            {(analysis.hybrid_boxes?.length ?? 0) > 0 && <>
              <h3 className="font-semibold">Khung định vị từ Hybrid</h3>
              {analysis.hybrid_boxes!.map((b, i) => <div key={i}>
                <p>{diseaseLabel(b.disease)} · Xác suất {formatPercent(b.probability)} · Điểm vùng {formatPercent(b.confidence)}</p>
                {b.bbox && <p>Tọa độ vùng: [{b.bbox.x1}, {b.bbox.y1}, {b.bbox.x2}, {b.bbox.y2}]</p>}
              </div>)}
            </>}
            <a href="/legacy.html" target="_blank" rel="noreferrer" className="text-primary underline">Mở công cụ nghiên cứu nâng cao ↗</a>
          </div>
        </details>
      </div> },
      { key: 'probabilities', label: 'Điểm AI', children: <div className="space-y-4">
        <p className="text-sm text-secondary dark:text-gray-400">{hasPathologyFindings ? `Hiển thị ${displayedProbabilities.length}/15 nhãn bệnh (đã ẩn nhãn Không phát hiện bất thường vì ảnh có bệnh vượt ngưỡng). Mỗi điểm thể hiện mức dự đoán cho một nhãn; các điểm không cộng thành 100%.` : 'Đầy đủ 15 nhãn của mô hình. Mỗi điểm thể hiện mức dự đoán cho một nhãn; các điểm không cộng thành 100%.'}</p>
        <ClassificationDetails classification={inferenceMeta?.classification} />
        <div className="grid gap-4 sm:grid-cols-2">{displayedProbabilities.map(p => <div key={p.disease}>
          <div className="flex justify-between gap-3 text-sm"><span>{diseaseLabel(p.disease)}</span><strong>{formatPercent(p.probability)}</strong></div>
          <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded mt-2 overflow-hidden" role="meter" aria-label={diseaseLabel(p.disease)} aria-valuemin={0} aria-valuemax={100} aria-valuenow={p.probability * 100}>
            <div className="h-full bg-blue-600 rounded" style={{ width: `${p.probability * 100}%` }} /></div>
          {inferenceMeta?.classification?.decisions.filter(item => item.disease === p.disease).map(item =>
            <p key={item.disease} className="text-xs text-secondary mt-1">Ngưỡng {formatPercent(item.threshold)} · {item.aboveThreshold ? 'Vượt ngưỡng' : 'Dưới ngưỡng'}</p>)}
        </div>)}</div>
      </div> },
      { key: 'report', label: 'Báo cáo', children: <div className="space-y-4 text-sm leading-relaxed">
        <InferenceDetails meta={inferenceMeta} />
        <ClassificationDetails classification={inferenceMeta?.classification} showFindings />
        {Boolean(inferenceMeta?.classification?.withoutHeatmap.length) && <p className="text-xs text-secondary">Mô tả dưới đây thuộc các nhãn có bản đồ chú ý, không thay thế danh sách đầy đủ phía trên.</p>}
        <h3 className="section-title">{inferenceMeta?.classification ? 'Mô tả từ các bản đồ chú ý' : 'Tóm tắt do AI đề xuất'}</h3><p>{report.summary}</p>
        {report.findings.map(f => <p key={f.disease}>{f.finding_text}</p>)}
        {report.recommendations.length > 0 && <><h3 className="font-semibold">Khuyến nghị do AI đề xuất</h3>
          <ul className="list-disc pl-5 space-y-2">{report.recommendations.map((r, i) => <li key={i}>{r}</li>)}</ul></>}
        {report.warnings.map((warning, i) => <Tag key={i} color="orange" className="whitespace-normal">{warning}</Tag>)}
        {analysis.co_occurrences.length > 0 && <details className="disclosure"><summary>Các dấu hiệu thường xuất hiện cùng nhau</summary>
          <div className="space-y-3 mt-3">{analysis.co_occurrences.map((c, i) => <p key={i}>{c.diseases.map(diseaseLabel).join(' + ')} · {c.clinical_note}</p>)}</div></details>}
      </div> },
      { key: 'similar', label: 'Ca tương tự', children: <div className="space-y-4 text-sm">
        <p className="text-secondary dark:text-gray-400">Các ảnh gần giống trong kho ca tham chiếu, dùng để đối chiếu kết quả.</p>
        {loadingCases && <p role="status">Đang tải ca tham chiếu…</p>}
        {similar.length ? <ul className="space-y-3">{similar.map((c, i) => <li key={i} className="rounded-lg border border-outline-variant/40 p-3 space-y-2 break-words">
          <p className="font-semibold">Ca tham chiếu {i + 1} · Mức tương đồng {formatNumber(c.similarity)}%</p>
          <p>{c.diseases.map(diseaseLabel).join(', ')}</p>
          {'sourceCase' in c && c.sourceCase && <Link className="block text-primary underline"
            to={`/patient-profile/${c.sourceCase.patientCode}?scan=${c.sourceCase.scanId}`}>Mở lần chụp tham chiếu</Link>}
          <details className="disclosure text-xs"><summary>Thông tin ca</summary><p className="mt-2">{c.filename}</p></details>
        </li>)}</ul> : <Empty description="Chưa có ca tham chiếu phù hợp" />}
      </div> },
    ]} />
  </section>;
}
