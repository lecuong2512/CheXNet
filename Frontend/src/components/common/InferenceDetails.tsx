import type { InferenceMeta } from '../../types/hybrid';
import { deviceLabel, formatNumber } from '../../utils/display';

export default function InferenceDetails({ meta }: { meta?: InferenceMeta }) {
  const trace = meta?.trace;
  if (!trace) return <p className="text-xs text-secondary">Lần phân tích cũ chưa lưu mã đối chiếu thực thi model.</p>;
  return <details className="disclosure technical-details text-sm">
    <summary>Thông tin lần phân tích</summary>
    <dl className="metric-list mt-3">
      <div><dt>Mã đối chiếu</dt><dd className="break-all font-mono text-xs" data-testid="inference-trace-id">{trace.id}</dd></div>
      <div><dt>Hoàn tất</dt><dd>{new Date(trace.completedAt).toLocaleString('vi-VN', { timeZone: 'Asia/Ho_Chi_Minh' })}</dd></div>
      <div><dt>Thiết bị</dt><dd>{deviceLabel(meta?.device)}</dd></div>
      {(['hybrid', 'yolo'] as const).map(key => <div key={key}>
        <dt>{key === 'hybrid' ? 'Hybrid CNN–ViT' : 'YOLO'}</dt>
        <dd>{trace.models[key].observed ? `Đã ghi nhận thực thi · ${trace.models[key].forwardCalls} lượt` : 'Chưa ghi nhận thực thi'}</dd>
      </div>)}
      <div><dt>Thời gian xử lý</dt><dd>{formatNumber(trace.durationMs / 1000, 2)} giây</dd></div>
    </dl>
    <p className="text-xs text-secondary mt-3">Dùng mã đối chiếu để tìm lượt xử lý tương ứng trong log máy chủ. Thông tin này xác nhận thực thi, không đánh giá độ chính xác chẩn đoán.</p>
  </details>;
}
