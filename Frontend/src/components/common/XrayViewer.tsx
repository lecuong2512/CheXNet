import { useState } from 'react';
import { Button, Segmented } from 'antd';
import ScanImage from './ScanImage';
import { diseaseLabel, formatNumber, formatPercent } from '../../utils/display';
import { yoloBoxes } from '../../utils/yoloBoxes';
import type { InferenceMeta, YoloDetection } from '../../types/hybrid';

export default function XrayViewer({ originalImage, attentionImage, disease, detections = [], detectorStatus }: {
  originalImage?: string; attentionImage?: string; disease?: string;
  detections?: YoloDetection[]; detectorStatus?: InferenceMeta['detectorStatus'];
}) {
  const hasDetections = yoloBoxes(detections, Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER).length > 0;
  const [view, setView] = useState(hasDetections ? 'yolo' : 'original');
  const [imageSize, setImageSize] = useState<{ source: string; width: number; height: number }>();
  const [zoom, setZoom] = useState(1);
  const [contrast, setContrast] = useState(false);
  const [inverted, setInverted] = useState(false);
  const source = view === 'attention' && attentionImage ? attentionImage : originalImage;
  const size = imageSize?.source === source ? imageSize : undefined;
  const boxes = size && view === 'yolo' ? yoloBoxes(detections, size.width, size.height) : [];
  const markerSize = size ? Math.max(size.width, size.height) / 70 : 0;
  const reset = () => { setZoom(1); setContrast(false); setInverted(false); };
  return <div className="space-y-3 min-w-0">
    <div className="viewer-controls">
      <Segmented aria-label="Chế độ xem ảnh" value={view} onChange={value => { setView(value); reset(); }}
        options={[{ value: 'original', label: 'Ảnh gốc', disabled: !originalImage },
          { value: 'yolo', label: 'Khung YOLO', disabled: !originalImage || !hasDetections },
          { value: 'attention', label: 'Bản đồ chú ý', disabled: !attentionImage }]} />
      <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Điều chỉnh ảnh">
        <Button aria-label="Thu nhỏ ảnh" title="Thu nhỏ ảnh" disabled={zoom <= 1} onClick={() => setZoom(z => Math.max(1, z - .25))}>−</Button>
        <span className="text-xs min-w-12 text-center" aria-live="polite">{formatNumber(zoom * 100, 0)}%</span>
        <Button aria-label="Phóng to ảnh" title="Phóng to ảnh" disabled={zoom >= 3} onClick={() => setZoom(z => Math.min(3, z + .25))}>+</Button>
        <Button aria-pressed={contrast} onClick={() => setContrast(v => !v)}>Tương phản</Button>
        <Button aria-pressed={inverted} onClick={() => setInverted(v => !v)}>Đảo màu</Button>
        <Button onClick={reset}>Đặt lại</Button>
      </div>
    </div>
    <div className="xray-canvas" role="region" aria-label={view === 'attention' ? 'Bản đồ chú ý của mô hình' : view === 'yolo' ? 'Ảnh X-quang với khung định vị YOLO' : 'Ảnh X-quang gốc'}>
      {source ? <div className="absolute inset-0" style={{ transform: `scale(${zoom})`, transition: 'transform .15s' }}>
        <ScanImage key={source} src={source} alt={view === 'attention' ? `Bản đồ chú ý: ${disease || 'dấu hiệu được chọn'}` : 'Ảnh X-quang ngực gốc'}
          onLoad={event => setImageSize({ source, width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
          onError={() => setImageSize(undefined)}
          style={{ filter: `contrast(${contrast ? 1.8 : 1}) invert(${inverted ? 1 : 0})` }} />
        {size && boxes.length > 0 && <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${size.width} ${size.height}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={`${boxes.length} khung định vị YOLO`}>
          {boxes.map((item, index) => <g key={`${item.disease}-${index}`}>
            <title>{index + 1}. {diseaseLabel(item.vindr_class || item.disease)} · Điểm YOLO {formatPercent(item.confidence)}</title>
            <rect x={item.bbox.x1} y={item.bbox.y1} width={item.bbox.x2 - item.bbox.x1} height={item.bbox.y2 - item.bbox.y1}
              fill="none" stroke="#58d6a4" strokeWidth="2" vectorEffect="non-scaling-stroke" />
            <circle cx={Math.min(size.width - markerSize, item.bbox.x1 + markerSize)} cy={Math.min(size.height - markerSize, item.bbox.y1 + markerSize)} r={markerSize} fill="#58d6a4" />
            <text x={Math.min(size.width - markerSize, item.bbox.x1 + markerSize)} y={Math.min(size.height - markerSize, item.bbox.y1 + markerSize)} textAnchor="middle" dominantBaseline="central"
              fill="#102c24" fontSize={markerSize * 1.25} fontWeight="700">{index + 1}</text>
          </g>)}
        </svg>}
      </div> :
        <p>Chưa có ảnh để hiển thị.</p>}
    </div>
    <p className="text-xs text-secondary dark:text-gray-400 leading-relaxed">
      {view === 'attention' ? 'Màu trên ảnh thể hiện vùng mô hình chú ý khi dự đoán dấu hiệu được chọn.' : view === 'yolo' ? 'Khung xanh là vùng định vị do YOLO trả về, sau bước lọc kết hợp với bộ phân loại.' : 'Ảnh gốc của lần chụp. Chọn “Bản đồ chú ý” để xem vùng mô hình tập trung.'}
    </p>
    {view === 'yolo' && boxes.length > 0 && <ol className="space-y-2 text-xs leading-relaxed" aria-label="Chú giải khung YOLO">
      {boxes.map((item, index) => <li key={`${item.disease}-${index}`}><strong>{index + 1}. {diseaseLabel(item.vindr_class || item.disease)}</strong> · Điểm YOLO {formatPercent(item.confidence)} · Điểm kết hợp {formatPercent(item.cascade_score)}</li>)}
    </ol>}
    {view === 'yolo' && size && hasDetections && boxes.length === 0 && <p role="note" className="text-xs text-secondary">Không có tọa độ khung hợp lệ trong phạm vi ảnh này.</p>}
    {!hasDetections && <p className="text-xs text-secondary dark:text-gray-400" role="note">
      {detectorStatus === 'unavailable' ? 'Chưa nạp mô hình định vị YOLO11m cho lần chụp này nên chưa có khung bao.' : detectorStatus === 'loaded-unconfirmed' ? 'Đã nạp và chạy mô hình YOLO11m, nhưng chưa phát hiện vùng tổn thương khu trú vượt ngưỡng trên ảnh này (YOLO hỗ trợ 14 lớp VinDr). Điều này không xác nhận ảnh bình thường.' : 'Lần chụp này chưa ghi nhận khung định vị YOLO.'}
    </p>}
  </div>;
}
