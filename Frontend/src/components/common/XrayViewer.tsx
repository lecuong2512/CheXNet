import { useState } from 'react';
import { Button, Segmented } from 'antd';
import ScanImage from './ScanImage';
import { diseaseLabel, formatNumber, formatPercent } from '../../utils/display';
import { filterBoxesByDisease, hybridFallbackBoxes, yoloBoxes } from '../../utils/yoloBoxes';
import type { HybridBox, InferenceMeta, YoloDetection } from '../../types/hybrid';

export default function XrayViewer({ originalImage, attentionImage, disease, selectedDisease, detections = [], hybridBoxes = [], detectorStatus }: {
  originalImage?: string; attentionImage?: string; disease?: string; selectedDisease?: string;
  detections?: YoloDetection[]; hybridBoxes?: HybridBox[]; detectorStatus?: InferenceMeta['detectorStatus'];
}) {
  const hasYoloDetections = yoloBoxes(detections, Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER).length > 0;
  const hasHybridDetections = hybridFallbackBoxes(hybridBoxes, Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER).length > 0;
  const hasDetections = hasYoloDetections || hasHybridDetections;
  const [view, setView] = useState(hasDetections ? 'yolo' : 'original');
  const [imageSize, setImageSize] = useState<{ source: string; width: number; height: number }>();
  const [zoom, setZoom] = useState(1);
  const [contrast, setContrast] = useState(false);
  const [inverted, setInverted] = useState(false);
  const activeView = view === 'attention' && !attentionImage ? (hasDetections ? 'yolo' : 'original') : view;
  const source = activeView === 'attention' && attentionImage ? attentionImage : originalImage;
  const size = imageSize?.source === source ? imageSize : undefined;
  const yoloList = size && activeView === 'yolo'
    ? filterBoxesByDisease(yoloBoxes(detections, size.width, size.height), selectedDisease)
    : [];
  const hybridList = size && activeView === 'yolo'
    ? filterBoxesByDisease(hybridFallbackBoxes(hybridBoxes, size.width, size.height), selectedDisease)
    : [];
  const boxTabLabel = hasYoloDetections && hasHybridDetections ? 'Khung YOLO + Hybrid' :
    hasHybridDetections ? 'Khung từ Hybrid' : 'Khung YOLO';
  const renderBoxes = [
    ...yoloList.map((item, idx) => ({
      key: `yolo-${idx}-${item.disease}`,
      bbox: item.bbox,
      stroke: '#58d6a4',
      textColor: '#102c24',
      dashed: false,
      title: `${diseaseLabel(item.vindr_class || item.disease)} · Điểm YOLO ${formatPercent(item.confidence)}`,
      caption: `${diseaseLabel(item.vindr_class || item.disease)} · Điểm YOLO ${formatPercent(item.confidence)} · Điểm kết hợp ${formatPercent(item.cascade_score)}`,
    })),
    ...hybridList.map((item, idx) => ({
      key: `hybrid-${idx}-${item.disease}`,
      bbox: item.bbox,
      stroke: '#f59e0b',
      textColor: '#1f1300',
      dashed: true,
      title: `${diseaseLabel(item.disease)} · Xác suất ${formatPercent(item.probability)}`,
      caption: `${diseaseLabel(item.disease)} · Xác suất ${formatPercent(item.probability)} · Độ tập trung vùng ${formatPercent(item.confidence)}`,
    })),
  ];
  const markerSize = size ? Math.max(size.width, size.height) / 70 : 0;
  const reset = () => { setZoom(1); setContrast(false); setInverted(false); };
  return <div className="space-y-3 min-w-0">
    <div className="viewer-controls">
      <Segmented aria-label="Chế độ xem ảnh" value={activeView} onChange={value => { setView(value); reset(); }}
        options={[{ value: 'original', label: 'Ảnh gốc', disabled: !originalImage },
          { value: 'yolo', label: boxTabLabel, disabled: !originalImage || !hasDetections },
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
    <div className="xray-canvas" role="region" aria-label={activeView === 'attention' ? 'Bản đồ chú ý của mô hình' : activeView === 'yolo' ? 'Ảnh X-quang với khung định vị' : 'Ảnh X-quang gốc'}>
      {source ? <div className="absolute inset-0" style={{ transform: `scale(${zoom})`, transition: 'transform .15s' }}>
        <ScanImage key={source} src={source} alt={activeView === 'attention' ? `Bản đồ chú ý: ${disease || 'dấu hiệu được chọn'}` : 'Ảnh X-quang ngực gốc'}
          onLoad={event => setImageSize({ source, width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
          onError={() => setImageSize(undefined)}
          style={{ filter: `contrast(${contrast ? 1.8 : 1}) invert(${inverted ? 1 : 0})` }} />
        {size && renderBoxes.length > 0 && <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${size.width} ${size.height}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={`${renderBoxes.length} khung định vị`}>
          {renderBoxes.map((item, index) => <g key={item.key}>
            <title>{index + 1}. {item.title}</title>
            <rect x={item.bbox.x1} y={item.bbox.y1} width={item.bbox.x2 - item.bbox.x1} height={item.bbox.y2 - item.bbox.y1}
              fill="none" stroke={item.stroke} strokeWidth="2" strokeDasharray={item.dashed ? '6 4' : undefined} vectorEffect="non-scaling-stroke" />
            <circle cx={Math.min(size.width - markerSize, item.bbox.x1 + markerSize)} cy={Math.min(size.height - markerSize, item.bbox.y1 + markerSize)} r={markerSize} fill={item.stroke} />
            <text x={Math.min(size.width - markerSize, item.bbox.x1 + markerSize)} y={Math.min(size.height - markerSize, item.bbox.y1 + markerSize)} textAnchor="middle" dominantBaseline="central"
              fill={item.textColor} fontSize={markerSize * 1.25} fontWeight="700">{index + 1}</text>
          </g>)}
        </svg>}
      </div> :
        <p>Chưa có ảnh để hiển thị.</p>}
    </div>
    {activeView !== 'yolo' && <p className="text-xs text-secondary dark:text-gray-400 leading-relaxed">
      {activeView === 'attention'
        ? 'Màu trên ảnh thể hiện vùng mô hình chú ý khi dự đoán dấu hiệu được chọn.'
        : 'Ảnh gốc của lần chụp. Chọn “Bản đồ chú ý” để xem vùng mô hình tập trung.'}
    </p>}
    {activeView === 'yolo' && renderBoxes.length > 0 && <ol className="space-y-2 text-xs leading-relaxed" aria-label="Chú giải khung định vị">
      {renderBoxes.map((item, index) => <li key={item.key}><strong>{index + 1}.</strong> {item.caption}</li>)}
    </ol>}
    {activeView === 'yolo' && size && hasDetections && renderBoxes.length === 0 && <p role="note" className="text-xs text-secondary">
      {selectedDisease ? `Chưa có khung định vị cho "${disease || diseaseLabel(selectedDisease)}" trên ảnh này.` : 'Không có tọa độ khung hợp lệ trong phạm vi ảnh này.'}
    </p>}
    {!hasDetections && <p className="text-xs text-secondary dark:text-gray-400" role="note">
      {detectorStatus === 'unavailable' ? 'Chưa nạp mô hình định vị YOLO11m cho lần chụp này.' :
        detectorStatus === 'loaded-unconfirmed' ? 'Đã nạp và chạy mô hình YOLO11m, nhưng chưa phát hiện vùng tổn thương khu trú vượt ngưỡng trên ảnh này (YOLO hỗ trợ 14 lớp VinDr). Điều này không xác nhận ảnh bình thường.' :
        'Lần chụp này chưa ghi nhận khung định vị YOLO.'}
    </p>}
  </div>;
}
