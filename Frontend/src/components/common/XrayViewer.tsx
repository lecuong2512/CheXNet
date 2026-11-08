import { useState } from 'react';
import { Button, Segmented } from 'antd';
import ScanImage from './ScanImage';
import { formatNumber } from '../../utils/display';

export default function XrayViewer({ originalImage, attentionImage, disease }: {
  originalImage?: string; attentionImage?: string; disease?: string;
}) {
  const [view, setView] = useState('original');
  const [zoom, setZoom] = useState(1);
  const [contrast, setContrast] = useState(false);
  const [inverted, setInverted] = useState(false);
  const source = view === 'attention' && attentionImage ? attentionImage : originalImage;
  const reset = () => { setZoom(1); setContrast(false); setInverted(false); };
  return <div className="space-y-3 min-w-0">
    <div className="viewer-controls">
      <Segmented aria-label="Chế độ xem ảnh" value={view} onChange={value => { setView(value); reset(); }}
        options={[{ value: 'original', label: 'Ảnh gốc', disabled: !originalImage },
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
    <div className="xray-canvas" role="region" aria-label={view === 'attention' ? 'Bản đồ chú ý của mô hình' : 'Ảnh X-quang gốc'}>
      {source ? <ScanImage src={source} alt={view === 'attention' ? `Bản đồ chú ý: ${disease || 'dấu hiệu được chọn'}` : 'Ảnh X-quang ngực gốc'}
        style={{ transform: `scale(${zoom})`, filter: `contrast(${contrast ? 1.8 : 1}) invert(${inverted ? 1 : 0})` }} /> :
        <p>Chưa có ảnh để hiển thị.</p>}
    </div>
    <p className="text-xs text-secondary dark:text-gray-400 leading-relaxed">
      {view === 'attention' ? 'Màu trên ảnh thể hiện vùng mô hình chú ý khi dự đoán dấu hiệu được chọn.' : 'Ảnh gốc của lần chụp. Chọn “Bản đồ chú ý” để xem vùng mô hình tập trung.'}
    </p>
  </div>;
}
