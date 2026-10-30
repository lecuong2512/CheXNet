// sampleData.js - Preset sample Chest X-rays for quick testing

export const SAMPLE_XRAYS = [
  {
    id: 'sample-1',
    title: 'Viêm phổi (Pneumonia)',
    subtitle: 'Nhiễm trùng thùy dưới phổi phải',
    severity: 'moderate',
    svgColor: '#ff7040',
    description: 'X-quang ngực tư thế PA hiển thị vùng thâm nhiễm đám mờ đông đặc ở thùy dưới phổi phải.',
    // Data URL SVG placeholder for sample chest x-ray image
    dataUrl: 'data:image/svg+xml;utf8,' + encodeURIComponent(`
      <svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
        <rect width="512" height="512" fill="#040a14"/>
        <!-- Spine & Ribs -->
        <line x1="256" y1="40" x2="256" y2="480" stroke="#2a3a4e" stroke-width="12" stroke-dasharray="8 6"/>
        <!-- Left Lung -->
        <path d="M 230 100 C 160 110, 110 160, 100 240 C 90 320, 120 400, 220 420 C 230 400, 230 200, 230 100 Z" fill="#122033" stroke="#2e425a" stroke-width="3"/>
        <!-- Right Lung with pneumonia lesion -->
        <path d="M 282 100 C 352 110, 402 160, 412 240 C 422 320, 392 400, 292 420 C 282 400, 282 200, 282 100 Z" fill="#122033" stroke="#2e425a" stroke-width="3"/>
        <!-- Infiltration lesion on right lower lobe -->
        <ellipse cx="340" cy="310" rx="45" ry="35" fill="rgba(240, 240, 255, 0.45)" filter="blur(8px)"/>
        <ellipse cx="325" cy="325" rx="30" ry="25" fill="rgba(255, 255, 255, 0.6)" filter="blur(5px)"/>
        <!-- Heart contour -->
        <path d="M 240 260 C 210 270, 190 310, 230 370 C 250 380, 260 350, 250 300 Z" fill="#1a2b40" stroke="#384f6c" stroke-width="2"/>
        <!-- Collarbones -->
        <path d="M 120 80 Q 200 110 256 120 Q 312 110 392 80" fill="none" stroke="#3a4f6a" stroke-width="6"/>
      </svg>
    `)
  },
  {
    id: 'sample-2',
    title: 'Phổi bình thường (Normal)',
    subtitle: 'Không phát hiện tổn thương',
    severity: 'normal',
    svgColor: '#00e5a0',
    description: 'X-quang ngực chuẩn PA, hai phế trường sáng đều, góc màng phổi nhọn, tim kích thước bình thường.',
    dataUrl: 'data:image/svg+xml;utf8,' + encodeURIComponent(`
      <svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
        <rect width="512" height="512" fill="#040a14"/>
        <line x1="256" y1="40" x2="256" y2="480" stroke="#2a3a4e" stroke-width="12" stroke-dasharray="8 6"/>
        <path d="M 230 100 C 160 110, 110 160, 100 240 C 90 320, 120 400, 220 420 C 230 400, 230 200, 230 100 Z" fill="#0d1928" stroke="#2e425a" stroke-width="3"/>
        <path d="M 282 100 C 352 110, 402 160, 412 240 C 422 320, 392 400, 292 420 C 282 400, 282 200, 282 100 Z" fill="#0d1928" stroke="#2e425a" stroke-width="3"/>
        <path d="M 240 260 C 210 270, 190 310, 230 370 C 250 380, 260 350, 250 300 Z" fill="#162536" stroke="#384f6c" stroke-width="2"/>
        <path d="M 120 80 Q 200 110 256 120 Q 312 110 392 80" fill="none" stroke="#3a4f6a" stroke-width="6"/>
      </svg>
    `)
  },
  {
    id: 'sample-3',
    title: 'Tràn dịch màng phổi (Effusion)',
    subtitle: 'Tù góc sườn hoành phải',
    severity: 'severe',
    svgColor: '#ff4560',
    description: 'X-quang ngực cho thấy tù góc sườn hoành phải do đọng dịch màng phổi.',
    dataUrl: 'data:image/svg+xml;utf8,' + encodeURIComponent(`
      <svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
        <rect width="512" height="512" fill="#040a14"/>
        <line x1="256" y1="40" x2="256" y2="480" stroke="#2a3a4e" stroke-width="12" stroke-dasharray="8 6"/>
        <path d="M 230 100 C 160 110, 110 160, 100 240 C 90 320, 120 400, 220 420 C 230 400, 230 200, 230 100 Z" fill="#122033" stroke="#2e425a" stroke-width="3"/>
        <path d="M 282 100 C 352 110, 402 160, 412 240 C 422 320, 392 400, 292 420 C 282 400, 282 200, 282 100 Z" fill="#122033" stroke="#2e425a" stroke-width="3"/>
        <!-- Fluid opacity at bottom right costophrenic angle -->
        <path d="M 282 350 Q 350 340 410 370 L 412 410 L 292 420 Z" fill="rgba(255, 255, 255, 0.75)" filter="blur(4px)"/>
        <path d="M 240 260 C 210 270, 190 310, 230 370 C 250 380, 260 350, 250 300 Z" fill="#1a2b40" stroke="#384f6c" stroke-width="2"/>
        <path d="M 120 80 Q 200 110 256 120 Q 312 110 392 80" fill="none" stroke="#3a4f6a" stroke-width="6"/>
      </svg>
    `)
  }
];

export const ALL_DISEASE_LABELS = [
  { key: 'Pneumonia', nameVN: 'Viêm phổi', desc: 'Nhiễm trùng nhu mô phế nang' },
  { key: 'Effusion', nameVN: 'Tràn dịch màng phổi', desc: 'Tích tụ dịch màng phổi' },
  { key: 'Atelectasis', nameVN: 'Xẹp phổi', desc: 'Giảm thể tích nhu mô phổi' },
  { key: 'Cardiomegaly', nameVN: 'Bóng tim to', desc: 'Chỉ số tim - ngực > 0.5' },
  { key: 'Infiltration', nameVN: 'Thâm nhiễm', desc: 'Đám mờ nhu mô lan tỏa' },
  { key: 'Mass', nameVN: 'Khối u', desc: 'Tổn thương dạng khối > 3cm' },
  { key: 'Nodule', nameVN: 'Nốt mờ', desc: 'Tổn thương dạng nốt ≤ 3cm' },
  { key: 'Pneumothorax', nameVN: 'Tràn khí màng phổi', desc: 'Khí trong khoang màng phổi' },
  { key: 'Consolidation', nameVN: 'Đông đặc', desc: 'Lấp đầy phế nang bởi dịch/mủ' },
  { key: 'Edema', nameVN: 'Phù phổi', desc: 'Tích tụ dịch trong kẽ & phế nang' },
  { key: 'Emphysema', nameVN: 'Khí phế thũng', desc: 'Tăng sáng nhu mô phế nang' },
  { key: 'Fibrosis', nameVN: 'Xơ phổi', desc: 'Tổn thương xơ hóa dải mờ' },
  { key: 'Pleural_Thickening', nameVN: 'Dày màng phổi', desc: 'Tăng độ dày màng phổi' },
  { key: 'Hernia', nameVN: 'Thoát vị hoành', desc: 'Thoát vị cơ hoành' },
  { key: 'No Finding', nameVN: 'Không tổn thương', desc: 'Hình ảnh X-quang bình thường' }
];
