/** Chỉ đổi nhãn hiển thị; giữ nguyên khóa lớp, giá trị và hợp đồng API. */
export const DISEASE_LABELS: Record<string, string> = {
  'No Finding': 'Không phát hiện bất thường',
  Atelectasis: 'Xẹp phổi', Cardiomegaly: 'Phì đại bóng tim (Tim to)', Effusion: 'Tràn dịch màng phổi',
  Infiltration: 'Thâm nhiễm nhu mô phổi', Mass: 'Khối u phổi (> 3cm)', Nodule: 'Nốt mờ phổi (≤ 3cm)',
  Pneumonia: 'Viêm phổi', Pneumothorax: 'Tràn khí màng phổi', Consolidation: 'Đông đặc phế nang',
  Edema: 'Phù phổi cấp', Emphysema: 'Khí phế thũng', Fibrosis: 'Xơ hóa phổi',
  Pleural_Thickening: 'Dày màng phổi', Hernia: 'Thoát vị hoành',
};
const DETECTOR_LABELS: Record<string, string> = {
  'aortic enlargement': 'Giãn động mạch chủ',
  calcification: 'Vôi hóa',
  ild: 'Bệnh phổi mô kẽ',
  'lung opacity': 'Đám mờ phổi',
  'nodule/mass': 'Nốt/khối mờ phổi',
  'nodule mass': 'Nốt/khối mờ phổi',
  'other lesion': 'Tổn thương khác',
  'pleural effusion': DISEASE_LABELS.Effusion,
  'pleural thickening': DISEASE_LABELS.Pleural_Thickening,
  'pulmonary fibrosis': DISEASE_LABELS.Fibrosis,
};
const formatBilingual = (vi: string | undefined, raw: string) => {
  const en = raw.replaceAll('_', ' ');
  if (!vi) return en;
  return vi.toLowerCase() === en.toLowerCase() ? vi : `${vi} (${en})`;
};
export const diseaseLabel = (value: string) => {
  const vi = DISEASE_LABELS[value] || DETECTOR_LABELS[value.replaceAll('_', ' ').toLowerCase()];
  return formatBilingual(vi, value);
};
export const formatNumber = (value: number, digits = 1) =>
  new Intl.NumberFormat('vi-VN', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
export const formatPercent = (value: number) => `${formatNumber(value * 100)}%`;
export const deviceLabel = (value?: string) => value === 'cpu' ? 'Bộ xử lý CPU' :
  value?.startsWith('cuda') ? 'Bộ xử lý đồ họa GPU' : value || 'Chưa ghi nhận';
export const scanTypeLabel = (value?: string) => value?.replace('X-Quang', 'X-quang').replace('Nghiêng', 'nghiêng') || 'Chưa có phim chụp';
const DIAGNOSIS_STATUS_LABELS: Record<string, string> = {
  pending: 'Chờ xác minh', 'Đang chờ': 'Chờ xác minh',
  verified: 'Đã xác minh', 'Đã xác minh': 'Đã xác minh',
  flagged: 'Cần xem lại', 'Đánh dấu': 'Cần xem lại',
};
export const diagnosisStatusLabel = (value?: string) => DIAGNOSIS_STATUS_LABELS[value || 'pending'] || value;
export const scanChoiceLabel = (scan: { id: string; date: string; time: string; type: string }) =>
  `Phim ${scan.id.slice(-6).toUpperCase()} · ${scan.date} ${scan.time} · ${scanTypeLabel(scan.type)}`;
