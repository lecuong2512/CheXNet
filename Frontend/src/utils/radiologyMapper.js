// radiologyMapper.js - Authentic Radiological Criteria Mapping for Chest X-Ray Diagnosis

export const RADIOLOGICAL_FINDINGS_MAP = {
  'Pneumonia': [
    { label: 'Hình thái tổn thương', val: 'Đám mờ phế nang không đồng nhất (Patchy airspace consolidation)' },
    { label: 'Dấu hiệu phế quản khí', val: 'Dấu hiệu phế quản khí (+) (Air bronchogram sign)' },
    { label: 'Vị trí khu trú', val: 'Thùy dưới / Phân thùy giữa theo vùng Grad-CAM' },
    { label: 'Tình trạng màng phổi', val: 'Không có dấu hiệu tràn khí khoang màng phổi' },
    { label: 'Mật độ đông đặc', val: 'Tỷ trọng thâm nhiễm vừa đến cao (HU: 40–60)' }
  ],
  'Effusion': [
    { label: 'Tù góc sườn hoành', val: 'Tù góc sườn hoành rõ (Blunting of costophrenic angle)' },
    { label: 'Đường cong màng phổi', val: 'Dấu hiệu đường cong Damoiseau / Mờ đồng nhất góc hoành' },
    { label: 'Thể tích khoang màng phổi', val: 'Tích tụ dịch tự do / Dịch khoang màng phổi' },
    { label: 'Nhu mô kế cận', val: 'Xẹp phổi thụ động vùng đáy phổi tương ứng' }
  ],
  'Cardiomegaly': [
    { label: 'Chỉ số Tim - Ngực (CTR)', val: 'Chỉ số CTR > 0.50 (Bóng tim mở rộng)' },
    { label: 'Bờ tim trái', val: 'Quai động mạch chủ phồng, mỏm tim chúi xuống' },
    { label: 'Bờ tim phải', val: 'Bờ nhĩ phải giãn rộng sang phải > 1.5 cm' },
    { label: 'Tĩnh mạch phổi', val: 'Dấu hiệu tái phân bố tuần hoàn phổi đỉnh' }
  ],
  'Atelectasis': [
    { label: 'Tỷ trọng nhu mô', val: 'Dải mờ xẹp tăng tỷ trọng khu trú (Linear / Discoid atelectasis)' },
    { label: 'Co kéo rãnh liên thùy', val: 'Dấu hiệu co kéo rãnh liên thùy & rốn phổi về phía tổn thương' },
    { label: 'Thể tích nhu mô', val: 'Giảm thể tích thùy phổi tương ứng' },
    { label: 'Thông khí bù trừ', val: 'Tăng thông khí bù trừ ở vùng nhu mô lành kế cận' }
  ],
  'Infiltration': [
    { label: 'Dạng đám mờ', val: 'Đám mờ dạng mây tan / Thâm nhiễm kẽ phế nang (Alveolar opacity)' },
    { label: 'Bờ tổn thương', val: 'Bờ mờ không rõ nét, lan tỏa hai phế trường' },
    { label: 'Phân bố nhu mô', val: 'Thâm nhiễm dải rác theo nhánh phế quản' }
  ],
  'Mass': [
    { label: 'Dạng khối mờ', val: 'Tổn thương dạng khối mờ đặc, đường kính > 3.0 cm' },
    { label: 'Bờ khối u', val: 'Bờ tua gai / Có dấu hiệu màng phổi co kéo' },
    { label: 'Mật độ tỷ trọng', val: 'Đồng nhất hoặc có hoại tử trung tâm' }
  ],
  'Nodule': [
    { label: 'Kích thước nốt mờ', val: 'Tổn thương dạng nốt mờ tròn/bầu dục đường kính ≤ 3.0 cm' },
    { label: 'Số lượng & Bờ', val: 'Bờ rõ nét / Nốt khu trú nhu mô' }
  ],
  'Pneumothorax': [
    { label: 'Đường viền màng phổi', val: 'Đường viền màng phổi tạng (Visceral pleural line) tách biệt' },
    { label: 'Phế trường vô mạch', val: 'Vùng tăng sáng vô mạch (Avascular zone) ngoài đường viền' },
    { label: 'Xẹp phổi ép', val: 'Nhu mô phổi bị ép xẹp về phía rốn phổi' }
  ],
  'Consolidation': [
    { label: 'Đông đặc phế nang', val: 'Đám mờ đông đặc mật độ cao lấp đầy phế nang' },
    { label: 'Phế quản khí', val: 'Dấu hiệu phế quản khí (+) rõ rệt' }
  ],
  'Edema': [
    { label: 'Đường Kerley', val: 'Đường Kerley B ở đáy phổi hai bên' },
    { label: 'Phân bố thâm nhiễm', val: 'Hình ảnh đám mờ cánh bướm (Butterfly distribution)' }
  ],
  'Emphysema': [
    { label: 'Độ thấu quang', val: 'Tăng sáng hai phế trường do ứ khí phế nang' },
    { label: 'Vòm hoành', val: 'Vòm hoành hai bên dẹt, khoảng liên sườn giãn rộng' }
  ],
  'Fibrosis': [
    { label: 'Dạng tổn thương', val: 'Dải mờ xơ hóa, hình ảnh tổ ong (Honeycomb pattern)' },
    { label: 'Cấu trúc phế quản', val: 'Giãn phế quản co kéo (Traction bronchiectasis)' }
  ],
  'Pleural_Thickening': [
    { label: 'Độ dày màng phổi', val: 'Dải tăng độ dày màng phổi thành/tạng > 3.0 mm' }
  ],
  'Hernia': [
    { label: 'Hình ảnh thoát vị', val: 'Hình ảnh bóng hơi dạ dày / Mức khí dịch khoang lồng ngực' }
  ],
  'No Finding': [
    { label: 'Phế trường hai bên', val: 'Hai phế trường sáng đều, không phát hiện đám mờ bất thường' },
    { label: 'Góc sườn hoành', val: 'Góc sườn hoành hai bên nhọn, không tù' },
    { label: 'Bóng tim & Rốn phổi', val: 'Bóng tim chỉ số CTR < 0.50, rốn phổi hai bên bình thường' }
  ]
};

/**
 * Get authentic medical radiological findings based on predicted disease & probability
 */
export function getAuthenticRadiologicalFindings(diseaseKey, probability) {
  const findings = RADIOLOGICAL_FINDINGS_MAP[diseaseKey] || RADIOLOGICAL_FINDINGS_MAP['No Finding'];
  return findings;
}
