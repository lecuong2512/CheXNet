import json
from typing import List, Dict, Any, Optional

CLASS_NAMES = [
    'No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

DISEASE_NAMES_VI = {
    'Atelectasis': 'Xẹp phổi',
    'Cardiomegaly': 'Phì đại tim',
    'Effusion': 'Tràn dịch màng phổi',
    'Infiltration': 'Thâm nhiễm phổi',
    'Mass': 'Khối u phổi',
    'Nodule': 'Nốt mờ phổi',
    'Pneumonia': 'Viêm phổi',
    'Pneumothorax': 'Tràn khí màng phổi',
    'Consolidation': 'Đông đặc phổi',
    'Edema': 'Phù phổi',
    'Emphysema': 'Khí phế thũng',
    'Fibrosis': 'Xơ phổi',
    'Pleural_Thickening': 'Dày màng phổi',
    'Hernia': 'Thoát vị hoành',
    'No Finding': 'Không phát hiện bất thường'
}

class ClinicalReportGenerator:
    """Clinical Report Generator for the CheXNet medical imaging system."""
    
    CLINICAL_TEMPLATES = {
        'Atelectasis': {
            'finding': 'Vùng mờ phổi gợi ý tình trạng xẹp phổi (Atelectasis).',
            'recommendation': 'Đề nghị chụp CT ngực độ phân giải cao và nội soi phế quản nếu cần.',
            'severity_note': {
                'mild': 'Xẹp phổi khu trú nhẹ, theo dõi lâm sàng.',
                'moderate': 'Xẹp phổi diện trung bình, cần kiểm tra nguyên nhân.',
                'severe': 'Xẹp phổi diện rộng, cần can thiệp thông khí và tìm nguyên nhân chèn ép.'
            }
        },
        'Cardiomegaly': {
            'finding': 'Bóng tim phình to, chỉ số Tim/Lồng ngực (CTR) có dấu hiệu vượt ngưỡng 0.50.',
            'recommendation': 'Đề nghị siêu âm tim Doppler (Echocardiogram) và điện tâm đồ (ECG).',
            'severity_note': {
                'mild': 'Phì đại tim nhẹ, theo dõi định kỳ 6 tháng.',
                'moderate': 'Phì đại tim trung bình, cần theo dõi sát và đánh giá chức năng tim.',
                'severe': 'Phì đại tim nặng, cần can thiệp chuyên khoa Tim mạch khẩn.'
            }
        },
        'Effusion': {
            'finding': 'Phát hiện vùng mờ đồng nhất góc sườn hoành gợi ý Tràn dịch màng phổi.',
            'recommendation': 'Đề nghị siêu âm màng phổi và chụp CT ngực bổ sung.',
            'severity_note': {
                'mild': 'Lượng dịch ít, theo dõi.',
                'moderate': 'Lượng dịch trung bình, có thể cần chọc dò.',
                'severe': 'Lượng dịch nhiều, cần dẫn lưu khẩn cấp.'
            }
        },
        'Infiltration': {
            'finding': 'Hình ảnh thâm nhiễm phế nang/mô kẽ rải rác.',
            'recommendation': 'Khảo sát thêm bằng CT ngực và tương quan lâm sàng (dấu hiệu nhiễm trùng).',
            'severity_note': {
                'mild': 'Thâm nhiễm nhẹ, theo dõi định kỳ.',
                'moderate': 'Thâm nhiễm vừa, cân nhắc điều trị nguyên nhân.',
                'severe': 'Thâm nhiễm lan tỏa, cần chẩn đoán phân biệt ARDS hoặc viêm phổi nặng.'
            }
        },
        'Mass': {
            'finding': 'Phát hiện tổn thương dạng khối (Mass) >3cm tại nhu mô phổi.',
            'recommendation': 'Đề nghị chụp CT ngực có thuốc cản quang và sinh thiết (FNA).',
            'severity_note': {
                'mild': 'Khối nhỏ, ít dấu hiệu ác tính.',
                'moderate': 'Khối kích thước vừa, cần sinh thiết đánh giá.',
                'severe': 'Khối lớn xâm lấn, hội chẩn Ung bướu khẩn.'
            }
        },
        'Nodule': {
            'finding': 'Phát hiện nốt mờ đơn độc hoặc rải rác <3cm tại nhu mô phổi.',
            'recommendation': 'Khuyến cáo chụp CT ngực theo dõi theo hướng dẫn Fleischner.',
            'severity_note': {
                'mild': 'Nốt nhỏ đơn độc, theo dõi định kỳ.',
                'moderate': 'Nhiều nốt hoặc nốt kích thước trung bình, cần làm rõ.',
                'severe': 'Nhiều nốt kích thước lớn, khả năng di căn hoặc u hạt lớn.'
            }
        },
        'Pneumonia': {
            'finding': 'Đám mờ phế nang khu trú gợi ý Viêm phổi.',
            'recommendation': 'Kết hợp kháng sinh đồ, cấy đàm và đánh giá lâm sàng nhiễm trùng.',
            'severity_note': {
                'mild': 'Viêm phổi thùy nhẹ.',
                'moderate': 'Viêm phổi diện vừa, cân nhắc nhập viện.',
                'severe': 'Viêm phổi diện rộng, nguy cơ suy hô hấp.'
            }
        },
        'Pneumothorax': {
            'finding': 'Mất vân phổi ngoại vi, đường viền màng phổi hiện rõ, gợi ý Tràn khí màng phổi.',
            'recommendation': 'Đề nghị chụp X-quang thì thở ra hoặc CT ngực khẩn.',
            'severity_note': {
                'mild': 'Tràn khí ít, chưa chèn ép.',
                'moderate': 'Tràn khí lượng vừa, theo dõi sát.',
                'severe': 'Tràn khí lượng nhiều/áp lực, chọc hút hoặc dẫn lưu khẩn.'
            }
        },
        'Consolidation': {
            'finding': 'Vùng mờ đông đặc phế nang, có thể kèm hình ảnh khí ảnh nội phế quản.',
            'recommendation': 'Theo dõi và điều trị nguyên nhân viêm hoặc nhồi máu phổi.',
            'severity_note': {
                'mild': 'Đông đặc ít, rải rác.',
                'moderate': 'Đông đặc theo thùy.',
                'severe': 'Đông đặc diện rộng.'
            }
        },
        'Edema': {
            'finding': 'Mờ kẽ, rốn phổi lớn, đường Kerley B gợi ý Phù phổi cấp.',
            'recommendation': 'Hội chẩn Tim mạch, đánh giá chức năng thất trái và lợi tiểu.',
            'severity_note': {
                'mild': 'Phù mô kẽ.',
                'moderate': 'Phù phế nang khu trú.',
                'severe': 'Phù phổi cấp lan tỏa, cấp cứu.'
            }
        },
        'Emphysema': {
            'finding': 'Phổi sáng quá mức, cơ hoành dẹt, khoảng gian sườn giãn rộng gợi ý Khí phế thũng.',
            'recommendation': 'Đo chức năng hô hấp (Spirometry) và CT ngực đánh giá tổn thương.',
            'severity_note': {
                'mild': 'Khí phế thũng nhẹ.',
                'moderate': 'Khí phế thũng vừa, cần hỗ trợ hô hấp.',
                'severe': 'Khí phế thũng nặng, bóng khí lớn nguy cơ vỡ.'
            }
        },
        'Fibrosis': {
            'finding': 'Hình ảnh lưới nốt, tổn thương xơ kẽ lan tỏa hoặc khu trú.',
            'recommendation': 'CT ngực phân giải cao (HRCT) để chẩn đoán xơ phổi kẽ.',
            'severity_note': {
                'mild': 'Xơ hóa mức độ nhẹ.',
                'moderate': 'Xơ phổi diện vừa, giảm chức năng hô hấp.',
                'severe': 'Xơ hóa tổ ong, bệnh lý phổi mô kẽ giai đoạn muộn.'
            }
        },
        'Pleural_Thickening': {
            'finding': 'Dày dính màng phổi, tù góc sườn hoành.',
            'recommendation': 'Theo dõi định kỳ, loại trừ bệnh lý màng phổi do amiăng hoặc lao.',
            'severity_note': {
                'mild': 'Dày màng phổi ít.',
                'moderate': 'Dày màng phổi mảng lớn.',
                'severe': 'Dày màng phổi lan tỏa gây hạn chế hô hấp.'
            }
        },
        'Hernia': {
            'finding': 'Hình ảnh bóng khí hoặc quai ruột xuất hiện trong lồng ngực qua cơ hoành.',
            'recommendation': 'Siêu âm, CT ngực/bụng để xác định thoát vị hoành và hội chẩn Ngoại khoa.',
            'severity_note': {
                'mild': 'Thoát vị nhỏ.',
                'moderate': 'Thoát vị vừa, nguy cơ chèn ép.',
                'severe': 'Thoát vị lớn, chèn ép tạng, cần phẫu thuật sớm.'
            }
        }
    }
    
    def _classify_severity(self, lesion_area_percent: float) -> str:
        """Phân loại mức độ nghiêm trọng dựa trên diện tích tổn thương."""
        if lesion_area_percent < 5:
            return 'mild'
        elif 5 <= lesion_area_percent < 15:
            return 'moderate'
        else:
            return 'severe'
            
    def _get_severity_label_vi(self, severity: str) -> str:
        """Chuyển đổi mức độ sang tiếng Việt."""
        mapping = {
            'mild': 'Nhẹ',
            'moderate': 'Trung bình',
            'severe': 'Nặng'
        }
        return mapping.get(severity, 'Không xác định')
        
    def _get_confidence_level_vi(self, confidence: float) -> str:
        if confidence >= 0.8:
            return 'Cao'
        elif confidence >= 0.5:
            return 'Trung bình'
        return 'Thấp'

    def generate(self, predictions: List[Dict[str, Any]], co_occurrences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tạo báo cáo lâm sàng có cấu trúc bằng tiếng Việt từ kết quả dự đoán.
        """
        sorted_preds = sorted(predictions, key=lambda x: x.get('calibrated_confidence', 0.0), reverse=True)
        
        findings_list = []
        recommendations_set = set()
        warnings_list = []
        
        critical_diseases = ['Pneumothorax', 'Pneumonia']
        
        for pred in sorted_preds:
            disease = pred.get('disease', 'No Finding')
            if disease == 'No Finding':
                continue
                
            prob = pred.get('probability', 0.0)
            conf = pred.get('calibrated_confidence', 0.0)
            uncert = pred.get('uncertainty', 0.0)
            area_pct = pred.get('lesion_area_percent', 0.0)
            loc = pred.get('lesion_location', 'Không xác định')
            
            template = self.CLINICAL_TEMPLATES.get(disease)
            if not template:
                continue
                
            severity = self._classify_severity(area_pct)
            severity_vi = self._get_severity_label_vi(severity)
            
            finding_text = template.get('finding', '')
            sev_note = template.get('severity_note', {}).get(severity, '')
            if sev_note:
                finding_text += f" {sev_note}"
                
            rec = template.get('recommendation', '')
            if rec:
                recommendations_set.add(rec)
                
            uncertainty_note = f"± {uncert*100:.0f}%" if uncert <= 0.12 else f"⚠️ Không chắc chắn (± {uncert*100:.0f}%)"
            
            if uncert > 0.12:
                warnings_list.append(f"Dự đoán {DISEASE_NAMES_VI.get(disease, disease)} có độ bất định cao ({uncert*100:.0f}%).")
                
            if disease in critical_diseases and prob > 0.3:
                warnings_list.append(f"CẢNH BÁO: Phát hiện {DISEASE_NAMES_VI.get(disease, disease)} (Xác suất {prob*100:.1f}%). Cần chú ý lâm sàng!")
                
            finding_obj = {
                'disease': disease,
                'disease_vi': DISEASE_NAMES_VI.get(disease, disease),
                'probability': prob,
                'confidence_level': self._get_confidence_level_vi(conf),
                'uncertainty_note': uncertainty_note,
                'severity': severity,
                'severity_vi': severity_vi,
                'finding_text': finding_text,
                'location': loc,
                'area_percent': area_pct,
            }
            findings_list.append(finding_obj)
            
        num_findings = len(findings_list)
        if num_findings == 0:
            summary = "Không phát hiện bất thường đáng kể trên phim X-quang hiện tại."
        else:
            severe_levels = {'severe': 3, 'moderate': 2, 'mild': 1}
            worst_finding = max(findings_list, key=lambda x: severe_levels.get(x['severity'], 0))
            summary = f"Phát hiện {num_findings} bất thường. Đáng chú ý nhất là {worst_finding['disease_vi']} ở mức độ {worst_finding['severity_vi'].lower()}."
            
        co_occurrence_notes = []
        for co in co_occurrences:
            note = co.get('clinical_note')
            if note:
                co_occurrence_notes.append(note)
                
        return {
            'summary': summary,
            'findings': findings_list,
            'recommendations': list(recommendations_set),
            'warnings': warnings_list,
            'co_occurrence_notes': co_occurrence_notes,
        }
