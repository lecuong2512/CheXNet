"""Describe every classifier decision without changing the original inference pipeline."""
import math


# Display names follow README's disease catalogue. Model class keys stay unchanged.
DISEASE_LABELS = {
    "No Finding": "Không phát hiện bất thường",
    "Atelectasis": "Xẹp phổi",
    "Cardiomegaly": "Phì đại bóng tim (Tim to)",
    "Effusion": "Tràn dịch màng phổi",
    "Infiltration": "Thâm nhiễm nhu mô phổi",
    "Mass": "Khối u phổi (> 3cm)",
    "Nodule": "Nốt mờ phổi (≤ 3cm)",
    "Pneumonia": "Viêm phổi",
    "Pneumothorax": "Tràn khí màng phổi",
    "Consolidation": "Đông đặc phế nang",
    "Edema": "Phù phổi cấp",
    "Emphysema": "Khí phế thũng",
    "Fibrosis": "Xơ hóa phổi",
    "Pleural_Thickening": "Dày màng phổi",
    "Hernia": "Thoát vị hoành",
}


def classification_summary(probabilities, class_names, thresholds=None, visualized=()):
    """Use thresholds in model channel order, never probability/ranking order."""
    if thresholds is not None and len(thresholds) != len(class_names):
        raise ValueError("Threshold count does not match classifier labels")
    by_class = {}
    for index, name in enumerate(class_names):
        threshold = float(thresholds[index]) if thresholds is not None else 0.5
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid classifier threshold")
        by_class[name] = threshold
    decisions = []
    for item in probabilities:
        name = item["disease"]
        decisions.append({
            "disease": name,
            "label": DISEASE_LABELS.get(name, name),
            "probability": item["probability"],
            "threshold": by_class[name],
            "aboveThreshold": item["probability"] >= by_class[name],
        })
    decisions.sort(key=lambda item: item["probability"], reverse=True)
    findings = [item for item in decisions if item["disease"] != "No Finding" and item["aboveThreshold"]]
    normal = next((item for item in decisions if item["disease"] == "No Finding"), None)
    conflict = bool(findings and normal and normal["aboveThreshold"])
    visualized = set(visualized)
    missing_maps = [item["disease"] for item in findings if item["disease"] not in visualized]
    warnings = []
    if thresholds is None:
        warnings.append("Đang dùng ngưỡng mặc định 0,5; chưa có tệp ngưỡng riêng cho checkpoint này.")
    if conflict:
        warnings.append("Nhãn không phát hiện bất thường và nhãn bệnh cùng vượt ngưỡng. Cần bác sĩ đối chiếu lại ảnh.")
    if missing_maps:
        warnings.append(f"Có {len(missing_maps)} nhãn bệnh vượt ngưỡng chưa có bản đồ chú ý. Các nhãn này vẫn được liệt kê đầy đủ.")
    return {
        "version": 1,
        "thresholdSource": "artifact" if thresholds is not None else "default",
        "decisions": decisions,
        "findings": findings,
        "conflictingNoFinding": conflict,
        "withoutHeatmap": missing_maps,
        "summary": f"Có {len(findings)} nhãn bệnh vượt ngưỡng của mô hình." if findings else
            "Không có nhãn bệnh vượt ngưỡng của mô hình. Kết quả này chưa xác nhận ảnh bình thường.",
        "warnings": warnings,
    }
