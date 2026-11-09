"""Render a stored diagnosis to PDF without loading the inference model."""
import io
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def render_report(data: dict) -> bytes:
    candidates = [
        os.environ.get("REPORT_FONT_PATH"),
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = next((Path(p) for p in candidates if p and Path(p).is_file()), None)
    if font is None:
        raise RuntimeError("Set REPORT_FONT_PATH to a Vietnamese-capable TrueType font")
    pdfmetrics.registerFont(TTFont("ReportBody", str(font)))
    body = ParagraphStyle("body", fontName="ReportBody", fontSize=10.5, leading=15, spaceAfter=7)
    heading = ParagraphStyle("heading", parent=body, fontSize=12, leading=17, spaceBefore=12,
                             textColor=colors.HexColor("#123450"), keepWithNext=True)
    title = ParagraphStyle("title", parent=heading, fontSize=17, leading=23, alignment=TA_CENTER,
                           spaceAfter=16)
    report = data.get("report") or {}
    patient = data.get("patient") or {}
    story = []

    def paragraph(text, style=body):
        story.append(Paragraph(escape(str(text)).replace("\n", "<br/>"), style))

    paragraph("PHIẾU PHÂN TÍCH X-QUANG NGỰC", title)
    paragraph(f"{patient.get('name', 'Chưa ghi nhận')} · {patient.get('patientCode', '')}")
    paragraph(f"Tuổi {patient.get('age', '—')} · Giới tính {patient.get('gender', '—')}")
    paragraph(f"Mã lần chụp {data['scanId']}")
    paragraph(f"Thời điểm chụp {data.get('createdAt', 'Chưa ghi nhận')}")
    paragraph(f"Trạng thái {data.get('statusLabel', '')} · Phiên bản {data.get('revision', 0)}")
    paragraph("Kỹ thuật và dữ liệu phân tích", heading)
    paragraph(f"{data.get('scanType', 'Chưa ghi nhận tư thế')} · {data.get('model', 'Chưa ghi nhận model')}")
    trace = data.get('inferenceTrace')
    if trace:
        paragraph(f"Mã đối chiếu xử lý: {trace['id']}")
        completed = datetime.fromisoformat(trace['completedAt']).astimezone(timezone(timedelta(hours=7)))
        paragraph(f"Hoàn tất lúc {completed.strftime('%d/%m/%Y %H:%M:%S')} (giờ Việt Nam)")
        for key, label in [('hybrid', 'Hybrid CNN–ViT'), ('yolo', 'YOLO')]:
            stage = trace['models'][key]
            paragraph(f"{label}: " + (f"Đã ghi nhận thực thi · {stage['forwardCalls']} lượt" if stage['observed'] else "Chưa ghi nhận thực thi"))
    classification = data.get("classification")
    if classification:
        paragraph("Danh sách nhãn vượt ngưỡng", heading)
        paragraph(classification["summary"])
        paragraph("Tên nhãn là nhóm phân loại; kích thước ghi trong tên nhãn không phải số đo từ ảnh.")
        for finding in classification["findings"]:
            score = f"{finding['probability'] * 100:.1f}".replace(".", ",")
            threshold = f"{finding['threshold'] * 100:.1f}".replace(".", ",")
            paragraph(f"{finding['label']}: {score}% · Ngưỡng {threshold}%")
        for warning in classification["warnings"]:
            paragraph(warning)
        if classification["withoutHeatmap"]:
            paragraph("Mô tả dưới đây chỉ thuộc các nhãn có bản đồ chú ý, không thay thế danh sách đầy đủ phía trên.")
    else:
        paragraph("Phiếu cũ chưa lưu ngưỡng phân loại. Danh sách mô tả có thể chưa bao gồm mọi nhãn vượt ngưỡng.")
    paragraph("Mô tả do AI đề xuất", heading)
    findings = report.get("findings") or []
    if findings:
        for finding in findings:
            paragraph(finding.get("finding_text", ""))
    else:
        paragraph(report.get("summary") or "Chưa có bản mô tả AI")
    paragraph("Kết luận AI", heading)
    paragraph((classification or {}).get("summary") or report.get("summary") or "Chưa có kết luận AI")
    paragraph("Khuyến nghị do AI đề xuất", heading)
    for recommendation in report.get("recommendations") or []:
        paragraph("- " + str(recommendation))
    for warning in report.get("warnings") or []:
        paragraph(warning)
    paragraph("Kết luận bác sĩ đã lưu", heading)
    paragraph(data.get("notes") or "Chưa có kết luận bác sĩ")
    if data.get("verifiedAt"):
        paragraph(f"Người cập nhật {data.get('verifiedByName') or data.get('verifiedBy', 'Chưa ghi nhận')}")
        paragraph(f"Thời điểm cập nhật {data['verifiedAt']}")
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=48, rightMargin=48,
                                topMargin=42, bottomMargin=48, title="Phiếu phân tích X-quang ngực")

    def footer(canvas, doc):
        canvas.setFont("ReportBody", 8)
        canvas.setFillColor(colors.HexColor("#596675"))
        canvas.drawString(48, 25, f"{data['scanId']} · v{data.get('revision', 0)}")
        canvas.drawRightString(A4[0] - 48, 25, f"Trang {doc.page}")

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


if __name__ == "__main__":
    sys.stdout.buffer.write(render_report(json.loads(sys.stdin.buffer.read().decode("utf-8"))))
