import unittest

import pymupdf as fitz
from Backend.ai_service.report_pdf import render_report


class ReportPdfTests(unittest.TestCase):
    def test_vietnamese_stored_conclusion_and_escaped_markup(self):
        pdf = render_report({
            "scanId": "SCAN-ONE", "revision": 3,
            "patient": {"name": "Bệnh nhân kiểm thử", "patientCode": "TEST-ONE"},
            "statusLabel": "Đã xác minh",
            "notes": "Kết luận đã lưu <tag> & nội dung tiếng Việt.",
            "report": {"summary": "Mô tả AI riêng", "findings": [], "recommendations": []},
        })
        with fitz.open(stream=pdf, filetype="pdf") as document:
            text = "".join(page.get_text() for page in document)
            self.assertIn("Bệnh nhân kiểm thử", text)
            self.assertIn("Kết luận đã lưu <tag> & nội dung tiếng Việt.", text)
            self.assertIn("SCAN-ONE", text)
            self.assertIn("Phiên bản 3", text)
            self.assertEqual(len(document), 1)

    def test_long_conclusion_flows_to_more_pages(self):
        pdf = render_report({"scanId": "SCAN-LONG", "notes":
            ("Dòng kết luận kiểm thử có tiếng Việt.\n" * 400) + "END-OF-CONCLUSION"})
        with fitz.open(stream=pdf, filetype="pdf") as document:
            self.assertGreater(len(document), 1)
            self.assertIn("END-OF-CONCLUSION", document[-1].get_text())
            for page in document:
                self.assertIn("SCAN-LONG", page.get_text())


if __name__ == "__main__":
    unittest.main()
