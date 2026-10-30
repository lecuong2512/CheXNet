# 📚 Thư Mục Tài Liệu Nghiên Cứu & Báo Cáo Kỹ Thuật (CheXNet Pro AI)

Thư mục này tập hợp toàn bộ các báo cáo kỹ thuật, phân tích kiến trúc, hướng dẫn triển khai và tổng kết thực nghiệm được phục hồi từ lịch sử phát triển của dự án **CheXNet Pro AI** (NCKH).

---

## 📑 Danh Mục Tài Liệu

| # | Tệp tin | Tiêu đề & Nội dung chính |
|---|---|---|
| 1 | [`bao_cao_tong_quat_chexnet.md`](./bao_cao_tong_quat_chexnet.md) | **Báo cáo tổng quát dự án CheXNet Pro AI** — Tổng hợp toàn diện mã nguồn, 18 bản walkthrough, kiến trúc Hybrid CNN-ViT (ConvNeXtV2 + SwinV2), các thông số AUROC, cấu hình GPU/VRAM. |
| 2 | [`bao_cao_du_an_chexnet.md`](./bao_cao_du_an_chexnet.md) | **Báo cáo phân tích toàn diện dự án** (700+ dòng) — Mục tiêu NCKH, danh sách 15 nhãn bệnh lý, kiến trúc 5 tầng, hệ thống training/fine-tune, timeline phát triển và danh mục toàn bộ file. |
| 3 | [`phan_tich_heatmap_vs_bbox.md`](./phan_tich_heatmap_vs_bbox.md) | **Phân tích cơ chế Attention Heatmap vs Bounding Box** — Lý giải nguyên nhân kiến trúc ViT chỉ tạo soft heatmap và cách giải quyết xung đột bằng cơ chế Cascade 2-Stage Filter + Display Fusion kết hợp YOLO11m. |
| 4 | [`toi_uu_vram_va_da_mo_hinh.md`](./toi_uu_vram_va_da_mo_hinh.md) | **Tối ưu hóa VRAM & Suy luận đa mô hình (CheXNet + YOLO11m)** — Giải pháp giảm tải bộ nhớ từ 3.7 GB xuống 1.21 GB trên GPU (RTX 3050 Laptop), load an toàn CPU -> GPU, Top-1 reporting. |
| 5 | [`vinbigdata_solutions_analysis.md`](./vinbigdata_solutions_analysis.md) | **Phân tích các giải pháp hàng đầu cuộc thi VinBigData Chest X-ray** — Chi tiết giải pháp Hạng Nhất (Cascade Classifier -> Detector) và Hạng Nhì (Ensemble MMDetection + Weighted Boxes Fusion - WBF của ZFTurbo). |
| 6 | [`enhanced_integration_analysis.md`](./enhanced_integration_analysis.md) | **Phân tích tích hợp nâng cao CheXNet × VinBigData** — 4 trụ cột cốt lõi (CAM-to-BBox, Feature Extraction, Auto Report, 2-Stage Cascade) và 7 chiều mở rộng (Multi-scale attention, FAISS CBIR, LLM Report). |
| 7 | [`analysis_results.md`](./analysis_results.md) | **Phân tích kiến trúc mô hình Hybrid CNN-ViT** — Chi tiết luồng xử lý: ConvNeXtV2 trích xuất đặc trưng cục bộ, SwinV2 trích xuất ngữ cảnh toàn cục, Attention Head, Gated Residual Masking, và Linear Classifier. |
| 8 | [`kaggle_yolo_hybrid_plan.md`](./kaggle_yolo_hybrid_plan.md) | **Kế hoạch chuẩn bị dữ liệu Hybrid YOLO trên Kaggle/Colab** — Bản đồ mapping 14 class giữa VinDr-CXR và NIH, kịch bản sinh pseudo-labels bằng CheXNet cho các nhãn thiếu bbox. |
| 9 | [`colab_faiss_guide.md`](./colab_faiss_guide.md) | **Hướng dẫn build FAISS Vector Index trên Google Colab** — Tối ưu hóa pipeline trích xuất vector đặc trưng 1536 chiều từ tập dữ liệu lớn bằng GPU T4/L4 trên Colab. |
| 10 | [`file_cleanup_list.md`](./file_cleanup_list.md) | **Danh sách phân loại và dọn dẹp file dự án** — Bảng đối chiếu tệp tin cốt lõi, tệp phụ trợ và quy tắc cấu hình `.gitignore`. |

---
*Tài liệu được trích xuất và bảo lưu tự động từ các phiên làm việc và hội thoại lịch sử của Antigravity Agent.*
