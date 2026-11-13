# Danh sách file đề xuất xóa

Dưới đây là danh sách các file **không cần thiết / không quan trọng** trong project CheXNet. Vui lòng duyệt trước khi tôi xóa.

---

## 1. `__pycache__` (Bytecode tự sinh, an toàn xóa)

| File/Thư mục | Kích thước | Lý do |
|---|---|---|
| `__pycache__/` (gốc project) | ~45KB | Bytecode Python, tự động tạo lại |
| `Models/__pycache__/` | ~108KB | Bytecode Python, tự động tạo lại |
| `Backend/__pycache__/` | ~81KB | Bytecode Python, tự động tạo lại |
| `Backend/scripts/__pycache__/` | ~7KB | Bytecode Python, tự động tạo lại |

## 2. File log cũ (Không ảnh hưởng chức năng)

| File | Kích thước | Lý do |
|---|---|---|
| [`faiss_build.log`](file:///d:/temp/tempnckh/CheXNet/faiss_build.log) | 2KB | Log cũ từ quá trình build FAISS index |
| [`faiss_full_build.log`](file:///d:/temp/tempnckh/CheXNet/faiss_full_build.log) | 1.5KB | Log cũ từ quá trình build FAISS index |

## 3. Script debug/scratch (Không dùng nữa)

| File | Kích thước | Lý do |
|---|---|---|
| [`inspect_ckpt.py`](file:///d:/temp/tempnckh/CheXNet/inspect_ckpt.py) | 422B | Script debug kiểm tra checkpoint, đã hoàn thành mục đích |
| [`check_size.py`](file:///d:/temp/tempnckh/CheXNet/check_size.py) | 565B | Script kiểm tra kích thước file, không cần cho production |

## 4. File rỗng / placeholder

| File | Kích thước | Lý do |
|---|---|---|
| [`Backend/utils.py`](file:///d:/temp/tempnckh/CheXNet/Backend/utils.py) | 0B (rỗng) | File rỗng, không chứa code nào |
| [`Trainedmodel/a.txt`](file:///d:/temp/tempnckh/CheXNet/Trainedmodel/a.txt) | - | File placeholder không rõ mục đích |
| [`Frontend/ft.txt`](file:///d:/temp/tempnckh/CheXNet/Frontend/ft.txt) | - | File không rõ mục đích |

---

## GIỮ LẠI (KHÔNG XÓA)

Các file sau **QUAN TRỌNG**, không nên xóa:

| File | Lý do giữ |
|---|---|
| `Backend/data/` (thư mục rỗng) | Cần cho FAISS index khi build |
| `Backend/scripts/build_faiss_index.py` | Script build FAISS index |
| `Finetune/*` | Code finetune model |
| `Notebooks/yolov11m_fixed.ipynb` | Notebook YOLO (sẽ được tối ưu) |
| Tất cả file `.py` chính | Core functionality |

> [!IMPORTANT]
> Hãy xác nhận những file nào bạn muốn xóa. Tôi sẽ chỉ xóa sau khi được duyệt.
