# 🏗️ GLOBAL CODE REVIEW GUIDELINES
**Version:** 1.1 (Standardized)
**Scope:** Node.js / TypeScript / MongoDB Ecosystem

---

## 1. QUY TRÌNH REVIEW (WORKFLOW)

### Thao tác Git
```bash
# Fetch và kiểm tra Merge Request (MR)
git checkout develop
git fetch origin refs/merge-requests/<ID>/head:mr-<ID>
git checkout mr-<ID>

# Phân tích so với nhánh gốc
git diff develop...mr-<ID> --stat
git log develop...mr-<ID> --no-merges --pretty=format:"%ae | %ad | %s" --date=short
```

### Cấu trúc báo cáo đầu ra
- **Technical Summary (`MR_#ID-Summary.md`)**: Tóm tắt tình trạng merge cho stakeholders.
- **Critical Risk Report (`MR_#ID-Critical.md`)**: Báo cáo kỹ thuật chi tiết dành cho Lead/Architect, tập trung vào rủi ro.

---

## 2. PHÂN LOẠI MỨC ĐỘ & PHÁN QUYẾT

| Cấp độ | Định nghĩa | Hành động yêu cầu |
|---|---|---|
| **P0 (Critical)** | Lỗ hổng bảo mật, rủi ro mất dữ liệu, bug production. | **Block Merge** ngay lập tức. |
| **P1 (High)** | Sai logic nghiệp vụ, thắt nút cổ chai hiệu năng. | **Bắt buộc Fix** trước khi merge. |
| **P2 (Medium)** | Code smell, vi phạm pattern tối ưu, khó bảo trì. | Nên fix, không bắt buộc block merge. |
| **P3 (Low)** | Convention, style, refactoring nhỏ. | Tùy chọn (Nice to have). |

### Trạng thái phán quyết
- **APPROVED ✅**: Không còn lỗi P0 hoặc P1.
- **CAUTION ⚠️**: Còn lỗi P1 nhưng không gây rủi ro vận hành ngay lập tức.
- **REQUEST CHANGES ❌**: Tồn tại bất kỳ lỗi P0 hoặc P1 ảnh hưởng bảo mật/toàn vẹn dữ liệu.

---

## 3. CHECKLIST KIỂM TRA KỸ THUẬT

### 3.1 Hiệu năng & Khả năng mở rộng
- [ ] **N+1 Query Avoidance**: Tuyệt đối không gọi database trong vòng lặp. Sử dụng Pre-fetch + Map.
- [ ] **Batch Operations**: Ưu tiên `insertMany()` hoặc `bulkWrite()` thay vì lặp lại các lệnh đơn lẻ.
- [ ] **Async Management**: Sử dụng `Promise.all()` cho các tác vụ I/O độc lập.
- [ ] **Memory Management**: Sử dụng Streams (`createReadStream`) khi xử lý file dung lượng lớn.

### 3.2 Toàn vẹn dữ liệu & Bảo mật
- [ ] **Transaction**: Các thao tác ghi trên nhiều collection đồng thời bắt buộc dùng `withTransaction`.
- [ ] **Resource Guard**: Mọi lớp truy cập dữ liệu phải lọc theo `ownerId` / `orgId` / `tenantId` từ context bảo mật.
- [ ] **I/O Validation**: Kiểm tra tên file bằng Regex để chống Path Traversal trước khi thực thi stream.

### 3.3 Localization & Timezone
- [ ] **Consistent Timezone**: Luôn sử dụng `moment-timezone` với múi giờ hệ thống (VD: `'Asia/Ho_Chi_Minh'`).
- [ ] **No UTC Hardcoding**: Tránh dùng chuỗi `Z` cứng; sử dụng `.endOf('day').toDate()` cho các bộ lọc khoảng thời gian.

---

## 4. CHỈ SỐ ĐÁNH GIÁ (SCORECARD)
| Tiêu chí | Điểm /10 | Ghi chú |
|---|---|---|
| Tính năng | | |
| Hiệu suất | | |
| Bảo mật | | |
| Code Quality | | |
| Git Hygiene | | |

---
*Cập nhật lần cuối: 25/04/2026 | Standardized Global Template*
