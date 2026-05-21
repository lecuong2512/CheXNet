# Frontend Skill Memory (FE DNA & Coding Standard)

Tài liệu này tổng hợp toàn bộ kiến trúc (Architecture), quy chuẩn lập trình (Coding Standards), và các mẫu thiết kế (Design Patterns) được đúc kết trực tiếp từ mã nguồn frontend của dự án **AHV HRM Frontend** (`ahv-hrm-frontend`). 

Mục tiêu của tài liệu là giúp các AI Coding Agent hoặc lập trình viên mới có thể xây dựng các tính năng mới, module mới, hoặc nhân bản (clone) trọn vẹn phong cách thiết kế này sang các dự án tương lai mà **không làm gãy vỡ kiến trúc hiện tại**, đảm bảo mã nguồn sinh ra hoàn toàn nhất quán với cách viết của đội ngũ phát triển gốc.

---

## 1. Project DNA

* **Tech Stack lõi**: React 19 (`^19.2.4`) + TypeScript (`~5.9.3`) + Vite (`^8.0.1`).
* **Package Manager**: Trình quản lý gói bắt buộc là **Yarn** (được xác thực qua sự hiện diện của tệp `yarn.lock`). Các lệnh khởi chạy chính: `yarn dev`, `yarn build`, `yarn lint`, `yarn format`.
* **UI/Component Library**: Ant Design v6 (`^6.3.4`) kết hợp với `@ant-design/icons` và `react-icons`.
* **Styling System**: Tailwind CSS v3 (`^3.4.1`) kết hợp với Vanilla CSS/SCSS (toàn cục qua `index.css` và `App.css`, ghi đè linh hoạt các thành phần của Ant Design).
* **State Management & API Layer**: Redux Toolkit (`^2.11.2`) kết hợp với **RTK Query** (`@reduxjs/toolkit/query/react`) làm API Service layer chính. Có sử dụng một instance `axios` thuần cho một số luồng cụ thể hoặc custom interceptor.
* **Routing**: React Router Dom v7 (`^7.13.2`) với mô hình routing khai báo qua component tĩnh kết hợp tính năng **Lazy Loading** (`React.lazy` + `Suspense`).
* **Realtime / WebSocket**: `socket.io-client` (`^4.8.3`) dùng để nhận thông báo realtime (OKR, Checkin, Nghỉ phép, Bảng lương).
* **Bảo mật Dữ liệu (Payload Encryption)**: Tích hợp thư viện `crypto-js` (`CryptoJS`) để mã hóa các luồng dữ liệu nhạy cảm theo tiêu chuẩn **AES-256-CBC** khớp hoàn toàn với Backend (định dạng `iv_hex:ciphertext_hex`).
* **Biểu đồ & Kéo thả**: Sử dụng `@xyflow/react` / `dagre` cho sơ đồ tổ chức (OrgChart), `@hello-pangea/dnd` cho tính năng kéo thả (DND), `@fortune-sheet/react` và `exceljs` cho xử lý bảng tính/báo cáo.
* **Rich Text Editor**: TiTap (`@tiptap/react`, `@tiptap/starter-kit`).

---

## 2. Folder Structure Standard

Kiến trúc thư mục tuân theo mô hình **Feature-driven / Module-based kết hợp Core Shared Layers**, giúp dự án dễ dàng mở rộng quy mô (scalable).

```text
src/
├── assets/             # Chứa hình ảnh tĩnh, icon, tệp SCSS/CSS dùng chung (global.scss)
├── components/         # Các thành phần giao diện (UI) dùng chung toàn hệ thống
│   ├── common/         # Layout tĩnh, Sidebar, Topbar, AccessControl, PageLoading...
│   └── ui/             # Các component UI nguyên tử/tùy biến: StatusTag, FilterCard, DebouncedSearchInput...
├── configs/            # Cấu hình các thư viện bên thứ ba (axios.ts, socket.ts)
├── constants/          # Hằng số toàn cục (common.ts, permissions.ts, roles.ts, status.ts...)
├── hooks/              # Custom hooks dùng chung toàn app (useDebounce.ts, usePermission.ts...)
├── modules/            # THƯ MỤC QUAN TRỌNG NHẤT: Chứa các tính năng độc lập (Feature modules)
│   ├── auth/           # Module xác thực (Login, Reset Password...)
│   ├── employees/      # Module nhân sự
│   ├── okrs/           # Module quản lý mục tiêu (OKRs)
│   ├── leave-requests/ # Module quản lý đơn từ nghỉ phép
│   └── ...             # Các module khác (attendance, contracts, payrolls, settings...)
├── pages/              # Các trang tĩnh cấp cao hoặc fallback (PlaceholderPages, UnauthorizedPage)
├── providers/          # Context Providers toàn cục (AppThemeProvider.tsx)
├── routers/            # Cấu hình định tuyến (AppRouter.tsx, GuardRoute.tsx, ProtectRouter.tsx, PublicRoute.tsx)
├── stores/             # Cấu hình Redux Store toàn cục và Base RTK Query API
│   ├── store.ts        # Redux store gốc, tích hợp middleware và reducers
│   ├── baseApi.ts      # Khởi tạo RTK Query baseApi với fetchBaseQuery & luồng tự động re-auth (refresh token)
│   ├── notificationSlice.ts # Redux slice quản lý trạng thái thông báo
│   └── themeSlice.ts   # Redux slice quản lý giao diện Sáng/Tối (Dark/Light mode)
└── utils/              # Các hàm tiện ích thuần túy (apiUtils.ts, formUtils.ts, cryptoUtils.ts...)
```

### Cấu trúc tiêu chuẩn bên trong một Module (`src/modules/<module-name>/`)
Mỗi module hoạt động như một micro-domain độc lập, đóng gói giao diện, logic và kết nối API của riêng nó:
```text
modules/<module-name>/
├── components/         # Các component con chỉ phục vụ riêng cho module này (VD: EmployeeModal.tsx)
├── pages/              # Các trang chính (Page components) được export để gắn vào Router (VD: EmployeePage.tsx)
└── services/           # Lớp kết nối API bằng cách injectEndpoints vào baseApi (VD: employeeApi.ts)
```

---

## 3. Quy chuẩn Format & Linter (Prettier & ESLint)

Dự án áp dụng chặt chẽ các quy tắc định dạng mã nguồn (đọc trực tiếp từ `.prettierrc` và `eslint.config.js`):

### 1. Prettier Rules
* **Dấu ngoặc kép (Quotes)**: Bắt buộc dùng chuỗi ngoặc kép `"double quotes"` (`singleQuote: false`).
* **Dấu chấm phẩy (Semicolons)**: Bắt buộc có dấu chấm phẩy ở cuối dòng (`semi: true`).
* **Dấu phẩy đuôi (Trailing Comma)**: Đặt ở mọi cấu trúc đối tượng/mảng (`trailingComma: "all"`).
* **Độ rộng dòng (Print Width)**: Giới hạn tối đa **120 ký tự** (`printWidth: 120`).
* **Ký tự xuống dòng (End of Line)**: Định dạng Windows CRLF (`endOfLine: "crlf"`).
* **Thụt lề (Tab Width)**: **2 khoảng trắng** (`tabWidth: 2`).

### 2. ESLint Rules
* **Kiểu dữ liệu `any`**: Được nới lỏng, cho phép sử dụng (`@typescript-eslint/no-explicit-any: 'off'`).
* **Biến không sử dụng**: Được bỏ qua cảnh báo (`@typescript-eslint/no-unused-vars: 'off'`).
* **Mảng phụ thuộc của Hook**: Không kiểm tra khắt khe việc liệt kê thiếu/thừa biến phụ thuộc (`react-hooks/exhaustive-deps: 'off'`).
* **Định tuyến Nhập liệu (Imports)**: Luôn ưu tiên dùng **đường dẫn tương đối** chuẩn (`../../..`) khi import giữa các module thay vì dùng alias `@/`. (Vite cấu hình duy nhất alias `bootstrap`).

---

## 4. Component Pattern

Dự án áp dụng chặt chẽ phong cách viết **Functional Component** với **React Hooks**.

### Quy tắc thiết kế Component:
1. **Khai báo kiểu dữ liệu**: Luôn sử dụng `React.FC` (hoặc định nghĩa hàm trả về `React.JSX.Element` / `React.ReactNode`).
2. **Quản lý Props**: Sử dụng `interface` để định nghĩa `Props`. Bắt buộc giải nén (destructure) props ngay tại tham số của hàm.
3. **Tối ưu hóa Re-render**:
   * Áp dụng triệt để `useMemo` cho các dữ liệu tính toán phức tạp, cấu hình cột của bảng (`columns`), hoặc danh sách options lọc.
   * Sử dụng `useCallback` cho các hàm xử lý sự kiện (`handleAdd`, `handleEdit`, `handleDelete`, `handleFilterChange`) truyền xuống component con để tránh tạo lại tham chiếu hàm.
   * Gói các component tĩnh hoặc component nhận callback bằng `React.memo` (ví dụ: `Sidebar`, `Topbar`).
4. **Tách biệt Logic và UI**: Các component trang (`<Module>Page.tsx`) chịu trách nhiệm gọi hooks lấy dữ liệu, xử lý phân quyền, cấu hình bộ lọc và bảng. Phần form nhập liệu hoặc chi tiết phức tạp bắt buộc tách sang thư mục `components/` dưới dạng Modal/Drawer (ví dụ: `EmployeeModal.tsx`, `OkrModal.tsx`).

### Mẫu mã Component Chuẩn (Template):
```tsx
import React, { useState, useMemo, useCallback } from "react";
import { Button, Table, Space, Card, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { PlusOutlined, EditOutlined, DeleteOutlined } from "@ant-design/icons";
import AccessControl from "../../../components/common/AccessControl";
import StatusTag from "../../../components/ui/StatusTag";
import FilterCard, { FilterItem } from "../../../components/ui/FilterCard";
import { PERMISSIONS } from "../../../constants/permissions";
import { ACTION_LABELS, PAGINATION_CONFIG } from "../../../constants/common";

const { Title } = Typography;

interface CustomFeatureProps {
  categoryId?: string;
}

const CustomFeaturePage: React.FC<CustomFeatureProps> = ({ categoryId }) => {
  // 1. State quản lý
  const [selectedRecord, setSelectedRecord] = useState<any>(null);
  const [pagination, setPagination] = useState({
    current: PAGINATION_CONFIG.DEFAULT_PAGE,
    pageSize: PAGINATION_CONFIG.DEFAULT_PAGE_SIZE,
  });

  // 2. Callback actions
  const handleEdit = useCallback((record: any) => {
    setSelectedRecord(record);
    // Mở modal...
  }, []);

  const handleDelete = useCallback((id: string) => {
    // Gọi API xóa...
  }, []);

  const handleTableChange = useCallback((newPagination: any) => {
    setPagination((prev) => ({
      ...prev,
      current: newPagination.current,
      pageSize: newPagination.pageSize,
    }));
  }, []);

  // 3. Cấu hình cột (Bắt buộc bọc trong useMemo)
  const columns: ColumnsType<any> = useMemo(() => [
    {
      title: "Tên tính năng",
      dataIndex: "name",
      key: "name",
      render: (text) => <span className="font-semibold text-[#1a1a1a] dark:text-gray-100">{text}</span>,
    },
    {
      title: "Trạng thái",
      dataIndex: "status",
      key: "status",
      render: (status: string) => <StatusTag status={status} />,
    },
    {
      title: "Hành động",
      key: "action",
      render: (_, record) => (
        <Space size="middle">
          <AccessControl permission={PERMISSIONS.ALL}>
            <Button
              type="text"
              size="small"
              icon={<EditOutlined className="text-primary-purple" />}
              onClick={() => handleEdit(record)}
              className="flex items-center justify-center hover:bg-purple-50 rounded-lg"
            >
              {ACTION_LABELS.UPDATE}
            </Button>
          </AccessControl>
        </Space>
      ),
    },
  ], [handleEdit]);

  return (
    <div className="custom-feature-container">
      <div className="flex justify-between items-center mb-6">
        <Title level={3} style={{ margin: 0, fontWeight: 800 }}>
          Danh sách Tính năng
        </Title>
        <AccessControl permission={PERMISSIONS.ALL}>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            className="h-10 px-6 rounded-xl flex items-center gap-2 font-bold bg-gradient-purple border-none shadow-lg shadow-purple-100"
          >
            {ACTION_LABELS.CREATE}
          </Button>
        </AccessControl>
      </div>

      {/* Bọc Bảng bằng Thẻ Card cao cấp */}
      <Card
        className="rounded-[2rem] border-none shadow-xl shadow-gray-200/50 overflow-hidden"
        bodyStyle={{ padding: 0 }}
      >
        <Table 
          columns={columns} 
          dataSource={[]} 
          rowKey="_id" 
          pagination={{
            ...pagination,
            total: 0,
            showSizeChanger: true,
            pageSizeOptions: PAGINATION_CONFIG.PAGE_SIZE_OPTIONS,
            showTotal: (total, range) => `${range[0]}-${range[1]} của ${total} bản ghi`,
          }}
          onChange={handleTableChange}
          className="custom-table"
          scroll={{ x: "max-content" }}
        />
      </Card>
    </div>
  );
};

export default CustomFeaturePage;
```

---

## 5. API Pattern (RTK Query Service Layer)

Toàn bộ ứng dụng sử dụng mô hình **RTK Query Code Splitting**. Thay vì khởi tạo nhiều API riêng lẻ, tất cả các module đều mở rộng từ một `baseApi` duy nhất đặt tại `src/stores/baseApi.ts`.

### Đặc điểm của `baseApi`:
* Đã cấu hình sẵn `fetchBaseQuery` tự động đính kèm `Authorization: Bearer <token>` từ `localStorage`.
* Tự động xử lý đánh chặn lỗi **401 Unauthorized** để gọi luồng **Refresh Token** ngầm, cập nhật lại token và thực hiện lại request gốc liền mạch (Re-auth pattern).
* Có cơ chế tự động serialize params (chuyển đổi định dạng URL, fix lỗi mã hóa ký tự ISO Date `:` bị biến thành `%3A`).

### Quy tắc tạo Service cho Module mới:
1. Tạo tệp `src/modules/<module-name>/services/<module>Api.ts`.
2. Import `baseApi` từ `src/stores/baseApi`.
3. Sử dụng phương thức `baseApi.injectEndpoints({...})` để chèn thêm các Query và Mutation.
4. Đặt thuộc tính `overrideExisting: false`.
5. Tận dụng hệ thống **Tag (providesTags / invalidatesTags)** để hệ thống tự động refetch dữ liệu giao diện khi có thao tác Thêm/Sửa/Xóa thành công, loại bỏ hoàn toàn việc gọi lại API thủ công.

### Mẫu API chuẩn:
```typescript
import { baseApi } from "../../../stores/baseApi";

export const customFeatureApi = baseApi.injectEndpoints({
  endpoints: (builder) => ({
    getFeatures: builder.query<any, { params?: any; scope?: string }>({
      query: ({ params, scope }) => ({
        url: "/features",
        params: { ...params, scope },
      }),
      providesTags: (result) =>
        result?.data
          ? [
              ...result.data.map(({ _id }: any) => ({ type: "GlobalSetting" as const, id: _id })),
              { type: "GlobalSetting", id: "LIST" },
            ]
          : [{ type: "GlobalSetting", id: "LIST" }],
    }),

    createFeature: builder.mutation<any, any>({
      query: (body) => ({
        url: "/features",
        method: "POST",
        body,
      }),
      invalidatesTags: [{ type: "GlobalSetting", id: "LIST" }],
    }),
  }),
  overrideExisting: false,
});

export const { useGetFeaturesQuery, useCreateFeatureMutation } = customFeatureApi;
```

---

## 6. Styling Pattern & Design System

Dự án sử dụng chiến lược **Hybrid Styling**: Tailwind CSS cho bố cục (layout/spacing/flex/grid) và Vanilla CSS để tùy chỉnh chuyên sâu các lớp giao diện Ant Design nhằm mang lại trải nghiệm thị giác cao cấp (Premium Aesthetics).

### Quy chuẩn thiết kế (Design Tokens):
* **Màu chủ đạo (Primary)**: Màu xanh lam Antd được tinh chỉnh `#0a6ed1` kết hợp với sắc tím đặc trưng `primary-purple` (`#7c4dff` trong CSS hoặc `#A05AFF` trong cấu hình Tailwind).
* **Màu Gradient (Vibrant Gradients)**: Tích hợp sẵn các lớp nền gradient sang trọng cho thẻ thống kê:
  * `.bg-gradient-blue`: `linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)`
  * `.bg-gradient-green`: `linear-gradient(135deg, #065f46 0%, #10b981 100%)`
  * `.bg-gradient-purple`: `linear-gradient(135deg, #5b21b6 0%, #8b5cf6 100%)`
  * `.bg-gradient-red`: `linear-gradient(135deg, #991b1b 0%, #ef4444 100%)`
* **Giao diện thẻ (Glassmorphism & Shadows)**:
  * Thẻ bộ lọc/bảng sử dụng viền bo tròn lớn `rounded-[2rem]` kết hợp đổ bóng mịn `shadow-xl shadow-gray-200/40`.
  * Có sẵn lớp `.glass-card` tạo hiệu ứng kính mờ: `backdrop-filter: blur(10px); background: rgba(255, 255, 255, 0.7);`
* **Typography**: Sử dụng triệt để font chữ **Inter** (`font-family: "Inter", sans-serif`). Các tiêu đề sử dụng font-weight đậm (`font-bold`, `font-extrabold`, `font-black`).

### Chiến lược Hỗ trợ Chế độ Tối (Dark Mode Strategy):
Hệ thống chuyển đổi toàn diện thông qua class `dark` gán trên thẻ `<html>` (quản lý bởi `AppThemeProvider.tsx` và Redux `themeSlice`).
* **Lưu ý quan trọng**: Khi viết giao diện, luôn cung cấp cặp class Tailwind tương ứng, ví dụ: `bg-white dark:bg-[#232736] text-[#1a1a1a] dark:text-gray-200 border-gray-100 dark:border-gray-800`.

### Tùy biến Ant Design tiêu chuẩn:
* **Bảng dữ liệu (Table)**: Luôn gắn class `custom-table`. Bọc bảng bên trong `<Card className="rounded-[2rem] border-none shadow-xl shadow-gray-200/50 overflow-hidden" bodyStyle={{ padding: 0 }}>`.
* **Nút bấm (Button)**: Gắn class Tailwind định hình chiều cao và độ bo góc (VD: `h-10 px-6 rounded-xl font-bold`).
* **Thông báo cao cấp (Notification)**: Bọc cấu hình thông báo với class `premium-notification` để áp dụng bo góc 24px, hiệu ứng kính mờ (blur), và bóng đổ nổi bật.

---

## 7. TypeScript Rules

* **Ưu tiên sử dụng `interface`**: Sử dụng `interface` để định nghĩa cấu trúc dữ liệu Props, State, Payload. Chỉ dùng `type` cho các trường hợp Union types, Tuple, hoặc trích xuất kiểu từ đối tượng (như `keyof typeof`).
* **Định nghĩa Props tĩnh**: Tên interface Props luôn tuân theo mẫu `<ComponentName>Props`.
* **Kiểm tra an toàn (Optional Chaining)**: Luôn sử dụng toán tử `?.` khi truy xuất dữ liệu lồng nhau từ API trả về (VD: `response?.data?.record?._id`) để tránh ứng dụng bị crash.

---

## 8. Naming Convention

* **Thư mục (Folders)**: Chữ thường, số nhiều, định dạng kebab-case (VD: `employees`, `leave-requests`, `okrs`).
* **Tệp mã nguồn (Files)**:
  * Component / Trang: **PascalCase** `.tsx` (VD: `EmployeePage.tsx`).
  * Hook: Tiền tố `use` chuẩn **camelCase** (VD: `useDebounce.ts`, `usePermission.ts`).
  * Dịch vụ API: `<module>Api.ts` (VD: `employeeApi.ts`).
* **Hằng số toàn cục**: **UPPER_SNAKE_CASE** đóng băng bằng `as const` (VD: `COMMON_STATUS`, `ROLE_CODE`).

---

## 9. Reusable UI, Helper & Validation Rules

### 1. Phân quyền hiển thị và Scope (`AccessControl` & `usePermission`)
Không bao giờ tự so sánh chuỗi role để ẩn/hiện nút bấm. Hãy áp dụng:
* **Mức độ Component / Nút bấm**: Bọc trong component `<AccessControl permission={PERMISSIONS.MODULE.ACTION}>`.
* **Mức độ Logic & Lấy Scope**: Gọi hook:
```typescript
const { hasPermission, getPermissionScope } = usePermission();
const canViewAll = hasPermission(PERMISSIONS.EMPLOYEES.VIEW);
const viewScope = getPermissionScope(PERMISSIONS.EMPLOYEES.VIEW);
```
* **Mức độ Router**: Bọc trang bằng `<GuardRoute permission={PERMISSIONS.MODULE.VIEW}>`.

### 2. Thẻ bộ lọc tìm kiếm (`FilterCard` & `DebouncedSearchInput`)
* Bọc toàn bộ các trường lọc đầu trang bằng component `<FilterCard onRefresh={handleRefresh} loading={isLoading}>`.
* Ô nhập liệu tìm kiếm văn bản bắt buộc dùng `<DebouncedSearchInput>` (đã tích hợp sẵn cơ chế gõ trễ 500ms).

### 3. Quy chuẩn Validate Form (Ngăn chặn khoảng trắng)
Dự án áp dụng một bộ kiểm tra tĩnh thống nhất trên các Form.Item bắt buộc nhập văn bản để **ngăn chặn người dùng chỉ nhập toàn dấu cách (khoảng trắng)**:
```typescript
rules={[
  { required: true, message: "Vui lòng nhập trường này!" },
  {
    validator: (_, value) => {
      if (value && typeof value === "string" && value.trim() === "" && value.length > 0) {
        return Promise.reject("Không được chỉ nhập khoảng trắng");
      }
      return Promise.resolve();
    },
  },
]}
```

### 4. Quản lý Form (`formUtils`)
* `trimValues(values)`: Cắt bỏ khoảng trắng thừa ở tất cả các chuỗi trong form trước khi gửi API.
* `getChangedValues(values, initialValues)`: Chỉ lọc ra các trường thực sự bị thay đổi so với dữ liệu gốc để tối ưu payload PATCH.

### 5. Mã hóa Dữ liệu Truyền tải (`cryptoUtils.ts`)
* Đối với các luồng yêu cầu tính bảo mật cao cấp theo chuẩn của hệ thống, sử dụng hàm `encryptData(payload, secret)` để sinh chuỗi mã hóa AES-256-CBC an toàn trước khi đẩy qua đường mạng.

---

## 10. Mẫu Code Chuẩn Cho Các Thư Mục Cốt Lõi

### 1. `src/hooks/usePermission.ts`
```typescript
import { useLoginByTokenQuery } from "../modules/auth/services/authApi";
import { useMemo } from "react";

export const usePermission = () => {
  const { data: userData } = useLoginByTokenQuery(undefined);
  const permissions = useMemo(() => userData?.record?.data?.permissions || [], [userData]);
  
  const hasPermission = (permission: string) => {
    if (permissions.includes("*") || permissions.includes("all")) return true;
    return permissions.includes(permission);
  };

  return { hasPermission, permissions, userData };
};
```

### 2. `src/components/common/AccessControl.tsx`
```tsx
import React from "react";
import { Tooltip } from "antd";
import { usePermission } from "../../hooks/usePermission";

interface AccessControlProps {
  permission: string;
  children: React.ReactElement;
}

const AccessControl: React.FC<AccessControlProps> = ({ permission, children }) => {
  const { hasPermission } = usePermission();
  const allowed = hasPermission(permission);

  if (!allowed) {
    return (
      <Tooltip title="Bạn không có quyền thực hiện hành động này">
        <span className="opacity-60 cursor-not-allowed">
          {React.cloneElement(children, { 
            disabled: true, 
            onClick: (e: any) => e.preventDefault(),
            style: { ...children.props.style, pointerEvents: "none" }
          })}
        </span>
      </Tooltip>
    );
  }

  return children;
};

export default AccessControl;
```

### 3. `src/utils/formUtils.ts`
```typescript
/**
 * Trims all string values in an object recursively.
 */
export const trimValues = (values: any): any => {
  if (!values) return values;
  const result = Array.isArray(values) ? [...values] : { ...values };
  
  Object.keys(result).forEach((key) => {
    if (typeof result[key] === "string") {
      result[key] = result[key].trim();
    } else if (typeof result[key] === "object" && result[key] !== null) {
      result[key] = trimValues(result[key]);
    }
  });
  return result;
};

/**
 * Returns only the fields that have changed compared to initial values.
 */
export const getChangedValues = (values: any, initialValues: any) => {
  const changedValues: any = {};
  Object.keys(values).forEach((key) => {
    if (JSON.stringify(values[key]) !== JSON.stringify(initialValues[key])) {
      changedValues[key] = values[key];
    }
  });
  return changedValues;
};
```

### 4. `src/configs/socket.ts`
```typescript
import { io, Socket } from "socket.io-client";

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || "http://localhost:5885";

export const socket: Socket = io(SOCKET_URL, {
  autoConnect: false,
  transports: ["websocket"],
});

export const connectSocket = (token: string) => {
  if (socket.connected) socket.disconnect();
  socket.auth = { token };
  socket.connect();
};
```

---

## 11. How To Build New Feature Same Style

Khi nhận yêu cầu xây dựng một luồng tính năng mới (Ví dụ: **Quản lý Tài sản - Asset Management**), hãy thực hiện quy trình chuẩn sau:

### Bước 1: Khai báo Hằng số và Quyền
Mở `src/constants/permissions.ts` và thêm khối quyền mới:
```typescript
ASSETS: {
  VIEW: "assets.view",
  CREATE: "assets.create",
  UPDATE: "assets.update",
  DELETE: "assets.delete",
},
```

### Bước 2: Thiết lập Service Layer (API)
Tạo `src/modules/assets/services/assetApi.ts`:
* Import `baseApi` và khai báo các luồng `getAssets`, `createAsset`, `updateAsset`, `deleteAsset`.
* Đăng ký chuỗi Tag `"Asset"` mới vào mảng `tagTypes` bên trong tệp `src/stores/baseApi.ts`.

### Bước 3: Xây dựng Giao diện dùng chung / Component phụ trợ
Tạo tệp `src/modules/assets/components/AssetModal.tsx` để xử lý form:
* Tích hợp quy tắc `validator` chống khoảng trắng cho các ô Input.
* Gọi `trimValues` và `getChangedValues` khi submit.

### Bước 4: Hoàn thiện Trang Giao diện Chính
Tạo tệp `src/modules/assets/pages/AssetPage.tsx`:
* Xây dựng phần Header trang chứa Breadcrumb, Tiêu đề và nút Thêm mới (bọc trong `AccessControl`).
* Thiết lập mảng `columns` (bọc trong `useMemo`), bọc bảng `<Table>` bên trong thẻ `<Card className="rounded-[2rem] border-none shadow-xl shadow-gray-200/50 overflow-hidden" bodyStyle={{ padding: 0 }}>`.
* Truyền cấu hình phân trang đồng bộ với `PAGINATION_CONFIG`.

### Bước 5: Đăng ký Định tuyến (Routing)
Mở `src/routers/AppRouter.tsx`:
```tsx
const AssetPage = lazy(() => import("../modules/assets/pages/AssetPage"));

<Route
  path="/assets"
  element={
    <GuardRoute permission={PERMISSIONS.ASSETS.VIEW}>
      <AssetPage />
    </GuardRoute>
  }
/>
```

---

## 12. Do / Don't (Quy tắc Sống còn)

### DO (BẮT BUỘC LÀM)
* **DO** tuân thủ chặt chẽ nguyên tắc **Tự động làm mới giao diện qua Cache Tags** của RTK Query.
* **DO** bọc bảng dữ liệu chính bằng thẻ `Card` Ant Design với cấu trúc viền bo tròn cực đại `rounded-[2rem]` và bóng đổ cao cấp.
* **DO** áp dụng quy chuẩn kiểm tra hợp lệ (validator) chống nhập nguyên khoảng trắng trên toàn bộ các biểu mẫu nhập liệu.
* **DO** luôn hỗ trợ class `dark:...` đi kèm với mọi class cấu hình màu sắc trong Tailwind CSS.
* **DO** sử dụng lệnh quản lý gói bằng **Yarn** (`yarn add`, `yarn dev`, `yarn build`) thay vì `npm` để đồng bộ hoàn toàn với file khóa `yarn.lock` hiện hữu.

### DON'T (NGHIÊM CẤM LÀM)
* **DON'T** sử dụng lệnh thao tác DOM trực tiếp (như `document.getElementById` hoặc `querySelector`).
* **DON'T** gán class CSS tùy tiện mang tính phá vỡ toàn cục.
* **DON'T** sử dụng thẻ `<a>` thuần túy để chuyển trang nội bộ, bắt buộc sử dụng component `<Link>` hoặc hook `useNavigate`.
* **DON'T** lưu trữ Token trong State tạm thời mà hãy lưu trực tiếp vào `localStorage` ngay khi nhận phản hồi đăng nhập thành công.
* **DON'T** cài đặt gói mới bằng `npm install` tránh tạo ra tệp `package-lock.json` gây xung đột cơ sở dữ liệu cây phụ thuộc với `yarn`.

---

## 13. Clone This Style For Future Projects

Khi khởi tạo một Frontend hoàn toàn mới dựa trên DNA của AHV HRM, hãy tuân theo các bước thiết lập kiến trúc nền tảng sau:

1. **Khởi tạo dự án gốc**:
   ```bash
   yarn create vite my-new-app --template react-ts
   ```
2. **Cài đặt hệ sinh thái phụ thuộc lõi**:
   ```bash
   yarn add antd @ant-design/icons react-icons @reduxjs/toolkit react-redux react-router-dom crypto-js clsx tailwind-merge
   yarn add -D tailwindcss postcss autoprefixer sass prettier eslint @types/crypto-js
   ```
3. **Cấu hình Quy chuẩn Format và Styling**:
   * Áp dụng trọn vẹn tệp cấu hình `.prettierrc` (double quotes, semi true, printWidth 120, crlf).
   * Cấu hình tệp `eslint.config.js` tắt cảnh báo `any` và `exhaustive-deps`.
   * Nhân bản trọn vẹn tệp `src/index.css` để giữ lại DNA thẩm mỹ bo góc, nút bấm, và Dark mode.
4. **Sao chép Lớp lõi (Core Layers)**:
   * Bê nguyên vẹn thư mục `src/components/common` và `src/components/ui`.
   * Mang theo tệp quản lý API gốc `src/stores/baseApi.ts` và cơ chế Provider Sáng/Tối.
   * Áp dụng trực tiếp tệp tiện ích `src/utils/notificationUtils.ts`, `src/utils/formUtils.ts`, và `src/utils/cryptoUtils.ts`.
5. **Phát triển Module mới**: Áp dụng quy chuẩn phát triển tách biệt theo từng Feature Module như đã hướng dẫn trong tài liệu này.

Mọi dòng mã được phát sinh sau này khi bám sát tuyệt đối cẩm nang **SkillFE.md** sẽ đảm bảo sản phẩm đạt chất lượng cao nhất, cấu trúc đồng nhất tuyệt đối và sẵn sàng mở rộng không giới hạn!

---

# 14. Bản Thiết Kế Tái Tạo Vũ Trụ (Core Boilerplate DNA)

Tài liệu này là "phần xác" bù đắp cho "phần hồn" của các quy chuẩn ở trên. Nó chứa mã nguồn gốc của tất cả các file cấu hình cốt lõi, CSS toàn cục, và logic phức tạp nhất của dự án. 
Khi kết hợp phần này cùng các quy chuẩn trên, một AI Agent có thể tái hiện chính xác 100% dự án từ con số không.

## 14.1. Môi trường & Cấu hình lõi (Environment & Configs)

### `.env.example`
```env
VITE_API_URL=http://localhost:5857/api/v1
VITE_SOCKET_URL=http://localhost:5885
VITE_SECRET_KEY=your_secret_key_here
VITE_PORT=3000
VITE_ALLOWED_HOSTS=localhost,127.0.0.1
```

### `vite.config.ts`
Quy định cách chia nhỏ file khi build (manualChunks) để tối ưu tải trang và thiết lập đường dẫn tắt.
```typescript
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const PORT = env.VITE_PORT;
  const allowedHosts = env.VITE_ALLOWED_HOSTS?.split(",").map((h) => h.trim()) || [];
  return {
    plugins: [react()],
    resolve: {
      alias: {
        bootstrap: path.resolve(__dirname, "node_modules/bootstrap"),
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: (id) => {
            if (id.includes("node_modules")) {
              if (id.includes("antd")) return "antd";
              if (id.includes("@ant-design/icons")) return "antd-icons";
              if (id.includes("@tiptap")) return "tiptap";
              if (id.includes("react") || id.includes("react-dom") || id.includes("react-router")) return "react-vendor";
              return "vendor";
            }
          },
        },
      },
      chunkSizeWarningLimit: 1000,
    },
    server: {
      port: PORT ? Number(PORT) : undefined,
      host: "0.0.0.0",
      allowedHosts: allowedHosts,
    },
  };
});
```

### `tailwind.config.js`
Bộ gen nền tảng Tailwind. LƯU Ý: Các giá trị màu sắc (`colors`), gradients (`backgroundImage`) cụ thể sẽ được định nghĩa lại theo thiết kế của dự án mới. Ở đây chỉ giữ lại cấu trúc lõi để hỗ trợ Dark Mode và quét file.
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      // Khai báo các mã màu, font chữ, animation đặc thù cho dự án mới tại đây
    },
  },
  plugins: [],
}
```

---

## 14.2. API & Quản lý State Lõi (Core API & State)

### `src/stores/baseApi.ts`
Chứa logic **Mutex Lock** cho luồng Refresh Token, giải quyết bài toán chống gọi API nhiều lần khi Token hết hạn và tự động sửa lỗi mã hóa URL cho ISO Date.
```typescript
import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import type { BaseQueryFn, FetchArgs, FetchBaseQueryError } from "@reduxjs/toolkit/query/react";

const rawBaseQuery = fetchBaseQuery({
  baseUrl: import.meta.env.VITE_API_URL || "http://localhost:5857/api/v1",
  prepareHeaders: (headers) => {
    const token = localStorage.getItem("token");
    if (token) headers.set("authorization", `Bearer ${token}`);
    headers.set("Content-Type", "application/json");
    return headers;
  },
  credentials: "include",
});

const baseQueryWithReformatting: BaseQueryFn<string | FetchArgs, unknown, FetchBaseQueryError> = async (
  args, api, extraOptions
) => {
  if (typeof args !== "string" && args.params) {
    const { url, params, ...rest } = args;
    const queryParams = new URLSearchParams();

    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) queryParams.append(key, String(value));
    });

    const queryString = queryParams.toString().replace(/%3A/g, ":");
    const newUrl = queryString ? `${url}${url.includes("?") ? "&" : "?"}${queryString}` : url;
    return rawBaseQuery({ ...rest, url: newUrl }, api, extraOptions);
  }
  return rawBaseQuery(args, api, extraOptions);
};

let refreshPromise: Promise<string | null> | null = null;

const baseQueryWithReauth: BaseQueryFn<string | FetchArgs, unknown, FetchBaseQueryError> = async (
  args, api, extraOptions
) => {
  let result = await baseQueryWithReformatting(args, api, extraOptions);

  if (result.error && result.error.status === 401) {
    if (!refreshPromise) {
      refreshPromise = (async () => {
        try {
          const refreshResult: any = await rawBaseQuery(
            { url: "/auth/refresh-token", method: "POST" }, api, extraOptions
          );

          if (refreshResult.data) {
            const accessToken = refreshResult.data.data.accessToken;
            localStorage.setItem("token", accessToken);
            return accessToken;
          } else {
            localStorage.removeItem("token");
            window.location.href = "/login";
            return null;
          }
        } catch (error) {
          localStorage.removeItem("token");
          window.location.href = "/login";
          return null;
        } finally {
          refreshPromise = null;
        }
      })();
    }

    const token = await refreshPromise;
    if (token) result = await baseQueryWithReformatting(args, api, extraOptions);
  }
  return result;
};

export const baseApi = createApi({
  reducerPath: "api",
  baseQuery: baseQueryWithReauth,
  tagTypes: ["Auth", "GlobalSetting", "LIST"], // Thêm các tag khác tùy module
  endpoints: () => ({}),
});
```

---

## 14.3. Core UI Components

### `src/components/ui/FilterCard.tsx`
Thẻ bộ lọc hiển thị cấp cao, thiết kế chuẩn kính mờ bo tròn (glassmorphism).
```tsx
import React from "react";
import { Card, Button } from "antd";
import { ReloadOutlined } from "@ant-design/icons";

interface FilterCardProps {
  children: React.ReactNode;
  onRefresh: () => void;
  loading?: boolean;
}

export const FilterItem: React.FC<{ label: string; children: React.ReactNode; width?: string | number }> = ({
  label, children, width,
}) => (
  <div style={{ width: width || "auto", minWidth: "200px" }} className="flex-1">
    <div className="text-[10px] font-black uppercase text-gray-400 mb-2 tracking-wider ml-1">{label}</div>
    {children}
  </div>
);

const FilterCard: React.FC<FilterCardProps> = ({ children, onRefresh, loading }) => (
  <Card
    className="mb-8 rounded-[2rem] border-t-[6px] border-t-primary-purple shadow-xl shadow-gray-200/40 overflow-hidden border-none"
    bodyStyle={{ padding: "24px 32px" }}
  >
    <div className="flex flex-wrap items-end gap-8">
      <div className="flex-1 flex flex-wrap items-end gap-8">{children}</div>
      <div className="pb-[2px]">
        <Button
          icon={<ReloadOutlined className="text-orange-500" />}
          onClick={onRefresh} loading={loading}
          className="h-11 px-6 rounded-2xl flex items-center gap-2 font-extrabold text-orange-500 border-2 border-orange-100 bg-orange-50/30 hover:bg-orange-50 hover:border-orange-200 transition-all shadow-sm"
        >
          Làm mới
        </Button>
      </div>
    </div>
  </Card>
);

export default FilterCard;
```

### `src/components/ui/DebouncedSearchInput.tsx`
Chống gửi request liên tục khi người dùng gõ phím.
```tsx
import React, { useState, useEffect } from "react";
import { Input } from "antd";
import type { InputProps } from "antd";

interface DebouncedSearchInputProps extends Omit<InputProps, "onChange" | "value"> {
  value: string;
  onChange: (value: string) => void;
  debounceTimeout?: number;
}

const DebouncedSearchInput: React.FC<DebouncedSearchInputProps> = ({
  value, onChange, debounceTimeout = 500, ...props
}) => {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => { setLocalValue(value); }, [value]);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (localValue !== value) onChange(localValue);
    }, debounceTimeout);
    return () => clearTimeout(timer);
  }, [localValue, onChange, debounceTimeout, value]);

  return <Input {...props} value={localValue} onChange={(e) => setLocalValue(e.target.value)} />;
};

export default DebouncedSearchInput;
```

---

## 14.4. Hệ mã hóa bảo mật (Crypto Utils)

### `src/utils/cryptoUtils.ts`
Chịu trách nhiệm mã hóa payload khớp với thuật toán giải mã của Backend.
```typescript
import CryptoJS from "crypto-js";

/**
 * Mã hóa payload bằng thuật toán AES-256-CBC
 * Trả về chuỗi định dạng: "iv_hex:encrypted_content_hex"
 */
export const encryptData = (data: any, secret: string): string => {
  const key = CryptoJS.SHA256(secret);
  const iv = CryptoJS.lib.WordArray.random(16);
  const dataToEncrypt = typeof data === "string" ? data : JSON.stringify(data);

  const encrypted = CryptoJS.AES.encrypt(dataToEncrypt, key, {
    iv: iv,
    mode: CryptoJS.mode.CBC,
    padding: CryptoJS.pad.Pkcs7,
  });

  return iv.toString(CryptoJS.enc.Hex) + ":" + encrypted.ciphertext.toString(CryptoJS.enc.Hex);
};
```

---

## 14.5. CSS Toàn Cục (Global Styles)

### `src/index.css` (Lược trích các thành phần cốt lõi)
Định nghĩa cấu trúc nền tảng (Scrollbar, Dark Mode, Table UI). LƯU Ý: Các mã màu nút bấm (Primary Button), màu nền riêng biệt sẽ do dự án mới tự định nghĩa.
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Scrollbar styling */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #cbd5e1; }

/* Dark Mode Global Overrides */
html.dark body { background: #1a1d27 !important; color: #ffffff !important; }
html.dark .bg-white { background-color: #232736 !important; color: #ffffff !important; }
html.dark .text-gray-900, html.dark .text-gray-800 { color: #f3f4f6 !important; }

html.dark .ant-select-selector, html.dark .ant-input, html.dark .ant-picker {
  background-color: #1a1d27 !important;
  border-color: #374151 !important;
  color: #ffffff !important;
}

/* Custom Table Premium Framework */
.custom-table .ant-table-thead > tr > th {
  background: #f8fafc !important;
  color: #64748b !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  font-size: 11px;
}

html.dark .custom-table .ant-table-thead > tr > th {
  background: #1a1d27 !important;
  color: #94a3b8 !important;
  border-bottom: 1px solid #374151 !important;
}

/* Glassmorphism Framework */
.glass-card {
  background: rgba(255, 255, 255, 0.7) !important;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.4) !important;
}
```

---

## 14.6. Cơ chế Bẫy lỗi & Tải trang (Resilience & UX)

Để hệ thống đạt chuẩn Enterprise 100%, không bao giờ được để ứng dụng chết trắng màn hình khi có lỗi logic. Dưới đây là 3 mảnh ghép sinh tử cuối cùng:

### `src/components/common/ErrorBoundary.tsx`
Bẫy mọi lỗi Crash UI (ví dụ: `undefined.map()`) để hiển thị giao diện báo lỗi thân thiện thay vì làm sập toàn bộ ứng dụng.
```tsx
import React, { Component, ErrorInfo, ReactNode } from "react";
import { Result, Button } from "antd";

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
    // Tại đây có thể đẩy log lỗi lên Sentry hoặc hệ thống theo dõi của Backend
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-[#1a1d27]">
          <Result
            status="500"
            title={<span className="text-gray-800 dark:text-gray-100 font-bold">Đã có lỗi không mong muốn xảy ra</span>}
            subTitle={<span className="text-gray-500 dark:text-gray-400">Giao diện gặp sự cố khi hiển thị. Vui lòng thử lại.</span>}
            extra={
              <Button type="primary" onClick={() => window.location.reload()} className="h-10 px-6 rounded-xl font-bold">
                Tải lại trang
              </Button>
            }
          />
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
```

### `src/components/common/PageLoading.tsx`
Màn hình chờ (Fallback) chuẩn mực khi sử dụng `React.lazy` (Lazy Load) để chuyển module.
```tsx
import React from "react";
import { Spin } from "antd";
import { LoadingOutlined } from "@ant-design/icons";

const PageLoading: React.FC = () => {
  const antIcon = <LoadingOutlined style={{ fontSize: 48 }} spin className="text-primary-purple" />;

  return (
    <div className="flex flex-col items-center justify-center w-full h-[80vh]">
      <Spin indicator={antIcon} />
      <p className="mt-4 text-gray-500 dark:text-gray-400 font-medium animate-pulse">
        Đang tải dữ liệu...
      </p>
    </div>
  );
};

export default PageLoading;
```

### `src/routers/AppRouter.tsx` (Core Router Pattern)
Cách kết hợp `ErrorBoundary`, `Suspense` và `Lazy Load` để tạo ra bộ định tuyến (Router) vững như bàn thạch.
```tsx
import React, { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";
import PageLoading from "../components/common/PageLoading";
import ErrorBoundary from "../components/common/ErrorBoundary";

// Lazy load các trang để tối ưu dung lượng (Code Splitting)
const DashboardPage = lazy(() => import("../modules/dashboard/pages/DashboardPage"));

const AppRouter: React.FC = () => {
  return (
    <ErrorBoundary>
      <Suspense fallback={<PageLoading />}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          {/* Các Route khác định nghĩa tại đây */}
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
};

export default AppRouter;
```
