# Backend Skill Memory (BE DNA & Coding Standard)

Tài liệu đúc kết trực tiếp từ mã nguồn **AHV HRM Backend** (`ahv-hrm-backend`), dùng làm kim chỉ nam để AI Agent hoặc lập trình viên xây dựng tính năng mới hoặc nhân bản phong cách sang dự án khác.

---

## 1. Project DNA

| Thành phần | Chi tiết |
|---|---|
| **Runtime** | Node.js 20 (alpine) |
| **Framework** | Express 5 (`^5.2.1`) |
| **Ngôn ngữ** | TypeScript (`^5.9.3`) — module `commonjs`, target `ESNext` |
| **Package Manager** | **Yarn** (file `yarn.lock` hiện hữu) |
| **Dev Runner** | `ts-node-dev` với `tsconfig-paths/register` |
| **Database** | MongoDB (Mongoose `^9.1.5`), hỗ trợ Replica Set & Transactions |
| **Cache/Realtime** | Redis (`ioredis ^5.10.1`), Socket.IO (`^4.8.3`) |
| **Auth** | JWT (`jsonwebtoken`), bcrypt, token-version revocation |
| **Validation** | Joi (`^18.0.2`) |
| **Logging** | Winston + winston-daily-rotate-file (ghi ra `logs/`) |
| **Cron Jobs** | `node-cron ^4.2.1` qua `cronManager` |
| **Email** | `nodemailer ^8.0.7` (SMTP Gmail) |
| **File Upload** | `multer ^2.0.2` (disk storage → `./uploads`) |
| **API Docs** | Swagger UI Express đọc file `swagger.yaml` bằng `yamljs` tại route `/docs` |
| **Encryption** | `crypto-js` (AES-256-CBC) khớp với Frontend |
| **Báo cáo/Thời gian**| `exceljs ^4.4.0` (xuất Excel), `moment-timezone ^0.6.0` |
| **Build** | `tsc` + `tscpaths` resolve alias → output `dist/` |

---

## 2. Folder Structure Standard

```text
src/
├── app.ts                  # Khởi tạo Express app, middleware chain, mount router
├── server.ts               # Bootstrap: connect DB → Redis → HTTP → Socket → Cron → Seed
├── common/                 # LỚP CHIA SẺ TOÀN HỆ THỐNG
│   ├── config/             # environment.ts, database.ts, redis.ts, permissionRoutes.ts
│   ├── helpers/            # permission.helper.ts, ...
│   ├── interfaces/         # 43+ file interface (response, employee, okrs, payroll...)
│   ├── middlewares/        # authen, author, error, validate, trim
│   ├── migration/          # initSeedData.ts (seed dữ liệu mẫu)
│   ├── routes/             # index.ts — Bảng đăng ký tập trung toàn bộ module routes
│   └── utils/              # ApiError, catchAsync, logger, token, hashPassword, mail, enum...
├── modules/                # 43 FEATURE MODULES (mỗi module = 1 thư mục)
│   ├── employees/          # Mẫu chuẩn: *.route, *.controller, *.service, *.model, *.validation
│   ├── auth/
│   ├── okrs/
│   └── ...
├── socket/                 # socket.service.ts — WebSocket server
└── worker/                 # Tác vụ nền
    ├── cron/               # cron.manager.ts — đăng ký & quản lý cron jobs
    └── startup/            # startup.manager.ts — tác vụ chạy một lần khi khởi động
```

### Cấu trúc bên trong một Module:
```text
modules/<module-name>/
├── <module>.route.ts           # Khai báo Router, gắn validate + authorizationScope middleware
├── <module>.controller.ts      # Export named functions, dùng catchAsync bọc async
├── <module>.service.ts         # Class chứa business logic, gọi Model trực tiếp
├── <module>.model.ts           # Mongoose Schema + indexes + export Model
├── <module>.validation.ts      # Joi schemas (create + update riêng)
└── helper/                     # (Tùy chọn) Logic phụ trợ riêng module
```

---

## 3. Quy chuẩn Format & Linter

### Prettier (`.prettierrc`)
| Quy tắc | Giá trị |
|---|---|
| Quotes | **Single quotes** (`singleQuote: true`) |
| Semicolons | Có (`semi: true`) |
| Trailing Comma | Mọi nơi (`trailingComma: "all"`) |
| Print Width | **120** ký tự |
| Tab Width | **4 spaces** |
| End of Line | **LF** (`endOfLine: "lf"`) |

### Path Aliases (`tsconfig.json`)
```text
@modules/*   → src/modules/*
@common/*    → src/common/*
@config/*    → src/common/config/*
@socket/*    → src/socket/*
@worker/*    → src/worker/*
```
**Quy tắc**: Luôn dùng alias `@modules/`, `@common/` khi import xuyên module. Chỉ dùng relative path (`./`, `../`) khi import **trong cùng module**.

---

## 4. API Response Rules

Toàn bộ API trả về JSON theo cấu trúc cố định (từ `response.interface.ts`):

### Response đơn lẻ (`IApiResponse<T>`):
```json
{
  "success": true,
  "message": "Tạo nhân viên thành công",
  "data": { ... }
}
```

### Response phân trang (`IPaginatedResponse<T>`):
```json
{
  "success": true,
  "message": "Lấy danh sách nhân viên thành công",
  "data": [ ... ],
  "meta": {
    "totalItems": 150,
    "itemCount": 10,
    "itemsPerPage": 10,
    "totalPages": 15,
    "currentPage": 1
  }
}
```

### Response lỗi:
```json
{
  "success": false,
  "status": 400,
  "message": "Email của nhân viên đã tồn tại"
}
```

---

## 5. Controller Pattern

* Export **named functions** (không dùng class).
* Mỗi handler bọc bằng `catchAsync()` để tự động bắt lỗi async.
* Khởi tạo service instances **ở đầu file** (module-level).
* Controller chỉ làm 3 việc: **parse request → gọi service → trả response**.
* **Đặc biệt quan trọng (Xử lý Phân Quyền/Scope)**: Controller có trách nhiệm lớn trong việc bóc tách `req.users.filter` (đối tượng Scope do `authorizationPermissionScopeMiddleware` sinh ra) và gộp vào tham số hoặc đẩy thẳng xuống Service để khóa chặn phạm vi dữ liệu (tránh lấy dư dữ liệu của phòng ban khác hoặc nhân viên khác khi không có quyền).

```typescript
import { catchAsync } from '@common/utils/catchAsync';
import { Request, Response } from 'express';
import { FeatureService } from './feature.service';
import { Types } from 'mongoose';

const featureService = new FeatureService();

export const createFeature = catchAsync(async (req: Request, res: Response) => {
    const result = await featureService.create(req.body);
    res.status(201).json(result);
});

export const getFeatures = catchAsync(async (req: Request, res: Response) => {
    const filter = {
        page: req.query.page ? parseInt(String(req.query.page), 10) : 1,
        limit: req.query.limit ? parseInt(String(req.query.limit), 10) : 10,
        // Ép kiểu ID và Ưu tiên scope phân quyền nếu client không gửi param
        employeeId:
            typeof req.query.employeeId === "string" && Types.ObjectId.isValid(req.query.employeeId)
                ? new Types.ObjectId(req.query.employeeId)
                : req.users?.filter?.employeeId,
    };
    const result = await featureService.getByFilter(filter);
    res.status(200).json(result);
});

export const updateFeature = catchAsync(async (req: Request, res: Response) => {
    // Luôn truyền req.users.filter xuống service để tự động ghép điều kiện
    const result = await featureService.updateFeature(String(req.params.id), req.body, req.users.filter);
    res.status(200).json(result);
});
```

---

## 6. Service Pattern

* Sử dụng **Class** với các method public.
* Trả về `IApiResponse<T>` hoặc `IPaginatedResponse<T>`.
* Ném lỗi nghiệp vụ bằng `throw new ApiError(statusCode, message)`.
* **Phân trang chuyên sâu**: Dùng MongoDB Aggregation Pipeline (`$match`, `$skip`, `$limit`, `$lookup`) kết hợp `Promise.all([Model.aggregate(pipeline), Model.aggregate(count)])` để tối đa hiệu năng tìm kiếm đồng thời với đếm số lượng (thay vì dùng `Model.find()` hai lần).
* **Kiểm soát Truy vấn**: Luôn ép kiểu các object filter với Mongoose `QueryFilter<T>` để TS cảnh báo sai type.
* **Bảo vệ quyền (Scope)**: Tại hàm Update/Delete, luôn kiểm tra tham số `scope?: IRequestScope`, nếu có `scope.employeeId` hoặc `scope.departmentId`, phải chèn vào filter của hàm cập nhật.

---

## 7. Validation Pattern (Joi)

* Tạo file `<module>.validation.ts`, export `createSchema` và `updateSchema` riêng biệt.
* Thông báo lỗi bằng **tiếng Việt**.
* Gắn vào route qua middleware `validate(schema)` (strip unknown, abort early = false).
* **Lưu ý Cực Kỳ Quan Trọng**: Khi gọi `stripUnknown: true`, Joi sẽ vứt bỏ các biến lạ. Tuy nhiên, middleware `validate` (đã custom ở common) được thiết kế đặc biệt để LUÔN GIỮ LẠI `employeeId` và `departmentId` (nếu có) trên `req.body` nhằm cung cấp cho chuỗi phân quyền ở các bước kế tiếp.

```typescript
import Joi from 'joi';

export const createFeatureSchema = Joi.object({
    name: Joi.string().required().messages({
        'string.empty': 'Tên không được để trống',
        'any.required': 'Tên là bắt buộc',
    }),
    status: Joi.string().valid('active', 'inactive').optional(),
});
```

---

## 8. Database Pattern (Mongoose)

* Schema dùng `new Schema({...}, { timestamps: { createdAt, updatedAt } })`.
* Khai báo index rõ ràng với `name` và `background: true`.
* Export default `mongoose.model<Interface>('CollectionName', Schema)`.
* Quan hệ dùng `Types.ObjectId` + `ref` + `$lookup` hoặc `.populate()`.
* **MongoDB Transactions**: BẮT BUỘC sử dụng khi lưu trữ/cập nhật qua nhiều bảng đồng thời (vd: Cập nhật Trạng thái Nhân viên -> Tự cập nhật OKR & Hợp đồng).
  Dùng `mongoose.startSession()`, bọc logic trong `session.withTransaction()`. Nhớ truyền cờ `{ session }` vào đuôi của tất cả các lệnh queries bên trong transaction.

---

## 9. Auth, Permission, Caching & Realtime Patterns (Nâng cao)

Hệ thống bảo mật và phân quyền của dự án được thiết kế chặt chẽ, đa lớp và tối ưu hóa cao độ bằng bộ nhớ đệm (Redis):

### 1. Middleware Chain chính (đăng ký tại `common/routes/index.ts`)
```text
authenticationMiddleware → authorizationPermissionMiddleware → authorizationPermissionScopeMiddleware → Route Handler
```
* **`authenticationMiddleware`**: Xác thực JWT token, đối chiếu `tokenVersion` trong DB (nếu user đổi mật khẩu/bị khóa, `tokenVersion` thay đổi → lập tức trả về 401). Tải thông tin tài khoản và Role name gắn vào đối tượng `req.users`.
* **`authorizationPermissionMiddleware`**:
  - Không truy vấn DB để kiểm tra quyền. Tận dụng tập quyền đã được nạp sẵn trên **Redis Cache** qua hàm `permissionService.getPermissionsByRoleNameInCacheOfDB(roleName)`.
  - Tự động quét bản đồ tuyến tĩnh (`permissionRoutes` tại `src/common/config/permissionRoutes.ts`) để so khớp `req.originalUrl` và `req.method` bằng Regex. Từ đó tự suy luận chuỗi hành động cần thiết (Ví dụ: `payrolls.view`, `leaveRequests.approve`) và kiểm tra xem có nằm trong tập quyền của user hay không.
  - Hỗ trợ ký tự đại diện wildcard (`*`, `module.*`, `*.*`). Lấy ra Scope cụ thể gắn vào `req.users.scope`.
* **`authorizationPermissionScopeMiddleware`**: Xử lý logic truy cập theo phạm vi (`self`, `department`, `*`):
  - `SELF`: Tự động ép buộc/kiểm tra `req.body.employeeId` / `req.query.employeeId` phải trùng khớp với ID của user đang đăng nhập.
  - `DEPARTMENT`: Tự động giới hạn truy cập trong phạm vi `departmentId` của user.
  - Gắn điều kiện truy vấn chung vào `req.users.filter` để Service tự động apply vào Mongoose query.
* **Các Guard đặc thù khác**: Cung cấp sẵn `verifyAdmin` và `verifyAdminOrOwner` (tại `author.middleware.ts`) dùng để cho phép Admin, nhân sự phòng ban BOF (Backoffice/BOD) hoặc chính chủ được phép thực thi.

### 2. Cơ chế Caching Phân quyền (Permissions Memcache lúc Startup)
* Khi server khởi động, `startupManager` (tại `src/worker/startup/startup.manager.ts`) sẽ kích hoạt tác vụ `PermissonsMemcache`.
* Tác vụ này quét toàn bộ collection `permissions` trong DB, chuyển đổi thành danh sách các chuỗi quyền theo từng Role, sau đó ghi trực tiếp vào **Redis Set** (lệnh `sadd`) với từ khóa `permissions:<role_name>` (tiền tố mặc định của ioredis client là `ahvwork::`).
* Khi có sự thay đổi quyền từ phía Admin, hệ thống chỉ cần cập nhật lại key tương ứng trong Redis.

### 3. Quy tắc Đặt tên Key & Idempotency trong Redis
* **Prefix chung**: Mọi thao tác qua `RedisAdapter` đều tự động mang tiền tố `ahvwork::`.
* **Idempotency (Chống lặp thao tác)**: Để tránh gửi lặp hoặc trừ tiền/sao hai lần (như đơn hàng Niko), dự án áp dụng khóa Idempotency sinh bằng `genNikoOrderIdempotencyKey(employeeId, idempotencyKey)`. Sử dụng lệnh `RedisAdapter.setnx` kết hợp với các trạng thái xử lý trong enum `INikoOrderIdempotencyStatus` (`processing`, `completed`, `failed`).

### 4. Cơ chế Mã hóa/Giải mã Payload (Crypto Layer)
* Các dữ liệu nhạy cảm truyền tải giữa client và server sử dụng thuật toán mã hóa đối xứng **AES-256-CBC** (tại `src/common/utils/crypto.utils.ts`).
* **Format đầu vào**: `iv:encryptedContent` (chuỗi hex).
* **Băm khóa**: Secret key dùng chung được băm qua `sha256` để cho ra chìa khóa 256-bit chuẩn xác, đảm bảo tương thích hai chiều hoàn hảo với lớp `cryptoUtils` dưới Frontend.

### 5. Xác thực & Quản lý Realtime (Socket.IO)
* Socket server (`src/socket/socket.service.ts`) gắn middleware `authenticationSocket` để tự động kiểm tra token qua `socket.handshake.auth.token` hoặc `socket.handshake.query.token`.
* **Quy tắc Room tự động**: User kết nối thành công lập tức được join vào 2 loại room:
  - Room cá nhân: `u_<employeeId>`
  - Room theo Role: `role_<ROLE_CODE>` (Mã Role viết hoa toàn bộ).
* **Hàm tiện ích broadcast**: Hỗ trợ gọi nhanh `socketService.sendNotification(accountId, event, data)` và `socketService.sendToRole(roleCode, event, data)`.

---

## 10. Middleware Rules

| Middleware | Chức năng |
|---|---|
| `trimRequest` | Tự động trim tất cả string trong `req.body` trước khi xử lý |
| `validate(schema)` | Joi validation, strip unknown fields, abort early = false |
| `errorHandler` | Catch cuối cùng: `ApiError` → trả status code + message, lỗi khác → 500 |
| `catchAsync(fn)` | Bọc async handler, tự động forward lỗi vào `next()` |

---

## 11. Error Handling Rules

* **Luôn** sử dụng `throw new ApiError(statusCode, 'Thông báo tiếng Việt')` trong Service.
* Không bao giờ `res.status().json()` trực tiếp từ catch block — để `errorHandler` xử lý.
* Unhandled errors (uncaughtException, unhandledRejection) → log bằng Winston → shutdown gracefully.

---

## 12. Logging Rules

* Logger: **Winston** + **DailyRotateFile** (ghi vào `logs/error-YYYY-MM-DD.log` và `logs/combined-YYYY-MM-DD.log`).
* Console log có màu sắc (colorize). File log không màu (uncolorize).
* Development: level `debug`. Production: level `warn`.
* Sử dụng: `logger.info(msg)`, `logger.error(msg, error)`, `logger.warn(msg)`.

---

## 13. Docker Standard

### Dockerfile (Multi-stage build):
```dockerfile
FROM node:20-alpine AS builder   # Stage 1: Build
WORKDIR /app
RUN apk --no-cache add g++ make python3 git && yarn global add node-gyp@9.4.0
ADD package.json yarn.lock /app/
RUN yarn --pure-lockfile
ADD . /app
RUN yarn build
RUN yarn --pure-lockfile --prod

FROM node:20-alpine              # Stage 2: Runtime
WORKDIR /app
RUN apk --no-cache add curl
COPY --from=builder /app .
CMD ["yarn", "start"]
```

### docker-compose.yml:
* Container: `ahv-hrm-be`, restart always.
* Health check: `curl --fail http://127.0.0.1:$PORT/health`.
* Logging: json-file, max 10m × 5 files.
* Network: external shared network.

---

## 14. CI/CD Standard (GitLab CI)

### Pipeline: `verify → build → deploy`
| Stage | Job | Trigger |
|---|---|---|
| **verify** | `check_lint` (yarn lint) | Mọi MR + push develop/production |
| **build** | `build_staging` / `build_production` | Docker build + push registry |
| **deploy** | `deploy_staging` (auto) / `deploy_production` (manual) | SSH → VPS → docker-compose up |

### Branch Strategy:
* `develop` → auto deploy staging.
* `production` → **manual** deploy production.

---

## 15. Environment Rules

Toàn bộ biến môi trường tập trung tại `src/common/config/environment.ts`, export dạng hằng số typed:
```typescript
export const PORT: number = parseInt(process.env.PORT, 10) || 3000;
export const MONGODB_URI: string = process.env.MONGODB_URI;
export const JWT_PRIVATE_KEY: string = process.env.JWT_PRIVATE_KEY;
```
* Khi thêm biến mới: thêm vào `environment.ts` + cập nhật `.env.example`.

---

## 16. How To Build New Feature Same Style

### Bước 1: Tạo thư mục module
```text
src/modules/assets/
├── asset.route.ts
├── asset.controller.ts
├── asset.service.ts
├── asset.model.ts
└── asset.validation.ts
```

### Bước 2: Viết Model (Mongoose Schema)
Đầu tiên, phải tạo file Interface tại `src/common/interfaces/asset.interface.ts` (xem mẫu ở Mục 18.9). Sau đó tạo Model:
```typescript
import mongoose, { Schema, Types } from 'mongoose';
import { IAsset } from '@common/interfaces/asset.interface';

const AssetSchema: Schema = new Schema(
    {
        name: { type: String, required: true },
        status: { type: String, enum: ['active', 'inactive'], default: 'active' },
        employeeId: { type: Types.ObjectId, ref: 'Employees' },
    },
    { timestamps: { createdAt: 'createdAt', updatedAt: 'updatedAt' } },
);

AssetSchema.index({ employeeId: 1 }, { name: 'idx_asset_employeeId', background: true });

export default mongoose.model<IAsset>('Assets', AssetSchema);
```

### Bước 3: Viết Validation (Joi)
```typescript
import Joi from 'joi';

export const createAssetSchema = Joi.object({
    name: Joi.string().required().messages({ 'any.required': 'Tên tài sản là bắt buộc' }),
    status: Joi.string().valid('active', 'inactive').optional(),
    employeeId: Joi.string().optional(),
});
```

### Bước 4: Viết Service (Class)
```typescript
export class AssetService {
    public async create(payload): Promise<IApiResponse<IAsset>> {
        const record = await AssetModel.create(payload);
        return { success: true, message: 'Tạo tài sản thành công', data: record };
    }
}
```

### Bước 5: Viết Controller (named exports + catchAsync)
```typescript
const assetService = new AssetService();

export const createAsset = catchAsync(async (req, res) => {
    const result = await assetService.create(req.body);
    res.status(201).json(result);
});
```

### Bước 6: Viết Route
```typescript
const router = Router();
router.post('/', validate(createAssetSchema), authorizationPermissionScopeMiddleware, createAsset);
export default router;
```

### Bước 7: Đăng ký Route vào hệ thống
Mở `src/common/routes/index.ts`, thêm:
```typescript
import AssetRoute from '@modules/assets/asset.route';
router.use('/assets', authenticationMiddleware, authorizationPermissionMiddleware, authorizationPermissionScopeMiddleware, AssetRoute);
```

---

## 17. Do / Don't

### DO
* **DO** dùng `catchAsync` cho mọi controller handler.
* **DO** ném lỗi bằng `throw new ApiError(code, message)` với message tiếng Việt.
* **DO** trả response theo cấu trúc `{ success, message, data }` hoặc `{ success, message, data, meta }`.
* **DO** dùng alias `@modules/`, `@common/` khi import xuyên module.
* **DO** dùng **Yarn** (`yarn add`, `yarn dev`).
* **DO** khai báo index Mongoose rõ ràng với `name` + `background: true`.
* **DO** dùng MongoDB Transactions khi cập nhật nhiều collection đồng thời.

### DON'T
* **DON'T** viết logic nghiệp vụ trong controller — controller chỉ parse request và gọi service.
* **DON'T** dùng `res.status().json()` trong catch block — để errorHandler xử lý.
* **DON'T** dùng `npm install` — tránh sinh `package-lock.json` xung đột `yarn.lock`.
* **DON'T** hardcode chuỗi kết nối DB, JWT key hay port — lấy từ `environment.ts`.

---

## 18. Mẫu Code Chuẩn Cho Các Thư Mục Cốt Lõi

Để đảm bảo AI Agent hiểu sâu cách vận hành của hệ thống, dưới đây là các đoạn mã mẫu "xương sống" của dự án:

### 18.1. Xử lý lỗi tập trung (`src/common/utils/ApiError.ts` & `catchAsync.ts`)
```typescript
// ApiError.ts: Lớp lỗi tùy chỉnh để mang theo status code
export class ApiError extends Error {
    public statusCode: number;
    constructor(statusCode: number, message: string) {
        super(message);
        this.statusCode = statusCode;
        Object.setPrototypeOf(this, ApiError.prototype);
    }
}

// catchAsync.ts: Wrapper để tự động forward lỗi async sang middleware error
export const catchAsync = (fn: (req, res, next) => void) => {
    return (req: Request, res: Response, next: NextFunction) => {
        Promise.resolve(fn(req, res, next)).catch((err) => next(err));
    };
};
```

### 18.2. Middleware xử lý lỗi toàn cục (`src/common/middlewares/error.middleware.ts`)
```typescript
export const errorHandler = (err: Error | ApiError, req: Request, res: Response, next: NextFunction) => {
    let statusCode = 500;
    let message = 'Internal Server Error';

    if (err instanceof ApiError) {
        statusCode = err.statusCode;
        message = err.message;
    } else {
        logger.error('Unrecognized Error:', err);
    }

    res.status(statusCode).json({
        success: false,
        status: statusCode,
        message: message,
    });
};
```

### 18.3. Middleware Validation (`src/common/middlewares/validate.middleware.ts`)
**Lưu ý cực kỳ quan trọng**: Luôn phải bảo toàn `employeeId` và `departmentId` trong `req.body` để các middleware phân quyền Scope phía sau có dữ liệu đối chiếu, ngay cả khi Joi schema áp dụng quy tắc `stripUnknown: true`.
```typescript
export const validate = (schema: Joi.ObjectSchema) => {
    return (req: Request, res: Response, next: NextFunction) => {
        const { employeeId, departmentId } = req.body;
        const { error, value } = schema.validate({
            ...req.body,
            ...(employeeId ? { employeeId: String(employeeId) } : {}),
            ...(departmentId ? { departmentId: String(departmentId) } : {}),
        }, {
            abortEarly: false,
            stripUnknown: true,
        });

        if (error) {
            const errorMessage = error.details.map((details) => details.message).join(', ');
            return next(new ApiError(400, errorMessage));
        }
        req.body = {
            ...value,
            ...(employeeId ? { employeeId: employeeId } : {}),
            ...(departmentId ? { departmentId: departmentId } : {}),
        };
        next();
    };
};
```

### 18.4. Mã hóa tương thích FE/BE (`src/common/utils/crypto.utils.ts`)
```typescript
import crypto from 'crypto';

export const decrypt = (encryptedData: string, secret: string): string | null => {
    try {
        const parts = encryptedData.split(':'); // iv:content
        if (parts.length !== 2) return null;

        const iv = Buffer.from(parts[0], 'hex');
        const encryptedText = Buffer.from(parts[1], 'hex');
        const key = crypto.createHash('sha256').update(secret).digest();

        const decipher = crypto.createDecipheriv('aes-256-cbc', Buffer.from(key), iv);
        let decrypted = decipher.update(encryptedText);
        decrypted = Buffer.concat([decrypted, decipher.final()]);

        return decrypted.toString();
    } catch (error) {
        logger.error('Decryption failed:', error);
        return null;
    }
};
```

### 18.5. Quản lý Realtime (`src/socket/socket.service.ts`)
```typescript
class SocketService {
    private server?: Server;
    
    // Gửi thông báo đích danh cho 1 nhân viên
    public sendNotification(accountId: string, event: string, data: unknown): void {
        if (this.server) {
            this.server.to(`u_${accountId}`).emit(event, data);
        }
    }

    // Gửi thông báo cho toàn bộ nhân viên có role cụ thể (VD: role_HR)
    public sendToRole(roleCode: string, event: string, data: unknown): void {
        if (this.server) {
            this.server.to(`role_${roleCode.toUpperCase()}`).emit(event, data);
        }
    }
}
```

### 18.6. Bản đồ phân quyền tĩnh (`src/common/config/permissionRoutes.ts`)
Mẫu định nghĩa quy tắc ánh xạ API Route sang chuỗi hành động kiểm tra quyền:
```typescript
export const permissionRoutes = {
    assets: {
        view: [
            { route: '/assets/:id', method: 'GET' },
            { route: '/assets', method: 'GET' },
        ],
        create: [{ route: '/assets', method: 'POST' }],
        update: [{ route: '/assets/:id', method: 'PATCH' }],
        delete: [{ route: '/assets/:id', method: 'DELETE' }],
    },
};
```

### 18.7. Mẫu Phân Trang Bằng Aggregation Pipeline (Service)
Chuẩn mực phân trang và map dữ liệu quan hệ, kết hợp bắt lỗi Scope chặt chẽ.
```typescript
import { QueryFilter, PipelineStage, Types } from 'mongoose';
import { generateDiacriticInsensitiveRegex } from '@common/utils/stringUtils';

export class ExampleService {
    public async getByFilter(filter: IFilter): Promise<IPaginatedResponse<IData>> {
        const query: QueryFilter<IFilter> = {};

        // Ràng buộc Scope (Phân quyền dữ liệu cấp dòng)
        if (filter.employeeId) query.employeeId = filter.employeeId;
        if (filter.departmentId) query.departmentId = filter.departmentId;
        
        // Search tiếng Việt không dấu
        if (filter.keyword) {
            const safeRegex = generateDiacriticInsensitiveRegex(filter.keyword);
            query.name = { $regex: `.*${safeRegex}.*`, $options: 'i' };
        }

        const aggregateQuery: PipelineStage[] = [
            { $match: query },
            { $sort: { createdAt: -1 } }
        ];

        // Dùng Promise.all để tính Count và lấy Data song song
        const [records, docCounter] = await Promise.all([
            DataModel.aggregate([
                ...aggregateQuery,
                { $skip: (filter.page - 1) * filter.limit },
                { $limit: filter.limit },
                // Map quan hệ
                { $lookup: { from: 'departments', localField: 'departmentId', foreignField: '_id', as: 'department' } },
                { $unwind: { path: '$department', preserveNullAndEmptyArrays: true } }
            ]),
            DataModel.aggregate([...aggregateQuery, { $count: 'count' }])
        ]);

        const count = docCounter[0]?.count || 0;
        return {
            success: true,
            message: 'Thành công',
            data: records,
            meta: {
                totalItems: count,
                itemCount: records.length,
                itemsPerPage: filter.limit,
                totalPages: Math.ceil(count / filter.limit),
                currentPage: filter.page,
            }
        };
    }
}
```

### 18.8. Mẫu Transaction Mongoose Cập Nhật Đa Bảng (Service)
Tuyệt đối phải truyền `session` qua tất cả các hàm để đảm bảo ACID.
```typescript
import mongoose from 'mongoose';

export class ExampleTransactionService {
    public async syncComplexData(employeeId: string, status: string): Promise<void> {
        const session = await mongoose.startSession();
        try {
            await session.withTransaction(async () => {
                // 1. Cập nhật record cha (nhớ truyền session vào Mongoose options)
                const doc = await MainModel.findOneAndUpdate(
                    { employeeId },
                    { status },
                    { new: true, session }
                );

                // 2. Gọi qua service con (nhớ bổ sung params { session } ở hàm con)
                if (doc) {
                    await otherService.updateRelatedStatus(doc._id.toString(), status, { session });
                }
            });
        } finally {
            session.endSession();
        }
    }
}
```

### 18.9. Mẫu Khai Báo Interface (Type Definition)
Lưu tại `src/common/interfaces/<module_name>.interface.ts`. Phải luôn kế thừa `Document` của Mongoose cho entity gốc và khai báo các Type Payload, Filter tương ứng.
```typescript
import { Document, Types } from 'mongoose';
// Nếu có Enum, import từ '@common/utils/enum'

export interface IAsset extends Document {
    _id: Types.ObjectId;
    name: string;
    status: string;
    employeeId: Types.ObjectId;
}

export interface IAssetPayload {
    name?: string;
    status?: string;
    employeeId?: string; // Payload nhận string từ request body
}

export interface IAssetFilter {
    name?: string;
    status?: string;
    employeeId?: Types.ObjectId | string; // Filter có thể nhận string rồi parse thành ObjectId ở Controller
    limit?: number;
    page?: number;
}
```

---

## 19. Clone This Style For Future Projects

1. **Scaffold**: Tạo thư mục `src/` với cấu trúc `common/` + `modules/` + `socket/` + `worker/`.
2. **Cài đặt**:
   ```bash
   yarn add express mongoose ioredis jsonwebtoken bcrypt joi winston winston-daily-rotate-file dotenv cors cookie-parser crypto-js multer nodemailer node-cron socket.io exceljs moment-timezone swagger-ui-express yamljs
   yarn add -D typescript ts-node-dev tsconfig-paths tscpaths @types/express @types/node @types/crypto-js @types/multer @types/jsonwebtoken @types/bcrypt rimraf
   ```
3. **Copy Core**: Bê nguyên `common/utils/` (ApiError, catchAsync, logger, token), `common/middlewares/`, và `common/interfaces/response.interface.ts`.
4. **Cấu hình**: Nhân bản `tsconfig.json` (paths alias), `.prettierrc`, `eslint.config.js`, `Dockerfile`, `docker-compose.yml`, `.gitlab-ci.yml`.
5. **Phát triển**: Tạo module mới theo template Bước 1-7 ở mục 16.
