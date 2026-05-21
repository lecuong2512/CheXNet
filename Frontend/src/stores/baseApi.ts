import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import type { BaseQueryFn, FetchArgs, FetchBaseQueryError } from '@reduxjs/toolkit/query';
import { logout } from './authSlice';

// ── Backend Connection ────────────────────────────────────────────────────────
const API_BASE_URL = 'http://localhost:3005/api/v1';
const BACKEND_ORIGIN = 'http://localhost:3005';

const AI_POLL_INTERVAL_MS = 1500;
const AI_POLL_MAX_WAIT_MS = 120_000;

/** Chờ backend + AI service hoàn tất inference cho một scan */
async function waitForScanDiagnosis(
    patientId: string,
    scanId: string,
    baseQuery: (arg: string) => Promise<{ data?: unknown; error?: FetchBaseQueryError }>,
): Promise<{ diagnosis: any; scan: any }> {
    const deadline = Date.now() + AI_POLL_MAX_WAIT_MS;

    while (Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, AI_POLL_INTERVAL_MS));

        const profileResult = await baseQuery(`/patients/${patientId}`);
        if (profileResult.error) continue;

        const patient = (profileResult.data as { data?: { scans?: any[]; diagnoses?: any[] } })?.data;
        const scan = (patient?.scans || []).find((s) => String(s._id) === scanId);

        if (scan?.status === 'failed') {
            const reason =
                scan.notes ||
                'Phân tích AI thất bại. Vui lòng upload đúng phim X-quang ngực (PA/AP).';
            throw new Error(reason);
        }

        const diagnosis = (patient?.diagnoses || []).find((d) => String(d.scanId) === scanId);
        if (diagnosis?.predictions?.length) {
            return { diagnosis, scan };
        }
    }

    throw new Error(
        'Hết thời gian chờ kết quả AI. Đảm bảo AI service (port 8000) đang chạy và thử lại.',
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// HELPER: Vietnamese Class Name Translation (15 labels ConvNeXtV2-Large)
// ══════════════════════════════════════════════════════════════════════════════
const classNameViMap: Record<string, string> = {
    'No Finding': 'Không phát hiện bất thường',
    'Atelectasis': 'Xẹp phổi',
    'Cardiomegaly': 'Tim to',
    'Effusion': 'Tràn dịch màng phổi',
    'Infiltration': 'Thâm nhiễm phổi',
    'Mass': 'Khối u phổi',
    'Nodule': 'Nốt mờ phổi',
    'Pneumonia': 'Viêm phổi đông đặc',
    'Pneumothorax': 'Tràn khí màng phổi',
    'Consolidation': 'Đông đặc phổi',
    'Edema': 'Phù nề phổi',
    'Emphysema': 'Khí phế thũng',
    'Fibrosis': 'Xơ phổi',
    'Pleural_Thickening': 'Dày màng phổi',
    'Hernia': 'Thoát vị hoành',
};

const diagStatusToVi: Record<string, string> = {
    pending: 'Đang chờ',
    verified: 'Đã xác minh',
    flagged: 'Đánh dấu',
};

const diagStatusToEn: Record<string, string> = {
    'Đang chờ': 'pending',
    'Đã xác minh': 'verified',
    'Đánh dấu': 'flagged',
};

// ══════════════════════════════════════════════════════════════════════════════
// HELPER: Date Formatting (ISO → DD/MM/YYYY + HH:MM SA/CH)
// ══════════════════════════════════════════════════════════════════════════════
const formatDateVi = (isoStr: string): { date: string; time: string } => {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return { date: '', time: '' };
    const day = String(d.getDate()).padStart(2, '0');
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const year = d.getFullYear();
    const date = `${day}/${month}/${year}`;
    const hours = d.getHours();
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const period = hours >= 12 ? 'CH' : 'SA';
    const h12 = hours % 12 || 12;
    const time = `${String(h12).padStart(2, '0')}:${minutes} ${period}`;
    return { date, time };
};

// ══════════════════════════════════════════════════════════════════════════════
// HELPER: Map raw AI predictions → Frontend display format
// ══════════════════════════════════════════════════════════════════════════════
/** Ngưỡng lâm sàng: chỉ hiển thị nhãn bệnh khi model đủ tin cậy (≥ 35%) */
const CLINICAL_PROB_THRESHOLD = 0.35;

const mapPredictions = (rawPreds: any[]): any[] => {
    if (!rawPreds || rawPreds.length === 0) return [];
    return rawPreds
        .filter((p) => p.className !== 'No Finding' && p.probability >= CLINICAL_PROB_THRESHOLD)
        .map((p) => ({
            name: classNameViMap[p.className] || p.className,
            probability: Math.round(p.probability * 100),
            status: p.probability > 0.5 ? 'warning' : 'normal',
        }))
        .sort((a, b) => b.probability - a.probability);
};

// ══════════════════════════════════════════════════════════════════════════════
// HELPER: Auto-generate AI clinical description from predictions
// ══════════════════════════════════════════════════════════════════════════════
const generateAiDescription = (rawPreds: any[]): string => {
    if (!rawPreds || rawPreds.length === 0) {
        return 'Mô hình AI không phát hiện các dấu hiệu bệnh lý rõ ràng trên phim phổi. Bệnh nhân nên tiếp tục lịch theo dõi lâm sàng định kỳ.';
    }
    const sig = rawPreds.filter(
        (p) => p.className !== 'No Finding' && p.probability >= CLINICAL_PROB_THRESHOLD,
    );
    if (sig.length === 0) {
        return 'Hình ảnh X-quang phổi ổn định. Không phát hiện bất thường rõ rệt so với phim cơ bản. Đề xuất theo dõi định kỳ.';
    }
    const top = sig[0];
    const name = classNameViMap[top.className] || top.className;
    const pct = Math.round(top.probability * 100);
    return (
        `Mô hình AI đã phát hiện dấu hiệu ${name} với độ tin cậy ${pct}%. ` +
        'Khuyến nghị bác sĩ xem xét đối chiếu với triệu chứng lâm sàng và kết quả xét nghiệm để xác nhận chẩn đoán.'
    );
};

const generateNextSteps = (rawPreds: any[]): string[] => {
    const sig = (rawPreds || []).filter(
        (p) => p.className !== 'No Finding' && p.probability >= CLINICAL_PROB_THRESHOLD,
    );
    if (sig.length === 0) return ['Theo dõi định kỳ 6 tháng.'];
    return [
        'Tương quan với xét nghiệm chức năng hô hấp (PFT) lâm sàng.',
        'Đánh giá tiền sử bệnh và các yếu tố nguy cơ của bệnh nhân.',
        'Cân nhắc tái chụp X-quang ngực sau điều trị hoặc chuyển tuyến khi cần chụp CT (ngoài phạm vi hệ thống).',
    ];
};

// ══════════════════════════════════════════════════════════════════════════════
// HELPER: Merge a scan document with its matching diagnosis
// ══════════════════════════════════════════════════════════════════════════════
const mapScanWithDiagnosis = (scan: any, diagnosis: any): any => {
    const { date, time } = formatDateVi(scan.createdAt);
    const rawPreds = diagnosis?.predictions || [];
    const diagStatus = diagnosis?.status || 'pending';
    return {
        id: String(scan._id),
        diagnosisId: diagnosis ? String(diagnosis._id) : null,
        date,
        time,
        type: scan.type || 'X-Quang ngực PA',
        status: diagStatusToVi[diagStatus] || diagStatus,
        image: scan.imageUrl ? `${BACKEND_ORIGIN}${scan.imageUrl}` : undefined,
        predictions: mapPredictions(rawPreds),
        description: generateAiDescription(rawPreds),
        nextSteps: generateNextSteps(rawPreds),
    };
};

// ══════════════════════════════════════════════════════════════════════════════
// LOCAL CONFIG (No backend endpoint — persists in localStorage)
// ══════════════════════════════════════════════════════════════════════════════
const defaultConfig = {
    preProcess: true,
    anonymousSend: false,
    thresholdNodule: 85,
    thresholdPleural: 92,
};

const getStoredData = (key: string, defaultVal: any) => {
    const data = localStorage.getItem(key);
    if (!data) {
        localStorage.setItem(key, JSON.stringify(defaultVal));
        return defaultVal;
    }
    return JSON.parse(data);
};

// ══════════════════════════════════════════════════════════════════════════════
// BASE QUERY: fetchBaseQuery + Redux Token + 401 → Logout
// ══════════════════════════════════════════════════════════════════════════════
const rawBaseQuery = fetchBaseQuery({
    baseUrl: API_BASE_URL,
    prepareHeaders: (headers, { getState }) => {
        const token = (getState() as any).auth?.accessToken;
        if (token) {
            headers.set('Authorization', `Bearer ${token}`);
        }
        return headers;
    },
});

const customBaseQuery: BaseQueryFn<string | FetchArgs, unknown, FetchBaseQueryError> = async (
    args,
    api,
    extraOptions,
) => {
    const result = await rawBaseQuery(args, api, extraOptions);

    // Khi token hết hạn hoặc không hợp lệ → đăng xuất
    if (result.error && result.error.status === 401) {
        api.dispatch(logout());
    }

    return result;
};

// ══════════════════════════════════════════════════════════════════════════════
// RTK QUERY API DEFINITION
// ══════════════════════════════════════════════════════════════════════════════
export const baseApi = createApi({
    reducerPath: 'api',
    baseQuery: customBaseQuery,
    tagTypes: ['Patients', 'Config', 'SystemHealth'],
    endpoints: (builder) => ({

        // ── GET Patients list (flattened scan history for History + Dashboard) ─
        getPatients: builder.query<any[], { search?: string; type?: string; risk?: string } | void>({
            queryFn: async (arg, _api, _extra, baseQuery) => {
                const params = new URLSearchParams();
                if (arg?.search) params.set('keyword', arg.search);
                params.set('limit', '50');

                const result = await baseQuery(`/patients?${params.toString()}`);
                if (result.error) return { error: result.error };

                const patients = (result.data as any).data || [];

                // Flatten: mỗi scan thành 1 dòng kèm thông tin bệnh nhân
                let flatScans: any[] = patients.flatMap((p: any) => {
                    const scans = p.scans || [];
                    const diagnoses = p.diagnoses || [];

                    if (scans.length === 0) {
                        // Bệnh nhân chưa có phim chụp — vẫn hiển thị trong danh sách
                        const { date, time } = p.createdAt
                            ? formatDateVi(p.createdAt)
                            : { date: '', time: '' };
                        return [
                            {
                                patientId: p.patientCode,
                                patientName: p.name,
                                gender: p.gender,
                                age: p.age,
                                bloodType: p.bloodType,
                                vitals: p.vitals,
                                id: String(p._id),
                                date,
                                time,
                                type: 'Chưa có phim chụp',
                                status: 'Đang chờ',
                                predictions: [],
                                description: '',
                                nextSteps: [],
                            },
                        ];
                    }

                    return scans.map((scan: any) => {
                        const diag = diagnoses.find(
                            (d: any) => String(d.scanId) === String(scan._id),
                        );
                        const mapped = mapScanWithDiagnosis(scan, diag);
                        return {
                            patientId: p.patientCode,
                            patientName: p.name,
                            gender: p.gender,
                            age: p.age,
                            bloodType: p.bloodType,
                            vitals: p.vitals,
                            ...mapped,
                        };
                    });
                });

                // ── Client-side type filter ──────────────────────────────────
                if (arg?.type && arg.type !== 'Tất cả Kết quả') {
                    if (arg.type === 'Bình thường') {
                        flatScans = flatScans.filter(
                            (s) => s.predictions.length === 0,
                        );
                    } else {
                        const typeMap: Record<string, string> = {
                            'Tràn dịch màng phổi': 'Tràn dịch màng phổi',
                            'Nốt mờ / Khối u': 'Nốt mờ',
                            'Viêm phổi': 'Viêm phổi',
                        };
                        const keyword = typeMap[arg.type] || arg.type;
                        flatScans = flatScans.filter((s) =>
                            s.predictions.some((pr: any) => pr.name.includes(keyword)),
                        );
                    }
                }

                // ── Client-side risk filter ──────────────────────────────────
                if (arg?.risk && arg.risk !== 'Mọi Mức rủi ro') {
                    if (arg.risk === 'Nghiêm trọng') {
                        flatScans = flatScans.filter(
                            (s) =>
                                s.status === 'Đang chờ' ||
                                s.predictions.some((pr: any) => pr.probability > 80),
                        );
                    } else if (arg.risk === 'Trung bình') {
                        flatScans = flatScans.filter(
                            (s) =>
                                s.status === 'Đánh dấu' ||
                                s.predictions.some(
                                    (pr: any) => pr.probability > 50 && pr.probability <= 80,
                                ),
                        );
                    } else if (arg.risk === 'Thấp') {
                        flatScans = flatScans.filter(
                            (s) =>
                                s.status === 'Đã xác minh' ||
                                s.predictions.every((pr: any) => pr.probability <= 50),
                        );
                    }
                }

                return { data: flatScans };
            },
            providesTags: ['Patients'],
        }),

        // ── GET Single Patient Profile ────────────────────────────────────────
        getPatientProfile: builder.query<any, string>({
            queryFn: async (id, _api, _extra, baseQuery) => {
                const result = await baseQuery(`/patients/${id}`);
                if (result.error) return { error: result.error };

                const p = (result.data as any).data;
                if (!p) {
                    return {
                        error: { status: 404, data: 'Không tìm thấy hồ sơ bệnh nhân' } as FetchBaseQueryError,
                    };
                }

                const scans = p.scans || [];
                const diagnoses = p.diagnoses || [];

                const mapped = {
                    id: p.patientCode,
                    _id: String(p._id),
                    name: p.name,
                    gender: p.gender,
                    age: p.age,
                    bloodType: p.bloodType,
                    department: p.department,
                    vitals: p.vitals || {
                        heartRate: 75,
                        bloodPressure: '120/80',
                        spo2: 98,
                        temperature: 36.8,
                        lungIndex: 90,
                    },
                    scans: scans.map((scan: any) => {
                        const diag = diagnoses.find(
                            (d: any) => String(d.scanId) === String(scan._id),
                        );
                        return mapScanWithDiagnosis(scan, diag);
                    }),
                };

                return { data: mapped };
            },
            providesTags: (_result, _error, id) => [{ type: 'Patients', id }],
        }),

        // ── PATCH Verify Scan (resolve diagnosis → patch status) ──────────────
        verifyScan: builder.mutation<
            any,
            { patientId: string; scanId: string; status: 'Đã xác minh' | 'Đánh dấu' | 'Đang chờ' }
        >({
            queryFn: async ({ patientId, scanId, status }, _api, _extra, baseQuery) => {
                // Lấy hồ sơ bệnh nhân để tìm diagnosisId khớp với scanId
                const profileResult = await baseQuery(`/patients/${patientId}`);
                if (profileResult.error) return { error: profileResult.error };

                const patient = (profileResult.data as any).data;
                const diagnoses = patient?.diagnoses || [];
                const diagnosis = diagnoses.find(
                    (d: any) => String(d.scanId) === scanId,
                );

                if (!diagnosis) {
                    return {
                        error: {
                            status: 404,
                            data: 'Không tìm thấy kết quả chẩn đoán cho phim chụp này',
                        } as FetchBaseQueryError,
                    };
                }

                const verifyResult = await baseQuery({
                    url: `/diagnoses/${diagnosis._id}/verify`,
                    method: 'PATCH',
                    body: { status: diagStatusToEn[status] || status },
                });

                return verifyResult.error
                    ? { error: verifyResult.error }
                    : { data: verifyResult.data };
            },
            invalidatesTags: ['Patients'],
        }),

        // ── GET System Config (localStorage — no backend endpoint) ────────────
        getSystemConfig: builder.query<any, void>({
            queryFn: async () => {
                const config = getStoredData('chexnet_config', defaultConfig);
                return { data: config };
            },
            providesTags: ['Config'],
        }),

        // ── POST Save System Config (localStorage) ───────────────────────────
        saveSystemConfig: builder.mutation<any, any>({
            queryFn: async (newConfig) => {
                localStorage.setItem('chexnet_config', JSON.stringify(newConfig));
                return { data: newConfig };
            },
            invalidatesTags: ['Config'],
        }),

        // ── GET Server Health Metrics ─────────────────────────────────────────
        getServerHealth: builder.query<any, void>({
            queryFn: async () => {
                // TODO: Kết nối API monitoring thực tế
                return {
                    data: {
                        gpuLoad: 0,
                        ramUsage: 0,
                        latency: 0,
                    },
                };
            },
            providesTags: ['SystemHealth'],
        }),

        // ── POST Upload New Scan (multipart/form-data → Multer → AI) ─────────
        uploadScan: builder.mutation<
            any,
            { patientId: string; scanType: string; imageFile?: File }
        >({
            queryFn: async ({ patientId, scanType, imageFile }, _api, _extra, baseQuery) => {
                // Resolve patientCode → MongoDB ObjectId
                const profileResult = await baseQuery(`/patients/${patientId}`);
                if (profileResult.error) return { error: profileResult.error };

                const patient = (profileResult.data as any).data;
                const mongoId = String(patient._id);

                if (!imageFile) {
                    return {
                        error: {
                            status: 400,
                            data: 'Chưa chọn file ảnh để tải lên',
                        } as FetchBaseQueryError,
                    };
                }

                const formData = new FormData();
                formData.append('image', imageFile);
                formData.append('type', scanType || 'X-Quang ngực PA');

                const uploadResult = await baseQuery({
                    url: `/scans/${mongoId}/upload`,
                    method: 'POST',
                    body: formData,
                });

                if (uploadResult.error) {
                    return { error: uploadResult.error };
                }

                const payload = uploadResult.data as {
                    data?: { _id: string; diagnosis?: { aiModel?: string; predictions?: unknown[]; processingTime?: number } };
                };
                const scan = payload?.data;
                const diagnosis = scan?.diagnosis;

                if (diagnosis?.predictions?.length) {
                    return {
                        data: {
                            ...(uploadResult.data as object),
                            scan,
                            diagnosis,
                            aiModel: diagnosis.aiModel,
                            predictions: diagnosis.predictions,
                            processingTime: diagnosis.processingTime,
                        },
                    };
                }

                const scanId = scan?._id ? String(scan._id) : '';
                if (!scanId) {
                    return {
                        error: {
                            status: 500,
                            data: { message: 'Upload thành công nhưng không nhận được kết quả AI' },
                        } as FetchBaseQueryError,
                    };
                }

                try {
                    const polled = await waitForScanDiagnosis(
                        patientId,
                        scanId,
                        (path) => baseQuery(path),
                    );
                    return {
                        data: {
                            ...(uploadResult.data as object),
                            scan: polled.scan,
                            diagnosis: polled.diagnosis,
                            aiModel: polled.diagnosis.aiModel,
                            predictions: polled.diagnosis.predictions,
                            processingTime: polled.diagnosis.processingTime,
                        },
                    };
                } catch (pollError) {
                    const message =
                        pollError instanceof Error ? pollError.message : 'Chờ kết quả AI thất bại';
                    return {
                        error: {
                            status: 504,
                            data: { message },
                        } as FetchBaseQueryError,
                    };
                }
            },
            invalidatesTags: ['Patients'],
        }),

        // ── POST Login ────────────────────────────────────────────────────────
        login: builder.mutation<any, any>({
            query: (credentials) => ({
                url: '/auth/login',
                method: 'POST',
                body: credentials,
            }),
        }),

        // ── POST Register ─────────────────────────────────────────────────────
        register: builder.mutation<any, any>({
            query: (userData) => ({
                url: '/auth/register',
                method: 'POST',
                body: userData,
            }),
        }),

        // ── POST Create Patient ────────────────────────────────────────────────
        createPatient: builder.mutation<any, any>({
            query: (patientData) => ({
                url: '/patients',
                method: 'POST',
                body: patientData,
            }),
            invalidatesTags: ['Patients'],
        }),
    }),
});

export const {
    useGetPatientsQuery,
    useGetPatientProfileQuery,
    useVerifyScanMutation,
    useGetSystemConfigQuery,
    useSaveSystemConfigMutation,
    useGetServerHealthQuery,
    useUploadScanMutation,
    useLoginMutation,
    useRegisterMutation,
    useCreatePatientMutation,
} = baseApi;


