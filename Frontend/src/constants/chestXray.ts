/** Phạm vi đề tài: chỉ X-quang ngực — chẩn đoán bệnh phổi */
export const CHEST_XRAY_SCAN_TYPES = [
    'X-Quang ngực PA',
    'X-Quang ngực AP',
    'X-Quang ngực Nghiêng',
] as const;

export type ChestXrayScanType = (typeof CHEST_XRAY_SCAN_TYPES)[number];

export const DEFAULT_CHEST_SCAN_TYPE: ChestXrayScanType = 'X-Quang ngực PA';

export const CHEST_XRAY_SCOPE_NOTE =
    'Hệ thống chỉ phân tích phim X-quang ngực (14 nhóm bệnh và nhãn không phát hiện bất thường). Không hỗ trợ CT, MRI hay X-quang vùng khác.';
