export enum UserRole {
    ADMIN = 'admin',
    DOCTOR = 'doctor',
    USER = 'user',
}

export enum ScanStatus {
    PENDING = 'pending',
    PROCESSING = 'processing',
    DONE = 'done',
    FAILED = 'failed',
}

export enum DiagnosisStatus {
    PENDING = 'pending',
    VERIFIED = 'verified',
    FLAGGED = 'flagged',
}

/** Chỉ hỗ trợ X-quang ngực — phục vụ chẩn đoán bệnh phổi (đề tài) */
export enum ScanType {
    PA = 'X-Quang ngực PA',
    AP = 'X-Quang ngực AP',
    LATERAL = 'X-Quang ngực Nghiêng',
}

export enum RiskLevel {
    HIGH = 'Nghiêm trọng',
    MEDIUM = 'Trung bình',
    LOW = 'Thấp',
    NORMAL = 'Bình thường',
}

export enum Gender {
    MALE = 'Nam',
    FEMALE = 'Nữ',
    OTHER = 'Khác',
}

// 15 class names của ConvNeXtV2-Large model
export const CLASS_NAMES = [
    'No Finding',
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia',
] as const;

export type ClassName = (typeof CLASS_NAMES)[number];
