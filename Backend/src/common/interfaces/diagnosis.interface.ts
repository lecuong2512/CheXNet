import { Document, Types } from 'mongoose';
import { DiagnosisStatus, ClassName } from '@common/utils/enum';

export interface IPrediction {
    className: ClassName | string;
    probability: number;
}

export interface IDiagnosis extends Document {
    _id: Types.ObjectId;
    scanId: Types.ObjectId;
    patientId: Types.ObjectId;
    predictions: IPrediction[];
    aiModel: string;
    processingTime: number;
    hybridAnalysis?: Record<string, unknown>;
    inferenceMeta?: Record<string, unknown>;
    status: DiagnosisStatus;
    verifiedBy?: Types.ObjectId;
    verifiedAt?: Date;
    notes?: string;
    revision?: number;
    verifiedByName?: string;
    referenceCase?: {
        referenceId: string;
        diseases: string[];
        status: 'pending' | 'ready';
        savedAt?: Date;
        savedBy?: string;
        revisionAtSave?: number;
    };
    createdAt: Date;
    updatedAt: Date;
}

export interface IDiagnosisPayload {
    scanId: string;
    patientId: string;
    predictions: IPrediction[];
    aiModel?: string;
    processingTime?: number;
}

export interface IDiagnosisFilter {
    patientId?: Types.ObjectId | string;
    scanId?: Types.ObjectId | string;
    status?: DiagnosisStatus;
    verifiedBy?: Types.ObjectId | string;
    page?: number;
    limit?: number;
}
