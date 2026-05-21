import { Document, Types } from 'mongoose';
import { ScanStatus, ScanType } from '@common/utils/enum';

export interface IScan extends Document {
    _id: Types.ObjectId;
    patientId: Types.ObjectId;
    imagePath: string;
    imageUrl: string;
    type: ScanType;
    status: ScanStatus;
    uploadedBy: Types.ObjectId;
    notes?: string;
    processingStartedAt?: Date;
    processingCompletedAt?: Date;
    createdAt: Date;
    updatedAt: Date;
}

export interface IScanPayload {
    patientId?: string;
    type?: ScanType;
    notes?: string;
}

export interface IScanFilter {
    patientId?: Types.ObjectId | string;
    status?: ScanStatus;
    type?: ScanType;
    uploadedBy?: Types.ObjectId | string;
    page?: number;
    limit?: number;
}
