import mongoose, { Schema, Types } from 'mongoose';
import { IScan } from '@common/interfaces/scan.interface';
import { ScanStatus, ScanType } from '@common/utils/enum';

const ScanSchema: Schema = new Schema(
    {
        patientId: { type: Types.ObjectId, ref: 'Patients', required: true },
        imagePath: { type: String, required: true },
        imageUrl: { type: String, required: true },
        type: { type: String, enum: Object.values(ScanType), default: ScanType.PA },
        status: { type: String, enum: Object.values(ScanStatus), default: ScanStatus.PENDING },
        uploadedBy: { type: Types.ObjectId, ref: 'Users', required: true },
        notes: { type: String },
        processingStartedAt: { type: Date },
        processingCompletedAt: { type: Date },
    },
    {
        timestamps: { createdAt: 'createdAt', updatedAt: 'updatedAt' },
    },
);

ScanSchema.index({ patientId: 1, createdAt: -1 }, { name: 'idx_scan_patient_date', background: true });
ScanSchema.index({ status: 1 }, { name: 'idx_scan_status', background: true });
ScanSchema.index({ uploadedBy: 1 }, { name: 'idx_scan_uploadedBy', background: true });

export default mongoose.model<IScan>('Scans', ScanSchema);
