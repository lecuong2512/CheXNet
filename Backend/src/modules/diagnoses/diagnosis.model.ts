import mongoose, { Schema, Types } from 'mongoose';
import { IDiagnosis } from '@common/interfaces/diagnosis.interface';
import { DiagnosisStatus } from '@common/utils/enum';

const PredictionSchema = new Schema(
    {
        className: { type: String, required: true },
        probability: { type: Number, required: true, min: 0, max: 1 },
    },
    { _id: false },
);

const DiagnosisSchema: Schema = new Schema(
    {
        scanId: { type: Types.ObjectId, ref: 'Scans', required: true, unique: true },
        patientId: { type: Types.ObjectId, ref: 'Patients', required: true },
        predictions: { type: [PredictionSchema], default: [] },
        aiModel: { type: String, default: 'convnextv2-large-v3' },
        processingTime: { type: Number, default: 0 },
        status: { type: String, enum: Object.values(DiagnosisStatus), default: DiagnosisStatus.PENDING },
        verifiedBy: { type: Types.ObjectId, ref: 'Users' },
        verifiedAt: { type: Date },
        notes: { type: String },
    },
    {
        timestamps: { createdAt: 'createdAt', updatedAt: 'updatedAt' },
    },
);

DiagnosisSchema.index({ scanId: 1 }, { name: 'idx_diagnosis_scan', background: true, unique: true });
DiagnosisSchema.index({ patientId: 1, createdAt: -1 }, { name: 'idx_diagnosis_patient_date', background: true });
DiagnosisSchema.index({ status: 1 }, { name: 'idx_diagnosis_status', background: true });

export default mongoose.model<IDiagnosis>('Diagnoses', DiagnosisSchema);
