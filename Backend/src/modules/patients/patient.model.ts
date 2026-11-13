import mongoose, { Schema, Types } from 'mongoose';
import { IPatient } from '@common/interfaces/patient.interface';
import { Gender, RiskLevel } from '@common/utils/enum';

const VitalsSchema = new Schema(
    {
        heartRate: { type: Number },
        bloodPressure: { type: String },
        spo2: { type: Number },
        temperature: { type: Number },
        lungIndex: { type: Number },
    },
    { _id: false },
);

const PatientSchema: Schema = new Schema(
    {
        patientCode: { type: String, required: true, unique: true, uppercase: true, trim: true },
        name: { type: String, required: true, trim: true },
        gender: { type: String, enum: Object.values(Gender), required: true },
        age: { type: Number, required: true, min: 0, max: 150 },
        bloodType: { type: String, trim: true },
        department: { type: String, trim: true },
        phone: { type: String, trim: true },
        address: { type: String, trim: true },
        vitals: { type: VitalsSchema, default: {} },
        riskLevel: { type: String, enum: Object.values(RiskLevel), default: RiskLevel.NORMAL },
        createdBy: { type: Types.ObjectId, ref: 'Users', required: true },
        isActive: { type: Boolean, default: true },
    },
    {
        timestamps: { createdAt: 'createdAt', updatedAt: 'updatedAt' },
    },
);

PatientSchema.index({ patientCode: 1 }, { name: 'idx_patient_code', background: true, unique: true });
PatientSchema.index({ name: 1 }, { name: 'idx_patient_name', background: true });
PatientSchema.index({ department: 1 }, { name: 'idx_patient_department', background: true });
PatientSchema.index({ riskLevel: 1 }, { name: 'idx_patient_riskLevel', background: true });
PatientSchema.index({ createdAt: -1 }, { name: 'idx_patient_createdAt', background: true });

export default mongoose.model<IPatient>('Patients', PatientSchema);
