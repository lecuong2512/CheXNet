import { Document, Types } from 'mongoose';
import { Gender, RiskLevel } from '@common/utils/enum';

export interface IVitals {
    heartRate?: number;
    bloodPressure?: string;
    spo2?: number;
    temperature?: number;
    lungIndex?: number;
}

export interface IPatient extends Document {
    _id: Types.ObjectId;
    patientCode: string;
    name: string;
    gender: Gender;
    age: number;
    bloodType?: string;
    department?: string;
    phone?: string;
    address?: string;
    vitals: IVitals;
    riskLevel: RiskLevel;
    createdBy: Types.ObjectId;
    isActive: boolean;
    createdAt: Date;
    updatedAt: Date;
}

export interface IPatientPayload {
    patientCode?: string;
    name?: string;
    gender?: Gender;
    age?: number;
    bloodType?: string;
    department?: string;
    phone?: string;
    address?: string;
    vitals?: IVitals;
    riskLevel?: RiskLevel;
}

export interface IPatientFilter {
    keyword?: string;
    gender?: Gender;
    department?: string;
    riskLevel?: RiskLevel;
    isActive?: boolean;
    createdBy?: Types.ObjectId | string;
    page?: number;
    limit?: number;
}
