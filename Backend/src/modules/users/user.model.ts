import mongoose, { Schema } from 'mongoose';
import { IUser } from '@common/interfaces/user.interface';
import { UserRole } from '@common/utils/enum';

const UserSchema: Schema = new Schema(
    {
        name: { type: String, required: true, trim: true },
        email: { type: String, required: true, unique: true, lowercase: true, trim: true },
        passwordHash: { type: String, required: true },
        role: { type: String, enum: Object.values(UserRole), default: UserRole.DOCTOR },
        department: { type: String, trim: true },
        avatar: { type: String },
        tokenVersion: { type: Number, default: 0 },
        isActive: { type: Boolean, default: true },
    },
    {
        timestamps: { createdAt: 'createdAt', updatedAt: 'updatedAt' },
    },
);

UserSchema.index({ email: 1 }, { name: 'idx_user_email', background: true, unique: true });
UserSchema.index({ role: 1 }, { name: 'idx_user_role', background: true });
UserSchema.index({ isActive: 1 }, { name: 'idx_user_isActive', background: true });

export default mongoose.model<IUser>('Users', UserSchema);
