import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

interface AuthUser {
    _id: string;
    name: string;
    email: string;
    role: string;
    department?: string;
    avatar?: string;
}

interface AuthState {
    user: AuthUser | null;
    accessToken: string | null;
    refreshToken: string | null;
    isAuthenticated: boolean;
}

const loadFromStorage = (): Partial<AuthState> => {
    try {
        const accessToken = localStorage.getItem('chexnet_accessToken');
        const refreshToken = localStorage.getItem('chexnet_refreshToken');
        const userStr = localStorage.getItem('chexnet_user');
        const user = userStr ? JSON.parse(userStr) : null;
        return {
            accessToken,
            refreshToken,
            user,
            isAuthenticated: !!accessToken && !!user,
        };
    } catch {
        return { accessToken: null, refreshToken: null, user: null, isAuthenticated: false };
    }
};

const persisted = loadFromStorage();

const initialState: AuthState = {
    user: persisted.user || null,
    accessToken: persisted.accessToken || null,
    refreshToken: persisted.refreshToken || null,
    isAuthenticated: persisted.isAuthenticated || false,
};

export const authSlice = createSlice({
    name: 'auth',
    initialState,
    reducers: {
        setCredentials: (
            state,
            action: PayloadAction<{ user: AuthUser; accessToken: string; refreshToken: string }>,
        ) => {
            state.user = action.payload.user;
            state.accessToken = action.payload.accessToken;
            state.refreshToken = action.payload.refreshToken;
            state.isAuthenticated = true;
            localStorage.setItem('chexnet_accessToken', action.payload.accessToken);
            localStorage.setItem('chexnet_refreshToken', action.payload.refreshToken);
            localStorage.setItem('chexnet_user', JSON.stringify(action.payload.user));
        },
        updateTokens: (
            state,
            action: PayloadAction<{ accessToken: string; refreshToken: string }>,
        ) => {
            state.accessToken = action.payload.accessToken;
            state.refreshToken = action.payload.refreshToken;
            localStorage.setItem('chexnet_accessToken', action.payload.accessToken);
            localStorage.setItem('chexnet_refreshToken', action.payload.refreshToken);
        },
        logout: (state) => {
            state.user = null;
            state.accessToken = null;
            state.refreshToken = null;
            state.isAuthenticated = false;
            localStorage.removeItem('chexnet_accessToken');
            localStorage.removeItem('chexnet_refreshToken');
            localStorage.removeItem('chexnet_user');
        },
    },
});

export const { setCredentials, updateTokens, logout } = authSlice.actions;
export default authSlice.reducer;
