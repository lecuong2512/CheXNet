import { configureStore } from "@reduxjs/toolkit";
import { baseApi } from "./baseApi";
import themeReducer from "./themeSlice";
import notificationReducer from "./notificationSlice";
import authReducer from "./authSlice";

export const store = configureStore({
  reducer: {
    [baseApi.reducerPath]: baseApi.reducer,
    theme: themeReducer,
    notification: notificationReducer,
    auth: authReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(baseApi.middleware),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
export default store;

