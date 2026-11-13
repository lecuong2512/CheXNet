import { createSlice } from "@reduxjs/toolkit";
import type { PayloadAction } from "@reduxjs/toolkit";

interface NotificationItem {
  id: string;
  title: string;
  description: string;
  type: "info" | "warning" | "error" | "success";
  timestamp: string;
  unread: boolean;
}

interface NotificationState {
  items: NotificationItem[];
}

const initialState: NotificationState = {
  items: [],
};

export const notificationSlice = createSlice({
  name: "notification",
  initialState,
  reducers: {
    addNotification: (state, action: PayloadAction<Omit<NotificationItem, "id" | "timestamp" | "unread">>) => {
      state.items.unshift({
        ...action.payload,
        id: `n-${Date.now()}`,
        timestamp: "Vừa xong",
        unread: true,
      });
    },
    markAllAsRead: (state) => {
      state.items.forEach(item => {
        item.unread = false;
      });
    },
    clearNotifications: (state) => {
      state.items = [];
    }
  }
});

export const { addNotification, markAllAsRead, clearNotifications } = notificationSlice.actions;
export default notificationSlice.reducer;
