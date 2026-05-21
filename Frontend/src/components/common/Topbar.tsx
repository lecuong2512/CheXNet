import React, { useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";
import type { RootState } from "../../stores/store";
import { markAllAsRead } from "../../stores/notificationSlice";
import { logout } from "../../stores/authSlice";
import { Badge, Popover, List, Typography, message } from "antd";

const { Text } = Typography;

interface TopbarProps {
  onSearch?: (value: string) => void;
}

const Topbar: React.FC<TopbarProps> = ({ onSearch }) => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [searchValue, setSearchValue] = useState("");
  const notifications = useSelector((state: RootState) => state.notification.items);
  const user = useSelector((state: RootState) => state.auth.user);
  const unreadCount = notifications.filter(n => n.unread).length;

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setSearchValue(val);
    if (onSearch) {
      onSearch(val);
    }
  };

  const handleNotificationClick = () => {
    dispatch(markAllAsRead());
  };

  const handleLogout = () => {
    dispatch(logout());
    message.success("Đăng xuất tài khoản thành công!");
    navigate("/login");
  };

  const notificationContent = (
    <div className="w-80">
      <div className="flex justify-between items-center border-b border-gray-100 pb-2 mb-2 dark:border-gray-800">
        <span className="font-bold text-on-surface">Thông báo lâm sàng</span>
        {unreadCount > 0 && (
          <button 
            onClick={handleNotificationClick} 
            className="text-primary text-xs hover:underline font-semibold"
          >
            Đánh dấu đã đọc
          </button>
        )}
      </div>
      {notifications.length === 0 ? (
        <div className="py-4 text-center text-gray-400">Không có thông báo mới</div>
      ) : (
        <List
          itemLayout="horizontal"
          dataSource={notifications}
          renderItem={(item) => (
            <List.Item className={`px-2 py-1.5 rounded-lg mb-1 transition-colors hover:bg-gray-50 dark:hover:bg-gray-800 ${item.unread ? "bg-primary/5" : ""}`}>
              <List.Item.Meta
                avatar={
                  <span className={`material-symbols-outlined mt-1 ${
                    item.type === "warning" ? "text-error" : item.type === "success" ? "text-emerald-500" : "text-primary"
                  }`}>
                    {item.type === "warning" ? "warning" : item.type === "success" ? "check_circle" : "info"}
                  </span>
                }
                title={<span className="font-bold text-sm text-on-surface">{item.title}</span>}
                description={
                  <div className="flex flex-col">
                    <Text type="secondary" className="text-xs">{item.description}</Text>
                    <span className="text-[10px] text-gray-400 mt-0.5">{item.timestamp}</span>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      )}
    </div>
  );

  const profileContent = (
    <div className="w-56 p-1">
      <div className="border-b border-gray-100 pb-3 mb-3 dark:border-gray-800">
        <div className="font-bold text-on-surface text-sm">{user?.name || "Bác sĩ CheXNet"}</div>
        <div className="text-[11px] text-gray-400 mt-0.5">{user?.email || "doctor@hospital.org"}</div>
        {user?.department && (
          <span className="inline-block mt-2 px-2 py-0.5 rounded bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 text-[10px] font-semibold uppercase">
            {user.department}
          </span>
        )}
      </div>
      <div className="space-y-1">
        <button
          onClick={() => navigate("/system-config")}
          className="w-full text-left px-2 py-1.5 rounded-lg text-xs hover:bg-gray-50 dark:hover:bg-gray-800 text-slate-700 dark:text-slate-200 flex items-center gap-2"
        >
          <span className="material-symbols-outlined text-[16px]">settings</span>
          Cấu hình hệ thống
        </button>
        <button
          onClick={handleLogout}
          className="w-full text-left px-2 py-1.5 rounded-lg text-xs hover:bg-red-50 dark:hover:bg-red-950/20 text-red-600 dark:text-red-400 flex items-center gap-2"
        >
          <span className="material-symbols-outlined text-[16px] text-red-500">logout</span>
          Đăng xuất tài khoản
        </button>
      </div>
    </div>
  );

  return (
    <header className="flex justify-between items-center w-full px-gutter h-16 sticky top-0 z-30 bg-white/70 dark:bg-[#232736]/70 backdrop-blur-xl border-b border-outline-variant/30 shadow-sm shrink-0">
      {/* Search Bar */}
      <div className="flex-1 max-w-md">
        <div className="relative group">
          <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-outline group-focus-within:text-primary transition-colors">
            search
          </span>
          <input
            value={searchValue}
            onChange={handleSearchChange}
            className="w-full bg-surface-container-low border border-outline-variant/30 rounded-full py-2 pl-10 pr-4 text-body-md font-body-md text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all placeholder-gray-400"
            placeholder="Tìm kiếm bệnh nhân hoặc mã quét..."
            type="text"
          />
        </div>
      </div>

      {/* Trailing Actions */}
      <div className="flex items-center gap-stack-sm">
        {/* Notification popover */}
        <Popover 
          content={notificationContent} 
          trigger="click" 
          placement="bottomRight"
          overlayClassName="premium-notification-popover"
        >
          <button className="p-2 rounded-full text-on-surface-variant hover:bg-primary-container/10 hover:text-primary transition-colors scale-95 active:opacity-80 relative">
            <Badge count={unreadCount} size="small" offset={[2, -2]}>
              <span className="material-symbols-outlined text-[24px]">notifications</span>
            </Badge>
          </button>
        </Popover>

        {/* System Settings Button */}
        <button 
          onClick={() => navigate("/system-config")}
          className="p-2 rounded-full text-on-surface-variant hover:bg-primary-container/10 hover:text-primary transition-colors scale-95 active:opacity-80"
        >
          <span className="material-symbols-outlined text-[24px]">settings</span>
        </button>

        <div className="h-8 w-px bg-outline-variant/30 mx-2"></div>

        {/* Profile Avatar Popover */}
        <Popover content={profileContent} trigger="click" placement="bottomRight">
          <button 
            className="w-8 h-8 rounded-full bg-blue-500 border border-outline-variant/30 overflow-hidden scale-95 active:opacity-80 transition-all focus:ring-2 focus:ring-primary focus:ring-offset-2 flex items-center justify-center text-white font-bold text-sm"
          >
            {user?.avatar ? (
              <img
                alt="User profile avatar"
                className="w-full h-full object-cover"
                src={user.avatar}
              />
            ) : (
              (user?.name ? user.name.charAt(0).toUpperCase() : "D")
            )}
          </button>
        </Popover>
      </div>
    </header>
  );
};

export default Topbar;

