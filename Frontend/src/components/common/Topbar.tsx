import React, { useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";
import type { RootState } from "../../stores/store";
import { markAllAsRead } from "../../stores/notificationSlice";
import { logout } from "../../stores/authSlice";
import { Badge, Popover, List, Typography, message } from "antd";
import { BellOutlined, DownOutlined, MenuOutlined, SearchOutlined, UserOutlined } from '@ant-design/icons';

const { Text } = Typography;

interface TopbarProps {
  onSearch?: (value: string) => void;
  onMenuClick?: () => void;
}

const Topbar: React.FC<TopbarProps> = ({ onSearch, onMenuClick }) => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [searchValue, setSearchValue] = useState("");
  const notifications = useSelector((state: RootState) => state.notification.items);
  const user = useSelector((state: RootState) => state.auth.user);
  const roleLabel = user?.role === 'admin' ? 'Quản trị viên' : user?.role === 'doctor' ? 'Bác sĩ' : 'Người dùng';
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
    <div style={{ width: 'min(320px, calc(100vw - 48px))' }}>
      <div className="flex justify-between items-center border-b border-gray-100 pb-2 mb-2 dark:border-gray-800">
        <span className="font-bold text-on-surface">Thông báo</span>
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
          <span aria-hidden="true" className="material-symbols-outlined text-[16px]">settings</span>
          Trạng thái hệ thống
        </button>
        <button
          onClick={handleLogout}
          className="w-full text-left px-2 py-1.5 rounded-lg text-xs hover:bg-red-50 dark:hover:bg-red-950/20 text-red-600 dark:text-red-400 flex items-center gap-2"
        >
          <span aria-hidden="true" className="material-symbols-outlined text-[16px] text-red-500">logout</span>
          Đăng xuất tài khoản
        </button>
      </div>
    </div>
  );

  return (
    <header className="workspace-topbar">
      <button aria-label="Mở menu điều hướng" onClick={onMenuClick} className="workspace-icon-button workspace-menu"><MenuOutlined aria-hidden="true" /></button>
      {/* Search Bar */}
      <form role="search" onSubmit={event => { event.preventDefault(); navigate(`/patient-history?search=${encodeURIComponent(searchValue.trim())}`); }} className="workspace-search">
          <SearchOutlined aria-hidden="true" className="workspace-search-icon" />
          <input
            value={searchValue}
            aria-label="Tìm tên hoặc mã bệnh nhân"
            onChange={handleSearchChange}
            placeholder="Tìm tên hoặc mã bệnh nhân…"
            type="search"
          />
          <button type="submit" aria-label="Tìm kiếm bệnh nhân" className="workspace-search-submit">Tìm</button>
      </form>

      {/* Trailing Actions */}
      <div className="workspace-actions">
        {/* Notification popover */}
        <Popover 
          content={notificationContent} 
          trigger="click" 
          placement="bottomRight"
          classNames={{ root: 'premium-notification-popover' }}
        >
          <button aria-label="Xem thông báo" className="workspace-icon-button">
            <Badge count={unreadCount} size="small" offset={[2, -2]}>
              <BellOutlined aria-hidden="true" />
            </Badge>
          </button>
        </Popover>
        <span className="workspace-action-divider" aria-hidden="true" />

        {/* Profile Avatar Popover */}
        <Popover content={profileContent} trigger="click" placement="bottomRight">
          <button 
            aria-label="Mở tài khoản"
            className="workspace-account"
          >
            <span className="workspace-avatar" aria-hidden="true">
            {user?.avatar ? (
              <img
                alt="Ảnh đại diện tài khoản"
                className="w-full h-full object-cover"
                src={user.avatar}
              />
            ) : (
              <UserOutlined />
            )}
            </span>
            <span className="workspace-account-text"><strong title={user?.name}>{user?.name || 'Tài khoản'}</strong><span>{roleLabel}</span></span>
            <DownOutlined aria-hidden="true" className="workspace-account-arrow" />
          </button>
        </Popover>
      </div>
    </header>
  );
};

export default Topbar;

