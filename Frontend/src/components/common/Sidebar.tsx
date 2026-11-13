import { useLocation, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { BarChartOutlined, FileSearchOutlined, HistoryOutlined, MoonOutlined, PlusOutlined, QuestionCircleOutlined, SettingOutlined, SunOutlined } from '@ant-design/icons';
import { toggleTheme } from '../../stores/themeSlice';
import type { RootState } from '../../stores/store';

export default function Sidebar({ onNewAnalysisClick, open = false, onClose }: {
  onNewAnalysisClick?: () => void; open?: boolean; onClose?: () => void;
}) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const mode = useSelector((state: RootState) => state.theme.mode);
  const go = (path: string) => { navigate(path); onClose?.(); };
  const links = [
    { path: '/', label: 'Phân tích X-quang', icon: <FileSearchOutlined aria-hidden="true" /> },
    { path: '/patient-history', label: 'Hồ sơ & lịch sử', icon: <HistoryOutlined aria-hidden="true" /> },
    { path: '/research-portal', label: 'Thống kê', icon: <BarChartOutlined aria-hidden="true" /> },
    { path: '/system-config', label: 'Trạng thái hệ thống', icon: <SettingOutlined aria-hidden="true" /> },
  ];
  return <nav aria-label="Điều hướng chính" className={`workspace-sidebar fixed left-0 top-0 h-full flex flex-col p-4 z-40 w-64 border-r transition-transform ${open ? 'translate-x-0 visible' : '-translate-x-full invisible'} md:translate-x-0 md:visible`}>
    <button className="workspace-brand" onClick={() => go('/')} aria-label="CheXNet – Trang phân tích">
      <span><strong>CheXNet</strong><span>Hệ thống X-quang ngực</span></span>
    </button>
    <button onClick={() => { onClose?.(); onNewAnalysisClick?.(); }} className="workspace-create"><PlusOutlined aria-hidden="true" />Thêm bệnh nhân</button>
    <div className="flex-1 flex flex-col gap-2">{links.map(link => {
      const active = link.path === '/' ? pathname === '/' || pathname === '/diagnostic-hub' : pathname.startsWith(link.path) || link.path === '/patient-history' && pathname.startsWith('/patient-profile');
      return <button aria-label={link.label} key={link.path} aria-current={active ? 'page' : undefined} onClick={() => go(link.path)}
        data-active={active} className="workspace-nav-item">
        <span className="text-lg">{link.icon}</span>{link.label}</button>;
    })}</div>
    <div className="pt-4 mt-6 border-t border-outline-variant/30 space-y-1">
      <button onClick={() => dispatch(toggleTheme())} className="flex items-center gap-3 w-full px-3 py-3 rounded-lg text-sm text-secondary hover:bg-surface-container-high">
        {mode === 'light' ? <MoonOutlined aria-hidden="true" /> : <SunOutlined aria-hidden="true" />}{mode === 'light' ? 'Chuyển sang giao diện tối' : 'Chuyển sang giao diện sáng'}</button>
      <button onClick={() => go('/huong-dan')} aria-current={pathname === '/huong-dan' ? 'page' : undefined} className="flex items-center gap-3 w-full px-3 py-3 rounded-lg text-sm text-secondary hover:bg-surface-container-high"><QuestionCircleOutlined aria-hidden="true" />Hướng dẫn sử dụng</button>
    </div>
  </nav>;
}
