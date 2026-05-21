import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { toggleTheme } from "../../stores/themeSlice";
import type { RootState } from "../../stores/store";

interface SidebarProps {
  onNewAnalysisClick?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onNewAnalysisClick }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const themeMode = useSelector((state: RootState) => state.theme.mode);

  const isActive = (path: string) => {
    if (path === "/" && (location.pathname === "/" || location.pathname === "/diagnostic-hub")) {
      return true;
    }
    return location.pathname.startsWith(path) && path !== "/";
  };

  const navItems = [
    {
      path: "/",
      label: "Diagnostic Hub",
      icon: "monitor_heart",
    },
    {
      path: "/patient-history",
      label: "Patient History",
      icon: "history",
    },
    {
      path: "/research-portal",
      label: "Research Portal",
      icon: "science",
    },
    {
      path: "/system-config",
      label: "System Config",
      icon: "settings",
    },
  ];

  return (
    <nav aria-label="Sidebar Navigation" className="fixed left-0 top-0 h-full flex flex-col p-stack-md z-40 bg-surface w-64 border-r border-outline-variant/30 shadow-md">
      {/* Header */}
      <div className="flex items-center gap-stack-sm mb-stack-lg px-2 pt-2">
        <div className="w-10 h-10 rounded-lg bg-primary-container text-on-primary-container flex items-center justify-center font-headline-md text-headline-md font-bold shrink-0">
          <span className="material-symbols-outlined text-[28px] icon-fill text-[#2563eb]">pulmonology</span>
        </div>
        <div>
          <h1 className="text-headline-md font-headline-md font-extrabold text-primary tracking-tight">CheXNet</h1>
          <p className="text-label-sm font-label-sm text-secondary">X-quang ngực · Bệnh phổi</p>
        </div>
      </div>

      {/* CTA Button */}
      <button
        onClick={() => {
          if (onNewAnalysisClick) {
            onNewAnalysisClick();
          } else {
            navigate("/");
          }
        }}
        className="w-full bg-primary text-on-primary text-label-bold font-label-bold py-3 rounded-lg mb-stack-lg shadow-sm hover:opacity-90 transition-all flex items-center justify-center gap-2"
      >
        <span className="material-symbols-outlined text-[20px]">add</span>
        New Analysis
      </button>

      {/* Main Navigation Links */}
      <div className="flex-1 flex flex-col gap-1">
        {navItems.map((item) => {
          const active = isActive(item.path);
          return (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-left text-label-bold font-label-bold transition-all duration-150 ${
                active
                  ? "bg-primary/10 text-primary font-bold translate-x-1"
                  : "text-on-secondary-container hover:bg-surface-container-high hover:bg-surface-container-highest"
              }`}
            >
              <span className={`material-symbols-outlined text-[22px] ${active ? "icon-fill" : ""}`}>
                {item.icon}
              </span>
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>

      {/* Footer Navigation and Theme Switcher */}
      <div className="mt-auto pt-stack-md border-t border-outline-variant/30 flex flex-col gap-1">
        {/* Theme Switcher Button */}
        <button
          onClick={() => dispatch(toggleTheme())}
          className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-left text-label-bold font-label-bold text-on-secondary-container hover:bg-surface-container-high transition-all"
        >
          <span className="material-symbols-outlined text-[22px]">
            {themeMode === "light" ? "dark_mode" : "light_mode"}
          </span>
          <span>{themeMode === "light" ? "Giao diện Tối" : "Giao diện Sáng"}</span>
        </button>

        <a
          href="#"
          onClick={(e) => e.preventDefault()}
          className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-on-secondary-container hover:bg-surface-container-high transition-all text-label-sm font-label-sm"
        >
          <span className="material-symbols-outlined text-[18px]">help_outline</span>
          <span>Support</span>
        </a>
        <a
          href="#"
          onClick={(e) => e.preventDefault()}
          className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-on-secondary-container hover:bg-surface-container-high transition-all text-label-sm font-label-sm"
        >
          <span className="material-symbols-outlined text-[18px]">description</span>
          <span>Documentation</span>
        </a>
      </div>
    </nav>
  );
};

export default Sidebar;
