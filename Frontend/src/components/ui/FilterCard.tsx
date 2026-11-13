import React from "react";
import { Card, Button } from "antd";
import { ReloadOutlined } from "@ant-design/icons";

interface FilterCardProps {
  children: React.ReactNode;
  onRefresh: () => void;
  loading?: boolean;
}

export const FilterItem: React.FC<{ label: string; children: React.ReactNode; width?: string | number }> = ({
  label, children, width,
}) => (
  <div style={{ width: width || "auto", minWidth: "200px" }} className="flex-1">
    <div className="text-[10px] font-black uppercase text-gray-400 mb-2 tracking-wider ml-1">{label}</div>
    {children}
  </div>
);

const FilterCard: React.FC<FilterCardProps> = ({ children, onRefresh, loading }) => (
  <Card
    className="mb-8 rounded-[2rem] border-t-[6px] border-t-primary shadow-xl shadow-gray-200/40 overflow-hidden border-none glass-card"
    bodyStyle={{ padding: "24px 32px" }}
  >
    <div className="flex flex-wrap items-end gap-8">
      <div className="flex-1 flex flex-wrap items-end gap-8">{children}</div>
      <div className="pb-[2px]">
        <Button
          icon={<ReloadOutlined className="text-orange-500" />}
          onClick={onRefresh} loading={loading}
          className="h-11 px-6 rounded-2xl flex items-center gap-2 font-extrabold text-orange-500 border-2 border-orange-100 bg-orange-50/30 hover:bg-orange-50 hover:border-orange-200 transition-all shadow-sm"
        >
          Làm mới
        </Button>
      </div>
    </div>
  </Card>
);

export default FilterCard;
