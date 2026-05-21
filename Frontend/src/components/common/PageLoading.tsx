import React from "react";
import { Spin } from "antd";
import { LoadingOutlined } from "@ant-design/icons";

const PageLoading: React.FC = () => {
  const antIcon = <LoadingOutlined style={{ fontSize: 48 }} spin className="text-primary" />;

  return (
    <div className="flex flex-col items-center justify-center w-full h-[80vh]">
      <Spin indicator={antIcon} />
      <p className="mt-4 text-gray-500 dark:text-gray-400 font-medium animate-pulse">
        Đang tải dữ liệu...
      </p>
    </div>
  );
};

export default PageLoading;
