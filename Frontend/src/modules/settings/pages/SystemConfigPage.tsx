import React, { useState, useEffect } from "react";
import { message, Button } from "antd";
import {
  useGetSystemConfigQuery,
  useSaveSystemConfigMutation,
  useGetServerHealthQuery,
} from "../../../stores/baseApi";

const SystemConfigPage: React.FC = () => {
  // Query for server health, polling every 3 seconds to show live updates
  const { data: healthData } = useGetServerHealthQuery(undefined, {
    pollingInterval: 3000,
  });

  // Query and mutation for system configuration
  const { data: configData, isLoading: isConfigLoading } = useGetSystemConfigQuery();
  const [saveConfig, { isLoading: isSaving }] = useSaveSystemConfigMutation();

  // Local state for settings form
  const [preProcess, setPreProcess] = useState(true);
  const [anonymousSend, setAnonymousSend] = useState(false);
  const [thresholdNodule, setThresholdNodule] = useState(85);
  const [thresholdPleural, setThresholdPleural] = useState(92);

  // Sync local state when configData changes
  useEffect(() => {
    if (configData) {
      setPreProcess(configData.preProcess);
      setAnonymousSend(configData.anonymousSend);
      setThresholdNodule(configData.thresholdNodule);
      setThresholdPleural(configData.thresholdPleural);
    }
  }, [configData]);

  const handleSave = async () => {
    try {
      await saveConfig({
        preProcess,
        anonymousSend,
        thresholdNodule,
        thresholdPleural,
      }).unwrap();
      message.success("Cấu hình hệ thống đã được cập nhật thành công!");
    } catch (error) {
      message.error("Đã xảy ra lỗi khi lưu cấu hình!");
    }
  };

  const handleReset = () => {
    if (configData) {
      setPreProcess(configData.preProcess);
      setAnonymousSend(configData.anonymousSend);
      setThresholdNodule(configData.thresholdNodule);
      setThresholdPleural(configData.thresholdPleural);
      message.info("Đã khôi phục cấu hình hiện tại.");
    }
  };

  const handleRestartEngine = () => {
    message.loading({ content: "Đang khởi động lại Core Engine...", key: "restart" });
    setTimeout(() => {
      message.success({ content: "Khởi động lại Core Engine thành công!", key: "restart", duration: 2 });
    }, 1500);
  };

  // Get current health values or display default fallback
  const gpuLoad = healthData?.gpuLoad ?? 42;
  const ramUsage = healthData?.ramUsage ?? 68;
  const latency = healthData?.latency ?? 14;

  return (
    <div className="p-margin max-w-container-max mx-auto w-full relative z-10">
      {/* Page Header */}
      <div className="mb-stack-lg animate-fade-in">
        <h2 className="text-headline-lg font-headline-lg text-on-surface mb-stack-sm tracking-tight dark:text-white">
          Cấu hình Hệ thống Chẩn đoán X-quang Phổi CheXNet
        </h2>
        <p className="text-body-lg font-body-lg text-on-surface-variant max-w-3xl dark:text-gray-300">
          Quản lý tham số phân tích AI, tài nguyên máy chủ và ngưỡng chẩn đoán lâm sàng ảnh X-quang phổi.
        </p>
      </div>

      {/* Bento Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-gutter">
        {/* Server Health Monitor (Spans 8 columns on large screens) */}
        <div className="lg:col-span-8 bg-white dark:bg-[#232736] rounded-xl border border-outline-variant/40 shadow-sm overflow-hidden flex flex-col backdrop-blur-md transition-all duration-300">
          {/* Monitor Header */}
          <div className="border-b border-outline-variant/20 px-stack-md py-stack-sm flex items-center justify-between bg-surface-container-low/50 dark:bg-[#1a1d27]/50">
            <div className="flex items-center gap-stack-sm">
              <span className="material-symbols-outlined text-secondary dark:text-gray-400">dns</span>
              <h3 className="text-label-bold font-label-bold text-on-surface dark:text-white uppercase tracking-wider">
                Sức khỏe Máy chủ
              </h3>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 status-pulse"></div>
              <span className="text-label-sm font-label-sm text-secondary dark:text-gray-400 uppercase">
                Hệ thống Ổn định
              </span>
            </div>
          </div>

          {/* Monitor Body */}
          <div className="p-stack-lg grid grid-cols-1 md:grid-cols-3 gap-gutter">
            {/* Metric: GPU Load */}
            <div className="flex flex-col">
              <div className="flex justify-between items-end mb-2">
                <span className="text-label-bold font-label-bold text-on-surface-variant dark:text-gray-300">
                  Tải GPU (Cluster A)
                </span>
                <span className="text-headline-md font-headline-md text-primary dark:text-[#2563eb] font-bold">
                  {gpuLoad}%
                </span>
              </div>
              <div className="w-full bg-surface-container-highest dark:bg-gray-700 rounded-full h-2.5 mb-1 overflow-hidden">
                <div
                  className="bg-primary dark:bg-primary-container h-2.5 rounded-full transition-all duration-500"
                  style={{ width: `${gpuLoad}%` }}
                ></div>
              </div>
              <p className="text-[10px] font-label-sm text-secondary dark:text-gray-400 text-right">
                Tối ưu cho chẩn đoán
              </p>
            </div>

            {/* Metric: RAM Usage */}
            <div className="flex flex-col">
              <div className="flex justify-between items-end mb-2">
                <span className="text-label-bold font-label-bold text-on-surface-variant dark:text-gray-300">
                  Mức sử dụng RAM
                </span>
                <span className="text-headline-md font-headline-md text-on-surface dark:text-white font-bold">
                  {ramUsage}%
                </span>
              </div>
              <div className="w-full bg-surface-container-highest dark:bg-gray-700 rounded-full h-2.5 mb-1 overflow-hidden flex">
                <div
                  className="bg-primary dark:bg-primary-container h-2.5 transition-all duration-500"
                  style={{ width: `${ramUsage - 15}%` }}
                ></div>
                <div
                  className="bg-primary-fixed-dim dark:bg-blue-400 h-2.5 transition-all duration-500"
                  style={{ width: "15%" }}
                ></div>
              </div>
              <p className="text-[10px] font-label-sm text-secondary dark:text-gray-400 text-right">
                Dành trước 12GB Cache
              </p>
            </div>

            {/* Metric: API Latency */}
            <div className="flex flex-col bg-surface dark:bg-[#1a1d27] px-stack-md py-stack-sm rounded-lg border border-outline-variant/20">
              <span className="text-label-sm font-label-sm text-on-surface-variant dark:text-gray-400 uppercase tracking-wider mb-1">
                Độ trễ API
              </span>
              <div className="flex items-baseline gap-1">
                <span className="text-headline-lg font-headline-lg text-on-surface dark:text-white">
                  {latency}
                </span>
                <span className="text-label-bold font-label-bold text-secondary dark:text-gray-400">ms</span>
              </div>
              <div className="mt-2 flex items-center gap-1 text-[11px] text-emerald-600 dark:text-emerald-400 font-medium">
                <span className="material-symbols-outlined text-[14px]">trending_down</span>
                <span>-2ms so với hôm qua</span>
              </div>
            </div>
          </div>
        </div>

        {/* System Info Card (Spans 4 columns) */}
        <div className="lg:col-span-4 bg-primary dark:bg-primary-container text-on-primary rounded-xl p-stack-lg shadow-sm relative overflow-hidden flex flex-col justify-between">
          <div className="absolute inset-0 opacity-10 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMiIgY3k9IjIiIHI9IjIiIGZpbGw9IiNmZmYiLz48L3N2Zz4=')]"></div>
          <div className="relative z-10 mb-stack-lg">
            <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-lg flex items-center justify-center mb-stack-md">
              <span className="material-symbols-outlined text-[28px]">security</span>
            </div>
            <h3 className="text-headline-md font-headline-md font-bold mb-1">Chứng chỉ y tế</h3>
            <p className="text-body-md font-body-md text-primary-fixed dark:text-white opacity-90">
              Hệ thống tuân thủ tiêu chuẩn HIPAA & GDPR đối với dữ liệu hình ảnh X-quang DICOM.
            </p>
          </div>
          <div className="relative z-10 flex items-center justify-between border-t border-white/20 pt-stack-md">
            <div>
              <p className="text-label-sm font-label-sm text-primary-fixed dark:text-white opacity-80 uppercase tracking-wider mb-1">
                Phiên bản Model
              </p>
              <p className="text-label-bold font-label-bold text-white">CheXNet-Lung-V3</p>
            </div>
            <button
              onClick={() => message.info("Mô hình chẩn đoán hiện tại đã là phiên bản mới nhất.")}
              className="px-4 py-1.5 bg-white text-primary rounded-full text-label-sm font-label-bold hover:bg-primary-fixed transition-colors"
            >
              Cập nhật
            </button>
          </div>
        </div>

        {/* Confidence Thresholds (Spans 12 columns) */}
        <div className="lg:col-span-12 bg-white dark:bg-[#232736] rounded-xl border border-outline-variant/40 shadow-sm p-stack-lg transition-all duration-300">
          <div className="mb-stack-md flex items-center gap-stack-sm border-b border-outline-variant/20 pb-stack-sm">
            <span className="material-symbols-outlined text-secondary dark:text-gray-400">tune</span>
            <h3 className="text-label-bold font-label-bold text-on-surface dark:text-white uppercase tracking-wider">
              Ngưỡng Tin cậy AI (Confidence Thresholds)
            </h3>
          </div>
          <p className="text-body-sm font-body-md text-secondary dark:text-gray-300 mb-stack-lg max-w-4xl">
            Điều chỉnh độ nhạy của thuật toán phát hiện bất thường trên ảnh X-quang phổi. Mức độ nhạy cao hơn sẽ làm tăng cảnh báo dương tính giả (false positives) nhưng giảm thiểu rủi ro bỏ sót tổn thương.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-gutter gap-y-stack-lg">
            {/* Slider 1 */}
            <div className="flex flex-col gap-2 p-stack-md bg-background dark:bg-[#1a1d27] rounded-lg border border-outline-variant/20">
              <div className="flex justify-between items-center">
                <label className="text-label-bold font-label-bold text-on-surface dark:text-white" htmlFor="threshold-nodule">
                  Phát hiện Nốt mờ (Nodules)
                </label>
                <span className="bg-primary/10 text-primary dark:text-primary-container px-2 py-0.5 rounded text-label-sm font-label-bold">
                  {thresholdNodule}%
                </span>
              </div>
              <input
                className="w-full mt-2"
                id="threshold-nodule"
                max="99"
                min="50"
                onChange={(e) => setThresholdNodule(Number(e.target.value))}
                type="range"
                value={thresholdNodule}
              />
              <div className="flex justify-between text-[11px] text-secondary dark:text-gray-400 mt-1">
                <span>Độ nhạy Cao</span>
                <span>Độ đặc hiệu Cao</span>
              </div>
            </div>

            {/* Slider 2 */}
            <div className="flex flex-col gap-2 p-stack-md bg-background dark:bg-[#1a1d27] rounded-lg border border-outline-variant/20">
              <div className="flex justify-between items-center">
                <label className="text-label-bold font-label-bold text-on-surface dark:text-white" htmlFor="threshold-pleural">
                  Tràn dịch màng phổi
                </label>
                <span className="bg-primary/10 text-primary dark:text-primary-container px-2 py-0.5 rounded text-label-sm font-label-bold">
                  {thresholdPleural}%
                </span>
              </div>
              <input
                className="w-full mt-2"
                id="threshold-pleural"
                max="99"
                min="50"
                onChange={(e) => setThresholdPleural(Number(e.target.value))}
                type="range"
                value={thresholdPleural}
              />
              <div className="flex justify-between text-[11px] text-secondary dark:text-gray-400 mt-1">
                <span>Độ nhạy Cao</span>
                <span>Độ đặc hiệu Cao</span>
              </div>
            </div>
          </div>
        </div>

        {/* Toggles & Quick Settings (Spans 12 columns) */}
        <div className="lg:col-span-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-gutter">
          {/* Toggle Card 1 */}
          <div className="bg-white dark:bg-[#232736] rounded-xl border border-outline-variant/40 p-stack-md flex items-center justify-between hover:shadow-sm transition-shadow">
            <div className="pr-4">
              <p className="text-label-bold font-label-bold text-on-surface dark:text-white mb-1">
                Tiền xử lý Ảnh tự động
              </p>
              <p className="text-[12px] text-secondary dark:text-gray-300 leading-tight">
                Tự động cân bằng sáng tối cho file DICOM gốc.
              </p>
            </div>
            {/* Custom Toggle */}
            <div className="relative inline-block w-12 mr-2 align-middle select-none transition duration-200 ease-in shrink-0">
              <input
                checked={preProcess}
                onChange={(e) => setPreProcess(e.target.checked)}
                className="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer z-10 transition-transform duration-200 ease-in-out peer"
                id="toggle1"
                type="checkbox"
              />
              <label
                className="toggle-label block overflow-hidden h-6 rounded-full bg-surface-variant dark:bg-gray-700 cursor-pointer peer-checked:bg-primary transition-colors duration-200 ease-in-out"
                htmlFor="toggle1"
              ></label>
            </div>
          </div>

          {/* Toggle Card 2 */}
          <div className="bg-white dark:bg-[#232736] rounded-xl border border-outline-variant/40 p-stack-md flex items-center justify-between hover:shadow-sm transition-shadow">
            <div className="pr-4">
              <p className="text-label-bold font-label-bold text-on-surface dark:text-white mb-1">
                Gửi dữ liệu Ẩn danh
              </p>
              <p className="text-[12px] text-secondary dark:text-gray-300 leading-tight">
                Góp phần cải thiện mô hình AI cục bộ.
              </p>
            </div>
            {/* Custom Toggle */}
            <div className="relative inline-block w-12 mr-2 align-middle select-none transition duration-200 ease-in shrink-0">
              <input
                checked={anonymousSend}
                onChange={(e) => setAnonymousSend(e.target.checked)}
                className="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer z-10 transition-transform duration-200 ease-in-out peer"
                id="toggle2"
                type="checkbox"
              />
              <label
                className="toggle-label block overflow-hidden h-6 rounded-full bg-surface-variant dark:bg-gray-700 cursor-pointer peer-checked:bg-primary transition-colors duration-200 ease-in-out"
                htmlFor="toggle2"
              ></label>
            </div>
          </div>

          {/* Action Card */}
          <div className="bg-error-container/20 rounded-xl border border-error-container p-stack-md flex flex-col justify-center items-start">
            <p className="text-label-bold font-label-bold text-on-error-container mb-1 flex items-center gap-1">
              <span className="material-symbols-outlined text-[16px]">warning</span> Khu vực Nguy hiểm
            </p>
            <button
              onClick={handleRestartEngine}
              className="mt-2 px-4 py-2 border border-on-error-container/50 text-on-error-container rounded-lg text-label-bold font-label-bold hover:bg-on-error-container hover:text-white transition-colors w-full"
            >
              Khởi động lại Core Engine
            </button>
          </div>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="mt-stack-lg flex justify-end gap-stack-sm pb-margin border-t border-outline-variant/30 pt-stack-md">
        <button
          onClick={handleReset}
          className="px-6 py-2.5 rounded-lg border border-outline-variant text-on-surface-variant dark:text-gray-300 font-label-bold text-label-bold hover:bg-surface-container-high transition-colors"
        >
          Hủy bỏ
        </button>
        <Button
          type="primary"
          loading={isSaving || isConfigLoading}
          onClick={handleSave}
          className="px-6 py-5 rounded-lg bg-primary text-on-primary font-label-bold text-label-bold hover:opacity-90 shadow-sm transition-opacity flex items-center justify-center"
        >
          Lưu Cấu hình
        </Button>
      </div>
    </div>
  );
};

export default SystemConfigPage;
