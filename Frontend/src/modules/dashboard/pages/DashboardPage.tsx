import React, { useState, useMemo, useEffect } from "react";
import { message, Button, Spin, Select, Switch, Progress } from "antd";
import {
  useGetPatientsQuery,
  useGetPatientProfileQuery,
  useVerifyScanMutation,
} from "../../../stores/baseApi";

const DashboardPage: React.FC = () => {
  // Available patients query for the case switcher
  const { data: allPatients } = useGetPatientsQuery();

  const [activePatientId, setActivePatientId] = useState("");

  const patientOptions = useMemo(() => {
    if (!allPatients?.length) return [];
    const seen = new Set<string>();
    return allPatients
      .filter((p: { patientId: string }) => {
        if (seen.has(p.patientId)) return false;
        seen.add(p.patientId);
        return true;
      })
      .map((p: { patientId: string; patientName: string }) => ({
        value: p.patientId,
        label: p.patientName,
      }));
  }, [allPatients]);

  useEffect(() => {
    if (!patientOptions.length) {
      setActivePatientId("");
      return;
    }
    if (!activePatientId || !patientOptions.some((o) => o.value === activePatientId)) {
      setActivePatientId(patientOptions[0].value);
    }
  }, [patientOptions, activePatientId]);

  const { data: patient, isLoading, refetch } = useGetPatientProfileQuery(activePatientId, {
    skip: !activePatientId,
  });

  // Verification mutation
  const [verifyScan, { isLoading: isVerifying }] = useVerifyScanMutation();

  // Active scan is always the primary scan (index 0) needing diagnosis
  const activeScan = useMemo(() => {
    return patient?.scans?.[0] || null;
  }, [patient]);

  // Image display controls
  const [zoomScale, setZoomScale] = useState(1);
  const [contrastVal, setContrastVal] = useState(100);
  const [isInverted, setIsInverted] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);

  // Animations for probability values
  const [animateWidths, setAnimateWidths] = useState(false);

  useEffect(() => {
    setAnimateWidths(false);
    const timer = setTimeout(() => setAnimateWidths(true), 150);
    return () => clearTimeout(timer);
  }, [activePatientId]);

  // Reset controls when case changes
  useEffect(() => {
    setZoomScale(1);
    setContrastVal(100);
    setIsInverted(false);
    setShowHeatmap(true);
  }, [activePatientId]);

  const handleVerify = async (status: "Đã xác minh" | "Đánh dấu") => {
    if (!activeScan) return;
    try {
      await verifyScan({
        patientId: activePatientId,
        scanId: activeScan.id,
        status,
      }).unwrap();
      message.success(`Hồ sơ chẩn đoán đã được cập nhật: ${status}`);
      refetch();
    } catch (e) {
      message.error("Lỗi khi cập nhật hồ sơ chẩn đoán.");
    }
  };

  const handleZoomIn = () => {
    setZoomScale((prev) => Math.min(prev + 0.25, 2.5));
  };

  const handleZoomOut = () => {
    setZoomScale((prev) => Math.max(prev - 0.25, 1));
  };

  const handleToggleContrast = () => {
    setContrastVal((prev) => (prev === 100 ? 180 : prev === 180 ? 250 : 100));
  };

  const handleToggleInvert = () => {
    setIsInverted((prev) => !prev);
  };

  const defaultScanImage =
    "https://images.unsplash.com/photo-1530497610245-94d3c16cda28?q=80&w=1600&auto=format&fit=crop";

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-background dark:bg-[#1a1d27]">
      {/* Dashboard Workspace */}
      <div className="flex-1 p-gutter flex flex-col lg:flex-row gap-gutter overflow-hidden">
        {/* Left Column: Massive X-Ray Viewer */}
        <section className="flex-[2] flex flex-col bg-white dark:bg-[#232736] border border-outline-variant/30 rounded-xl shadow-sm overflow-hidden relative">
          {/* Viewer Header */}
          <div className="px-stack-md py-stack-sm border-b border-outline-variant/30 bg-white/50 dark:bg-[#1a1d27]/50 backdrop-blur-md flex flex-wrap justify-between items-center z-10 gap-3">
            <div className="flex items-center gap-stack-sm">
              <span className="material-symbols-outlined text-primary dark:text-[#2563eb]">personal_injury</span>
              <div>
                <h2 className="text-label-bold font-label-bold text-on-surface dark:text-white flex items-center gap-2">
                  <span>Ca bệnh: {patient?.name || "Đang tải..."}</span>
                  <span className="text-xs text-outline dark:text-gray-400 font-mono">({activePatientId})</span>
                </h2>
                <p className="text-[11px] text-secondary dark:text-gray-400">
                  Quét: {activeScan?.date || "Không rõ"} • {activeScan?.time || ""}
                </p>
              </div>
            </div>

            {/* Case Selector and Heatmap Switch */}
            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold text-on-surface dark:text-gray-200">AI Heatmap:</span>
              <Switch checked={showHeatmap} onChange={setShowHeatmap} size="small" />

              <div className="w-[1px] h-6 bg-outline-variant/50"></div>

              <span className="text-xs font-semibold text-on-surface dark:text-gray-200">Chọn Ca bệnh:</span>
              <Select
                value={activePatientId}
                onChange={setActivePatientId}
                size="small"
                className="w-40 mr-1"
                options={patientOptions}
                disabled={patientOptions.length === 0}
                placeholder={patientOptions.length === 0 ? "Chưa có bệnh nhân" : undefined}
              />

              <Button
                type="primary"
                size="small"
                onClick={() => window.dispatchEvent(new CustomEvent('open-add-patient-modal'))}
                className="bg-primary hover:opacity-90 flex items-center gap-1 h-[24px]"
              >
                <span className="material-symbols-outlined text-[14px]">add</span>
                Thêm ca bệnh
              </Button>
            </div>
          </div>

          {/* Main Image Area */}
          <div className="flex-1 bg-[#050505] relative overflow-hidden flex items-center justify-center">
            {isLoading ? (
              <div className="text-white flex flex-col items-center gap-2">
                <Spin size="large" />
                <span className="text-sm opacity-85">Đang chạy thuật toán AI...</span>
              </div>
            ) : (
              <>
                {/* X-ray Canvas */}
                <div
                  className="w-full h-full bg-contain bg-center bg-no-repeat transition-all duration-300"
                  style={{
                    backgroundImage: `url('${activeScan?.image || defaultScanImage}')`,
                    transform: `scale(${zoomScale})`,
                    filter: `contrast(${contrastVal}%) invert(${isInverted ? 1 : 0})`,
                  }}
                />

                {/* AI Bounding Box Overlay Simulation */}
                {showHeatmap && activeScan && activeScan.predictions?.map((pred: any, idx: number) => {
                  if (pred.probability < 50) return null;
                  return (
                    <div
                      key={idx}
                      className="absolute top-1/4 left-1/3 w-[35%] h-[40%] border-2 border-error border-dashed rounded-lg bg-error/15 pointer-events-none animate-pulse flex flex-col justify-between p-2 shadow-2xl"
                    >
                      <div className="self-start bg-error text-white text-[10px] font-bold px-2 py-0.5 rounded shadow">
                        {pred.name}: {pred.probability}%
                      </div>
                      <div className="self-end bg-error/80 text-white text-[9px] px-1.5 py-0.5 rounded">
                        Độ nhạy AI: Cao
                      </div>
                    </div>
                  );
                })}

                {/* Status Watermark */}
                <div className="absolute top-3 left-3 bg-black/60 backdrop-blur border border-white/10 rounded px-2.5 py-1 text-white text-[10px] uppercase font-mono tracking-widest pointer-events-none flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping"></span>
                  DICOM LIVE: ZOOM x{zoomScale.toFixed(2)}
                </div>

                {/* Floating Tool Overlay (Glassmorphism) */}
                <div className="absolute right-stack-md top-stack-md bg-white/95 dark:bg-[#232736]/95 backdrop-blur-xl border border-outline-variant/30 rounded-lg p-1 flex flex-col gap-1 shadow-lg transition-all duration-300">
                  <button
                    onClick={handleZoomIn}
                    className="p-2 rounded hover:bg-surface-container-high dark:hover:bg-gray-800 text-on-surface-variant dark:text-gray-300 hover:text-on-surface transition-colors flex items-center justify-center"
                    title="Thu phóng gần"
                  >
                    <span className="material-symbols-outlined">zoom_in</span>
                  </button>
                  <button
                    onClick={handleZoomOut}
                    className="p-2 rounded hover:bg-surface-container-high dark:hover:bg-gray-800 text-on-surface-variant dark:text-gray-300 hover:text-on-surface transition-colors flex items-center justify-center"
                    title="Thu phóng xa"
                  >
                    <span className="material-symbols-outlined">zoom_out</span>
                  </button>
                  <div className="w-full h-px bg-outline-variant/30 my-1"></div>
                  <button
                    onClick={handleToggleContrast}
                    className={`p-2 rounded hover:bg-surface-container-high dark:hover:bg-gray-800 text-on-surface-variant dark:text-gray-300 hover:text-on-surface transition-colors flex items-center justify-center ${
                      contrastVal > 100 ? "text-primary dark:text-[#2563eb] bg-primary/10" : ""
                    }`}
                    title="Tăng tương phản"
                  >
                    <span className="material-symbols-outlined">contrast</span>
                  </button>
                  <button
                    onClick={handleToggleInvert}
                    className={`p-2 rounded hover:bg-surface-container-high dark:hover:bg-gray-800 text-on-surface-variant dark:text-gray-300 hover:text-on-surface transition-colors flex items-center justify-center ${
                      isInverted ? "text-primary dark:text-[#2563eb] bg-primary/10" : ""
                    }`}
                    title="Đảo ngược âm bản"
                  >
                    <span className="material-symbols-outlined">invert_colors</span>
                  </button>
                </div>
              </>
            )}
          </div>
        </section>

        {/* Right Column: Analysis Results (Sleek Panel) */}
        <aside className="w-full lg:w-[420px] flex flex-col gap-stack-md h-auto lg:h-full overflow-y-auto pr-1 pb-2">
          {/* Probability Overview Card */}
          <div className="bg-white dark:bg-[#232736] border border-outline-variant/30 rounded-xl p-stack-md shadow-sm transition-all duration-300">
            <h3 className="text-headline-sm font-headline-md text-on-surface dark:text-white mb-stack-md flex items-center gap-2">
              <span className="material-symbols-outlined text-primary dark:text-[#2563eb]">analytics</span>
              Kết quả phân tích AI
            </h3>

            {isLoading ? (
              <div className="py-6 flex justify-center">
                <Spin size="small" />
              </div>
            ) : (
              <div className="flex flex-col gap-stack-sm">
                {activeScan && activeScan.predictions?.length > 0 ? (
                  activeScan.predictions.map((pred: any, idx: number) => {
                    const isHigh = pred.probability >= 70;
                    return (
                      <div key={idx} className="group">
                        <div className="flex justify-between items-end mb-1 text-sm">
                          <span className={`text-label-bold font-label-bold flex items-center gap-1 ${
                            isHigh ? "text-error dark:text-red-400" : "text-on-surface-variant dark:text-gray-300"
                          }`}>
                            {isHigh && <span className="material-symbols-outlined text-[16px]">warning</span>}
                            {pred.name}
                          </span>
                          <span className={`text-label-bold font-label-bold ${
                            isHigh ? "text-error dark:text-red-400" : "text-on-surface-variant dark:text-gray-300"
                          }`}>
                            {pred.probability}%
                          </span>
                        </div>
                        <Progress
                          percent={animateWidths ? pred.probability : 0}
                          showInfo={false}
                          strokeColor={isHigh ? "#ba1a1a" : "#2563eb"}
                          trailColor="rgba(0,0,0,0.05)"
                          className="mb-0"
                        />
                      </div>
                    );
                  })
                ) : (
                  <div className="text-center py-4 text-on-surface-variant dark:text-gray-400 text-sm">
                    Không phát hiện dấu hiệu bất thường cấp tính.
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Clinical Insight Details (Bento Style Card) */}
          <div className="bg-white dark:bg-[#232736] border border-outline-variant/30 rounded-xl overflow-hidden shadow-sm flex-1 flex flex-col relative transition-all duration-300">
            {/* Warning indicator line at top */}
            <div className="absolute top-0 left-0 w-full h-1 bg-error"></div>
            
            <div className="p-stack-md flex-1 overflow-y-auto">
              <div className="inline-flex items-center gap-1 px-2 py-0.5 bg-error/10 text-error text-[10px] uppercase font-bold rounded mb-3">
                <span className="material-symbols-outlined text-[12px]">priority_high</span> Phát hiện lâm sàng
              </div>

              {isLoading ? (
                <div className="py-12 flex justify-center">
                  <Spin size="small" />
                </div>
              ) : (
                <>
                  <h4 className="text-headline-md font-headline-md text-on-surface dark:text-white mb-2">
                    {activeScan?.predictions?.[0]?.name || "Bình thường"}
                  </h4>
                  <div className="mb-stack-md">
                    <h5 className="text-xs font-bold text-on-surface dark:text-white uppercase tracking-wider mb-1">
                      Mô tả lâm sàng
                    </h5>
                    <p className="text-body-md font-body-md text-secondary dark:text-gray-300 leading-relaxed text-sm">
                      {activeScan?.description || "Hình ảnh chụp phế trường lồng ngực cân đối hai bên. Phổi sáng đều, không thấy tổn thương khu trú bất thường. Cơ hoành và các góc sườn hoành hai bên nhọn, bình thường."}
                    </p>
                  </div>

                  {activeScan?.nextSteps && activeScan.nextSteps.length > 0 && (
                    <div className="bg-surface-container-low dark:bg-[#1a1d27] rounded-lg p-3 border border-outline-variant/30 mt-4">
                      <h5 className="text-xs font-bold text-on-surface dark:text-white uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <span className="material-symbols-outlined text-[16px] text-primary dark:text-[#2563eb]">checklist</span>
                        Khuyến nghị xử trí tiếp theo
                      </h5>
                      <ul className="flex flex-col gap-2 pl-1">
                        {activeScan.nextSteps.map((step: string, idx: number) => (
                          <li key={idx} className="flex items-start gap-2 text-xs text-secondary dark:text-gray-300">
                            <div className="w-1.5 h-1.5 rounded-full bg-primary dark:bg-[#2563eb] mt-1.5 shrink-0"></div>
                            <span>{step}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Action Buttons */}
            <div className="p-stack-md bg-surface-container-low dark:bg-[#1a1d27]/70 border-t border-outline-variant/30 flex gap-2 shrink-0">
              <Button
                onClick={() => handleVerify("Đánh dấu")}
                loading={isVerifying}
                className="flex-1 bg-white dark:bg-gray-800 border border-outline-variant/30 text-on-surface dark:text-white font-label-bold text-label-bold py-5 rounded-lg hover:bg-surface-container transition-colors shadow-sm flex items-center justify-center"
              >
                Đánh dấu nghi vấn
              </Button>
              <Button
                onClick={() => handleVerify("Đã xác minh")}
                loading={isVerifying}
                type="primary"
                className="flex-1 bg-primary text-on-primary font-label-bold text-label-bold py-5 rounded-lg hover:bg-primary-fixed-variant transition-colors shadow-sm flex items-center justify-center"
              >
                Xác minh kết quả
              </Button>
            </div>
          </div>
        </aside>
      </div>

      {/* System Metadata Footer */}
      <footer className="h-10 bg-white dark:bg-[#232736] border-t border-outline-variant/30 flex items-center justify-between px-gutter text-xs text-secondary dark:text-gray-400 shrink-0">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></span>
            Hệ thống AI trực tuyến
          </span>
          <span className="w-px h-3 bg-outline-variant/50 hidden sm:inline-block"></span>
          <span className="hidden sm:inline-block">X-quang ngực · CheXNet · 15 bệnh lý phổi</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1 font-mono">
            <span className="material-symbols-outlined text-[14px]">speed</span> 124ms latency
          </span>
          <span className="w-px h-3 bg-outline-variant/50"></span>
          <span className="font-semibold text-primary dark:text-[#2563eb]">Độ tin cậy: Rất cao</span>
        </div>
      </footer>
    </div>
  );
};

export default DashboardPage;
