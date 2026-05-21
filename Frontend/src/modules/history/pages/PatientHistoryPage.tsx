import React, { useState, useMemo, useCallback } from "react";
import { Table, Card, Button, Select, message } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useNavigate } from "react-router-dom";
import DebouncedSearchInput from "../../../components/ui/DebouncedSearchInput";
import { useGetPatientsQuery } from "../../../stores/baseApi";

const PatientHistoryPage: React.FC = () => {
  const navigate = useNavigate();

  // Filter States
  const [search, setSearch] = useState("");
  const [dateRange, setDateRange] = useState("30 ngày qua");
  const [pathologyType, setPathologyType] = useState("Tất cả Kết quả");
  const [riskLevel, setRiskLevel] = useState("Mọi Mức rủi ro");

  // Fetching Patients based on search/filters
  const { data: scansList, isLoading, refetch } = useGetPatientsQuery({
    search: search || undefined,
    type: pathologyType !== "Tất cả Kết quả" ? pathologyType : undefined,
    risk: riskLevel !== "Mọi Mức rủi ro" ? riskLevel : undefined,
  });

  const handleExportCSV = useCallback(() => {
    if (!scansList || scansList.length === 0) {
      message.warning("Không có dữ liệu để xuất!");
      return;
    }
    // CSV generation
    const headers = "Patient Name,Patient ID,Date,Scan Type,AI Prediction,Status\n";
    const rows = scansList
      .map((s) => {
        const pred = s.predictions?.[0]
          ? `${s.predictions[0].name} (${s.predictions[0].probability}%)`
          : "Bình thường";
        return `"${s.patientName}","${s.patientId}","${s.date} ${s.time}","${s.type}","${pred}","${s.status}"`;
      })
      .join("\n");

    const blob = new Blob([headers + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `Lich_su_chan_doan_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    message.success("Xuất file CSV thành công!");
  }, [scansList]);

  // Handle pagination & table change
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 5,
  });

  const handleTableChange = useCallback((newPagination: any) => {
    setPagination((prev) => ({
      ...prev,
      current: newPagination.current,
      pageSize: newPagination.pageSize,
    }));
  }, []);

  // Columns definition (useMemo wrapped as per DNA standard)
  const columns: ColumnsType<any> = useMemo(
    () => [
      {
        title: "Hình ảnh",
        dataIndex: "image",
        key: "image",
        width: 80,
        render: (image: string, record: any) => {
          if (!image) {
            return (
              <div className="w-12 h-12 bg-surface-variant dark:bg-gray-700 rounded border border-outline-variant/30 flex items-center justify-center text-outline">
                <span className="material-symbols-outlined">image</span>
              </div>
            );
          }
          return (
            <div className="w-12 h-12 bg-surface-variant dark:bg-gray-700 rounded border border-outline-variant/30 overflow-hidden relative">
              <img
                alt={`Scan of ${record.patientName}`}
                className="w-full h-full object-cover opacity-80"
                src={image}
              />
            </div>
          );
        },
      },
      {
        title: "Bệnh nhân / ID",
        key: "patientInfo",
        render: (_, record) => (
          <div className="flex flex-col">
            <span className="text-label-bold font-label-bold text-on-surface dark:text-white">
              {record.patientName}
            </span>
            <span className="text-[11px] font-mono text-on-surface-variant dark:text-gray-400 mt-0.5">
              {record.patientId}
            </span>
          </div>
        ),
      },
      {
        title: "Ngày chụp",
        key: "scanDate",
        render: (_, record) => (
          <div className="text-on-surface-variant dark:text-gray-300 text-sm">
            {record.date}
            <span className="text-outline dark:text-gray-400 text-xs block">{record.time}</span>
          </div>
        ),
      },
      {
        title: "Loại Scan",
        dataIndex: "type",
        key: "type",
        render: (type: string) => (
          <span className="text-on-surface dark:text-white font-medium text-sm">{type}</span>
        ),
      },
      {
        title: "AI Dự đoán",
        key: "aiPrediction",
        render: (_, record) => {
          const mainPrediction = record.predictions?.[0];
          if (!mainPrediction) {
            return (
              <span className="text-on-surface-variant dark:text-gray-400 text-sm">
                Không phát hiện bất thường
              </span>
            );
          }
          return (
            <span className="text-error dark:text-red-400 font-semibold flex items-center gap-1 text-sm">
              <span className="material-symbols-outlined text-[16px]">priority_high</span>
              {mainPrediction.name} ({mainPrediction.probability}%)
            </span>
          );
        },
      },
      {
        title: "Trạng thái",
        dataIndex: "status",
        key: "status",
        render: (status: string) => {
          if (status === "Đang chờ") {
            return (
              <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-surface-container dark:bg-gray-800 text-on-surface-variant dark:text-gray-200 border border-outline-variant/50">
                <span className="material-symbols-outlined text-[14px] mr-1 animate-spin">sync</span>
                Đang chờ
              </span>
            );
          }
          if (status === "Đánh dấu") {
            return (
              <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-error-container text-on-error-container border border-error-container/50">
                <span className="w-1.5 h-1.5 rounded-full bg-error mr-1.5"></span>
                Đánh dấu
              </span>
            );
          }
          // Default: verified/stable/evaluated
          return (
            <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-secondary-container text-on-secondary-container border border-secondary-container/50">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1.5"></span>
              {status || "Đã xác minh"}
            </span>
          );
        },
      },
      {
        title: "Thao tác",
        key: "action",
        align: "right",
        render: (_, record) => (
          <button
            onClick={() => navigate(`/patient-profile/${record.patientId}`)}
            className="p-1.5 text-on-surface-variant dark:text-gray-300 hover:text-primary dark:hover:text-[#2563eb] hover:bg-primary-container/10 dark:hover:bg-gray-800 rounded transition-colors"
            title="Xem hồ sơ chi tiết"
          >
            <span className="material-symbols-outlined text-[20px]">visibility</span>
          </button>
        ),
      },
    ],
    [navigate]
  );

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Header Section */}
      <header className="px-margin py-stack-lg flex justify-between items-end border-b border-outline-variant/20 bg-surface-container-lowest/50 backdrop-blur-sm sticky top-0 z-30">
        <div>
          <h2 className="text-headline-lg font-headline-lg text-on-surface dark:text-white tracking-tight">
            Lịch sử Phân tích X-quang Ngực
          </h2>
          <p className="text-body-md font-body-md text-on-surface-variant dark:text-gray-300 mt-1">
            Quản lý và tra cứu hồ sơ phân tích chẩn đoán hình ảnh phổi chuyên sâu.
          </p>
        </div>
        <Button
          onClick={handleExportCSV}
          className="h-10 px-4 border border-outline dark:border-gray-600 bg-white dark:bg-gray-800 text-on-surface dark:text-white rounded-lg text-label-bold font-label-bold flex items-center gap-2 hover:bg-surface-container-low transition-colors shadow-sm"
        >
          <span className="material-symbols-outlined text-[18px]">download</span>
          Xuất CSV
        </Button>
      </header>

      {/* Content Scrollable Area */}
      <div className="p-margin overflow-y-auto flex-1">
        {/* Robust Filter Bar */}
        <div className="bg-white dark:bg-[#232736] border border-outline-variant/30 rounded-xl p-stack-sm mb-stack-md flex flex-wrap items-center gap-stack-md shadow-sm">
          {/* Search bar using custom debounced component */}
          <div className="flex-1 min-w-[280px]">
            <DebouncedSearchInput
              value={search}
              onChange={setSearch}
              placeholder="Tìm kiếm theo Tên bệnh nhân hoặc Mã hồ sơ..."
            />
          </div>

          <div className="w-px h-8 bg-outline-variant/30 hidden md:block"></div>

          {/* Selector filters */}
          <div className="flex flex-wrap gap-stack-sm items-center">
            {/* Time interval filter */}
            <div className="relative">
              <Select
                value={dateRange}
                onChange={setDateRange}
                className="w-36 text-label-bold font-label-bold"
                variant="outlined"
                options={[
                  { value: "30 ngày qua", label: "30 ngày qua" },
                  { value: "7 ngày qua", label: "7 ngày qua" },
                  { value: "Năm nay", label: "Năm nay" },
                ]}
              />
            </div>

            {/* Pathology Type filter */}
            <div className="relative">
              <Select
                value={pathologyType}
                onChange={setPathologyType}
                className="w-44 text-label-bold font-label-bold"
                options={[
                  { value: "Tất cả Kết quả", label: "Tất cả Kết quả" },
                  { value: "Tràn dịch màng phổi", label: "Tràn dịch phổi" },
                  { value: "Nốt mờ / Khối u", label: "Nốt mờ / Khối u" },
                  { value: "Viêm phổi", label: "Viêm phổi" },
                  { value: "Bình thường", label: "Bình thường" },
                ]}
              />
            </div>

            {/* Risk level filter */}
            <div className="relative">
              <Select
                value={riskLevel}
                onChange={setRiskLevel}
                className="w-40 text-label-bold font-label-bold"
                options={[
                  { value: "Mọi Mức rủi ro", label: "Mọi Mức rủi ro" },
                  { value: "Nghiêm trọng", label: "Nghiêm trọng" },
                  { value: "Trung bình", label: "Trung bình" },
                  { value: "Thấp", label: "Thấp" },
                ]}
              />
            </div>

            {/* Refetch/Reload Button */}
            <button
              onClick={() => {
                refetch();
                message.success("Đã làm mới danh sách chẩn đoán.");
              }}
              className="p-2 border border-outline-variant/50 rounded-lg hover:bg-surface-container-high dark:hover:bg-gray-700 transition-colors flex items-center justify-center shrink-0"
              title="Làm mới"
            >
              <span className="material-symbols-outlined text-[20px] text-primary dark:text-[#2563eb]">autorenew</span>
            </button>
          </div>
        </div>

        {/* High Density Table Container */}
        <Card
          className="rounded-[2rem] border-none shadow-xl shadow-gray-200/40 dark:shadow-none overflow-hidden"
          bodyStyle={{ padding: 0 }}
        >
          <Table
            columns={columns}
            dataSource={scansList || []}
            rowKey={(record) => `${record.patientId}-${record.id}`}
            loading={isLoading}
            pagination={{
              ...pagination,
              total: scansList?.length || 0,
              showSizeChanger: true,
              pageSizeOptions: ["5", "10", "20"],
              showTotal: (total, range) => `${range[0]}-${range[1]} của ${total} hồ sơ`,
            }}
            onChange={handleTableChange}
            className="custom-table"
            scroll={{ x: "max-content" }}
          />
        </Card>
      </div>
    </div>
  );
};

export default PatientHistoryPage;
