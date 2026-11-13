import { diseaseLabel, formatNumber } from '../../../utils/display';
import { useState } from 'react';
import { Alert, Button, Empty, Select, Spin } from 'antd';
import { useGetResearchStatsQuery } from '../services/researchApi';

export default function ResearchPortalPage() {
    const [weeks, setWeeks] = useState(8);
    const { currentData: data, error, isLoading, isFetching, refetch } = useGetResearchStatsQuery(weeks);
    const summary = data?.summary;
    const rows: { weekStart: string; weekEnd: string; data: { _id: string; count: number }[] }[] = data?.trends || [];
    const locations: { location: string; disease: string; count: number }[] = data?.locations || [];
    const date = (value: string) => new Date(value).toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' });
    const exportCsv = () => {
        const content = [
            ['Tuần bắt đầu', 'Tuần kết thúc', 'Nhãn AI vượt ngưỡng', 'Số lần chụp'],
            ...rows.flatMap(week => week.data.map(item => [date(week.weekStart), date(week.weekEnd), diseaseLabel(item._id), item.count])),
        ].map(row => row.map(value => '"' + String(value).replaceAll('"', '""') + '"').join(',')).join('\r\n');
        const url = URL.createObjectURL(new Blob(['\ufeff' + content], { type: 'text/csv;charset=utf-8' }));
        const link = document.createElement('a');
        link.href = url;
        link.download = 'chexnet-thong-ke-' + weeks + '-tuan.csv';
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    if (isLoading) return <Spin className="m-8" />;
    return <div className="page-shell space-y-6 text-on-surface dark:text-white max-w-6xl mx-auto">
        <div className="flex flex-wrap justify-between gap-3 items-center">
            <h2 className="text-2xl font-semibold">Nghiên cứu và thống kê</h2>
            <div className="flex flex-wrap gap-2">
                <Select aria-label="Khoảng thống kê" value={weeks} onChange={setWeeks} options={[4, 8, 12].map(value => ({ value, label: value + ' tuần' }))} />
                <Button loading={isFetching} onClick={() => refetch()}>Cập nhật</Button>
                <Button onClick={exportCsv} disabled={!data || isFetching}>Xuất CSV</Button>
            </div>
        </div>
        {error && <Alert type="error" showIcon title="Chưa tải được thống kê. Vui lòng thử lại." />}
        {summary && <>
            <p>{date(summary.period.since)} – {date(summary.period.until)} · Thống kê nhãn AI vượt ngưỡng trong các lần chụp hoàn tất.</p>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {[
                    ['Lần chụp hoàn tất', summary.totalScans],
                    ['Bệnh nhân có phim trong kỳ', summary.patientsInPeriod],
                    ['Phiếu đã xác minh', summary.verifiedDiagnoses],
                    ['Thời gian xử lý trung bình', summary.avgProcessingTime == null ? 'Chưa có dữ liệu' : formatNumber(summary.avgProcessingTime, 2) + ' giây'],
                ].map(([label, value]) => <div key={label} className="border rounded-xl p-4 bg-white dark:bg-[#232736]">
                    <p className="text-sm mb-2">{label}</p><p className="text-xl font-semibold">{value}</p>
                </div>)}
            </div>
            {summary.missingDetectionData > 0 && <Alert type="info" title={summary.missingDetectionData + ' lần chụp cũ thiếu dữ liệu nhãn phát hiện, được ghi nhận riêng.'} />}
        </>}
        <section className="space-y-3">
            <h3 className="text-lg font-semibold">Số lần chụp phát hiện từng bệnh</h3>
            <div className="overflow-x-auto">
                <table className="w-full text-sm"><thead><tr className="text-left border-b"><th className="p-2">Khoảng thời gian</th><th>Nhãn và số lần chụp</th></tr></thead>
                    <tbody>{rows.map(week => <tr key={week.weekStart} className="border-b">
                        <td className="p-2 whitespace-nowrap">{date(week.weekStart)} – {date(week.weekEnd)}</td>
                        <td className="p-2">{week.data.length ? week.data.map(item => diseaseLabel(item._id) + ' (' + item.count + ')').join(', ') : 'Không ghi nhận nhãn vượt ngưỡng'}</td>
                    </tr>)}</tbody>
                </table>
            </div>
        </section>
        <section className="space-y-3">
            <h3 className="text-lg font-semibold">Vị trí tổn thương được mô hình ghi nhận</h3>
            {locations.length ? <div className="overflow-x-auto"><table className="w-full text-sm">
                <thead><tr className="text-left border-b"><th className="p-2">Vị trí</th><th>Bệnh</th><th>Số lần chụp</th></tr></thead>
                <tbody>{locations.map((item, index) => <tr key={index} className="border-b"><td className="p-2">{item.location}</td><td>{diseaseLabel(item.disease)}</td><td>{item.count}</td></tr>)}</tbody>
            </table></div> : <Empty description="Chưa có dữ liệu vị trí trong các kết quả của kỳ này" />}
        </section>
        <section className="space-y-3">
            <h3 className="text-lg font-semibold">Cảnh báo từ báo cáo AI</h3>
            {summary?.clinicalAlerts?.length ? summary.clinicalAlerts.map((item: { id: string; message: string }) =>
                <Alert key={item.id} type="warning" title={item.message} />) : <p>Chưa có cảnh báo được ghi nhận trong kỳ.</p>}
        </section>
        <section className="space-y-3">
            <h3 className="text-lg font-semibold">Kết quả tham khảo trong tài liệu dự án</h3>
            <p className="text-sm">Các số liệu dưới đây chưa được gắn với lần kiểm định trực tiếp của mô hình đang chạy.</p>
            {summary?.benchmarks?.map((item: { dataset: string; auroc: number; source: string }) =>
                <p key={item.dataset}>{item.dataset} · AUROC {formatNumber(item.auroc, 4)} · {item.source}</p>)}
        </section>
    </div>;
}
