import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Alert, Button, Empty, Grid, Pagination, Select, Spin, Table, Tag, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import ScanImage from '../../../components/common/ScanImage';
import DebouncedSearchInput from '../../../components/ui/DebouncedSearchInput';
import { useGetPatientsQuery } from '../../../stores/baseApi';
import { scanTypeLabel } from '../../../utils/display';

export default function PatientHistoryPage() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const screens = Grid.useBreakpoint();
  const [search, setSearch] = useState(params.get('search') || '');
  const [period, setPeriod] = useState('all');
  const [status, setStatus] = useState('all');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const { data: scans = [], isLoading, isFetching, error, refetch } = useGetPatientsQuery({ search: search || undefined });
  useEffect(() => { setSearch(params.get('search') || ''); setPage(1); }, [params]);
  const filtered = useMemo(() => scans.filter(scan => {
    if (status !== 'all' && (status === 'no-scan' ? !!scan.image : !scan.image || scan.status !== status)) return false;
    if (period !== 'all') {
      const created = new Date(scan.createdAt);
      if (Number.isNaN(created.getTime())) return false;
      const since = new Date();
      if (period === 'year') since.setMonth(0, 1), since.setHours(0, 0, 0, 0);
      else since.setTime(since.getTime() - Number(period) * 86400000);
      if (created < since) return false;
    }
    return true;
  }), [scans, period, status]);
  const current = Math.min(page, Math.max(1, Math.ceil(filtered.length / pageSize)));
  const pageRows = filtered.slice((current - 1) * pageSize, current * pageSize);
  const open = (row: typeof scans[number]) => navigate(`/patient-profile/${row.patientId}${row.image ? `?scan=${row.id}` : ''}`);
  const statusText = (row: typeof scans[number]) => !row.image ? 'Chưa có phim chụp' : row.status === 'Đang chờ' ? 'Chờ xác minh' : row.status === 'Đánh dấu' ? 'Cần xem lại' : row.status;
  const predictionText = (row: typeof scans[number]) => !row.image ? 'Chưa phân tích' : row.predictions?.length ? `${row.predictions[0].name} (${row.predictions[0].probability}%)` : 'Không có nhãn vượt ngưỡng';
  const exportCsv = () => {
    if (!filtered.length) return;
    const rows = [['Tên bệnh nhân', 'Mã bệnh nhân', 'Ngày ghi nhận', 'Loại phim', 'Kết quả AI', 'Trạng thái'],
      ...filtered.map(row => [row.patientName, row.patientId, `${row.date} ${row.time}`, scanTypeLabel(row.type), predictionText(row), statusText(row)])];
    const content = rows.map(row => row.map(value => `"${String(value || '').replaceAll('"', '""')}"`).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob(['\ufeff' + content], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = 'chexnet-lich-su.csv'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000); message.success('Đã xuất danh sách đang lọc');
  };
  const columns: ColumnsType<typeof scans[number]> = [
    { title: 'Bệnh nhân', key: 'patient', width: 310, render: (_, row) => <div className="flex items-center gap-3">
      {row.image && <ScanImage src={row.image} alt={`Ảnh X-quang của ${row.patientName}`} className="w-12 h-14 object-contain bg-slate-950 rounded-md shrink-0" />}
      <button className="text-left min-w-0" onClick={() => open(row)}><strong className="block text-primary break-words">{row.patientName}</strong><span className="text-xs text-secondary">{row.patientId}</span></button></div> },
    { title: 'Lần chụp', key: 'date', width: 195, render: (_, row) => <div className="space-y-1"><p>{row.date} · {row.time}</p><p className="text-xs text-secondary">{scanTypeLabel(row.type)}</p></div> },
    { title: 'Kết quả AI', key: 'prediction', render: (_, row) => <span className="text-sm">{predictionText(row)}</span> },
    { title: 'Trạng thái phiếu', key: 'status', width: 165, render: (_, row) => <Tag color={row.status === 'Đã xác minh' ? 'green' : row.status === 'Đánh dấu' ? 'orange' : 'default'}>{statusText(row)}</Tag> },
    { title: 'Thao tác', key: 'action', width: 95, render: (_, row) => <Button onClick={() => open(row)}>Xem</Button> },
  ];
  return <div className="page-shell space-y-5">
    <header className="page-heading"><div><p className="eyebrow">QUẢN LÝ HỒ SƠ</p><h1>Hồ sơ & lịch sử</h1><p className="page-description">Tìm bệnh nhân, mở đúng lần chụp và theo dõi trạng thái xác minh.</p></div>
      <Button onClick={exportCsv} disabled={!filtered.length || isFetching}>Xuất danh sách CSV</Button></header>
    <section className="ui-card grid gap-4 md:grid-cols-[minmax(0,1fr)_180px_200px_auto] items-end" aria-label="Lọc hồ sơ">
      <label className="field-label">Tên hoặc mã bệnh nhân<DebouncedSearchInput aria-label="Lọc theo tên hoặc mã bệnh nhân" value={search} onChange={value => { setSearch(value); setParams(value ? { search: value } : {}); setPage(1); }} placeholder="Nhập tên hoặc mã bệnh nhân" allowClear /></label>
      <label className="field-label">Thời gian<Select aria-label="Lọc thời gian" value={period} onChange={value => { setPeriod(value); setPage(1); }} options={[
        { value: 'all', label: 'Tất cả thời gian' }, { value: '7', label: '7 ngày qua' }, { value: '30', label: '30 ngày qua' }, { value: 'year', label: 'Năm nay' }]} /></label>
      <label className="field-label">Trạng thái<Select aria-label="Lọc trạng thái phiếu" value={status} onChange={value => { setStatus(value); setPage(1); }} options={[
        { value: 'all', label: 'Tất cả trạng thái' }, { value: 'Đang chờ', label: 'Chờ xác minh' }, { value: 'Đã xác minh', label: 'Đã xác minh' }, { value: 'Đánh dấu', label: 'Cần xem lại' }, { value: 'no-scan', label: 'Chưa có phim chụp' }]} /></label>
      <Button onClick={() => refetch()} loading={isFetching}>Làm mới</Button>
    </section>
    {error && <Alert type="error" showIcon title="Chưa tải được danh sách hồ sơ" action={<Button onClick={() => refetch()}>Thử lại</Button>} />}
    <div className="ui-card !p-0 overflow-hidden">
      {screens.md ? <Table className="custom-table" columns={columns} dataSource={pageRows} rowKey={row => `${row.patientId}-${row.id}`} loading={isLoading || isFetching} pagination={false} scroll={{ x: 850 }} locale={{ emptyText: 'Không có hồ sơ phù hợp với bộ lọc' }} /> :
      <div className="divide-y divide-slate-200 dark:divide-slate-700">{isFetching ? <div className="p-8 text-center"><Spin /></div> : pageRows.length ? pageRows.map(row => <article key={row.id} className="p-4 space-y-3">
        <div className="flex items-start gap-3">{row.image && <ScanImage src={row.image} alt="Ảnh X-quang thu nhỏ" className="w-12 h-16 rounded-md object-contain bg-slate-950" />}<div className="min-w-0 flex-1"><h2 className="font-semibold break-words">{row.patientName}</h2><p className="text-xs text-secondary mt-1">{row.patientId} · {row.date}</p></div></div>
        <p className="text-sm">{predictionText(row)}</p><div className="flex items-center justify-between gap-2"><Tag>{statusText(row)}</Tag><Button onClick={() => open(row)}>Xem hồ sơ</Button></div>
      </article>) : <div className="p-8"><Empty description="Không có hồ sơ phù hợp với bộ lọc" /></div>}</div>}
      <div className="p-4 border-t border-outline-variant/30"><Pagination current={current} pageSize={pageSize} total={filtered.length} showSizeChanger pageSizeOptions={[5, 10, 20]}
        onChange={(next, size) => { setPage(next); setPageSize(size); }} showTotal={total => `${total} lần chụp / hồ sơ`} /></div>
    </div>
    {(search || period !== 'all' || status !== 'all') && <Button onClick={() => { setSearch(''); setParams({}); setPeriod('all'); setStatus('all'); setPage(1); }}>Xóa bộ lọc</Button>}
  </div>;
}
