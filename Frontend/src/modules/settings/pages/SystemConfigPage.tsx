import { deviceLabel } from '../../../utils/display';
import { formatAiModelLabel } from '../../../utils/aiModel';
import { Alert, Button, Descriptions, Tag } from 'antd';
import { useGetServerHealthQuery } from '../../../stores/baseApi';

export default function SystemConfigPage() {
    const { data, error, isFetching, refetch } = useGetServerHealthQuery(undefined, { pollingInterval: 15000 });
    const ready = !error && data?.status === 'healthy';
    return (
        <div className="page-shell max-w-4xl mx-auto space-y-5 text-on-surface dark:text-white">
            <div className="flex justify-between gap-3 items-center">
                <h2 className="text-2xl font-semibold">Trạng thái hệ thống AI</h2>
                <Button onClick={() => refetch()} loading={isFetching}>Kiểm tra lại</Button>
            </div>
            {error ? <Alert type="error" showIcon title="Không kết nối được dịch vụ AI" description="Kiểm tra kết nối hệ thống và thử lại." /> :
                <Tag color={ready ? 'green' : 'orange'}>{ready ? 'Sẵn sàng phân tích' : 'Đang chờ mô hình và bộ xác thực'}</Tag>}
            {data && <Descriptions bordered column={1} size="small">
                <Descriptions.Item label="Thời điểm kiểm tra">{new Date(data.checkedAt).toLocaleString('vi-VN')}</Descriptions.Item>
                <Descriptions.Item label="Mô hình phân tích">{data.modelLoaded ? 'Đã nạp' : 'Chưa nạp'} · {formatAiModelLabel(data.modelVersion)}{data.model_file ? ` (${data.model_file})` : ''}</Descriptions.Item>
                <Descriptions.Item label="Thiết bị">{deviceLabel(data.device)}</Descriptions.Item>
                <Descriptions.Item label="Bộ định vị bổ sung (YOLO)">{data.yolo_loaded ? 'Đã nạp (YOLO11m)' : 'Chưa nạp trọng số'}</Descriptions.Item>
                <Descriptions.Item label="Bộ xác thực X-quang">{data.imageValidation?.enabled ? data.imageValidation.loaded ? 'Sẵn sàng' : 'Chưa sẵn sàng' : 'Đang tắt'}</Descriptions.Item>
                <Descriptions.Item label="Suy luận đang chạy">{data.inference?.active ?? 'Chưa ghi nhận'}</Descriptions.Item>
                <Descriptions.Item label="Yêu cầu đang chờ">{data.inference?.waiting ?? 'Chưa ghi nhận'}</Descriptions.Item>
                <Descriptions.Item label="Thời gian phản hồi">{data.latencyMs} ms</Descriptions.Item>
            </Descriptions>}
        </div>
    );
}
