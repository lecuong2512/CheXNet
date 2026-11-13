import { deviceLabel } from '../../utils/display';
import { useGetServerHealthQuery } from '../../stores/baseApi';

export default function AiStatus() {
    const { data, error, isLoading } = useGetServerHealthQuery(undefined, { pollingInterval: 15000 });
    const yoloText = data?.yolo_loaded ? 'YOLO11m đã nạp' : 'YOLO chưa nạp';
    const modelText = data?.model_file ? ` · ${data.model_file}` : '';
    return <span role="status">{error ? 'Mất kết nối AI' : isLoading ? 'Đang kiểm tra AI…' :
        data?.status === 'healthy' ? `AI sẵn sàng · ${deviceLabel(data.device)} · ${yoloText}${modelText}` : 'AI chưa sẵn sàng'}</span>;
}
