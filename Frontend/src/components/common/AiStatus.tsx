import { deviceLabel } from '../../utils/display';
import { useGetServerHealthQuery } from '../../stores/baseApi';

export default function AiStatus() {
    const { data, error, isLoading } = useGetServerHealthQuery(undefined, { pollingInterval: 15000 });
    return <span role="status">{error ? 'Mất kết nối AI' : isLoading ? 'Đang kiểm tra AI…' :
        data?.status === 'healthy' ? `AI sẵn sàng · ${deviceLabel(data.device)}` : 'AI chưa sẵn sàng'}</span>;
}
