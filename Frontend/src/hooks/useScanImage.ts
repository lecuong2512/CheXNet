import { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import type { RootState } from '../stores/store';

const origin = import.meta.env.VITE_BACKEND_ORIGIN || 'http://127.0.0.1:3006';

export function useScanImage(source?: string) {
    const token = useSelector((state: RootState) => state.auth.accessToken);
    const protectedImage = !!source?.startsWith(origin + '/api/v1/scans/');
    const [loaded, setLoaded] = useState<{ source: string; token: string; url?: string; error?: boolean }>();
    useEffect(() => {
        if (!protectedImage || !source || !token) return;
        const controller = new AbortController();
        let objectUrl: string | undefined;
        void fetch(source, {
            headers: { Authorization: 'Bearer ' + token },
            signal: controller.signal,
        }).then(async response => {
            if (!response.ok) throw new Error('Không tải được ảnh');
            const blob = await response.blob();
            if (controller.signal.aborted) return;
            objectUrl = URL.createObjectURL(blob);
            setLoaded({ source, token, url: objectUrl });
        }).catch(() => {
            if (!controller.signal.aborted) setLoaded({ source, token, error: true });
        });
        return () => {
            controller.abort();
            if (objectUrl) URL.revokeObjectURL(objectUrl);
        };
    }, [source, token, protectedImage]);
    const current = loaded?.source === source && loaded?.token === token ? loaded : undefined;
    return {
        url: protectedImage ? current?.url : source,
        error: protectedImage && !!current?.error,
    };
}
