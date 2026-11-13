import { Router } from 'express';
import axios from 'axios';
import { performance } from 'perf_hooks';
import { AI_SERVICE_URL } from '@config/environment';
import { catchAsync } from '@common/utils/catchAsync';
import { ApiError } from '@common/utils/ApiError';

const router = Router();

router.get('/ai-health', catchAsync(async (_req, res) => {
    const started = performance.now();
    try {
        const response = await axios.get(`${AI_SERVICE_URL}/health`, { timeout: 5000 });
        const health = response.data;
        if (typeof health.modelLoaded !== 'boolean' || typeof health.status !== 'string') {
            throw new Error('Invalid health response');
        }
        res.setHeader('Cache-Control', 'no-store');
        res.json({ success: true, message: 'Trạng thái dịch vụ AI', data: {
            ...health,
            checkedAt: new Date().toISOString(),
            latencyMs: Math.round(performance.now() - started),
        } });
    } catch {
        throw new ApiError(503, 'Không kết nối được dịch vụ AI. Vui lòng thử lại.');
    }
}));

export default router;
