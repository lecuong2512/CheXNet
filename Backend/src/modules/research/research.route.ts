import { Router } from 'express';
import { getStats, getTrends, getHeatmap } from './research.controller';

const router = Router();

router.get('/stats', getStats);
router.get('/trends', getTrends);
router.get('/heatmap', getHeatmap);

export default router;
