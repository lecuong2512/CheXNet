import { Request, Response } from 'express';
import { catchAsync } from '@common/utils/catchAsync';
import { ResearchService } from './research.service';

const researchService = new ResearchService();

export const getStats = catchAsync(async (_req: Request, res: Response) => {
    const result = await researchService.getStats();
    res.status(200).json(result);
});

export const getTrends = catchAsync(async (_req: Request, res: Response) => {
    const result = await researchService.getTrends();
    res.status(200).json(result);
});

export const getHeatmap = catchAsync(async (_req: Request, res: Response) => {
    const result = await researchService.getHeatmap();
    res.status(200).json(result);
});
