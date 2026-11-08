import { Request, Response } from 'express';
import { catchAsync } from '@common/utils/catchAsync';
import { ResearchService } from './research.service';
import { ApiError } from '@common/utils/ApiError';

const researchService = new ResearchService();
function period(req: Request): number {
    const weeks = req.query.weeks === undefined ? 8 : Number(req.query.weeks);
    if (![4, 8, 12].includes(weeks)) throw new ApiError(400, 'Chọn khoảng 4, 8 hoặc 12 tuần');
    return weeks;
}

export const getStats = catchAsync(async (req: Request, res: Response) => {
    const result = await researchService.getStats(period(req));
    res.status(200).json(result);
});

export const getTrends = catchAsync(async (req: Request, res: Response) => {
    const result = await researchService.getTrends(period(req));
    res.status(200).json(result);
});

export const getHeatmap = catchAsync(async (req: Request, res: Response) => {
    const result = await researchService.getHeatmap(period(req));
    res.status(200).json(result);
});
