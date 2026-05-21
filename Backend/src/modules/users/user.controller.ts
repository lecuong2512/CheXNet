import { Request, Response } from 'express';
import { catchAsync } from '@common/utils/catchAsync';
import { UserService } from './user.service';

const userService = new UserService();

export const getUsers = catchAsync(async (req: Request, res: Response) => {
    const filter = {
        page: req.query.page ? parseInt(String(req.query.page), 10) : 1,
        limit: req.query.limit ? parseInt(String(req.query.limit), 10) : 10,
        keyword: typeof req.query.keyword === 'string' ? req.query.keyword : undefined,
        role: typeof req.query.role === 'string' ? (req.query.role as any) : undefined,
        department: typeof req.query.department === 'string' ? req.query.department : undefined,
    };
    const result = await userService.getByFilter(filter);
    res.status(200).json(result);
});

export const createUser = catchAsync(async (req: Request, res: Response) => {
    const result = await userService.create(req.body);
    res.status(201).json(result);
});

export const updateUser = catchAsync(async (req: Request, res: Response) => {
    const result = await userService.update(String(req.params.id), req.body);
    res.status(200).json(result);
});

export const deleteUser = catchAsync(async (req: Request, res: Response) => {
    const result = await userService.remove(String(req.params.id));
    res.status(200).json(result);
});
