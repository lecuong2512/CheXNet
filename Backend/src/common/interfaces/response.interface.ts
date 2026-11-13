/**
 * Cấu trúc phản hồi API chuẩn cho toàn hệ thống CheXNet V3
 */
export interface IApiResponse<T> {
    success: boolean;
    message: string;
    data?: T;
}

export interface IPaginationMeta {
    totalItems: number;
    itemCount: number;
    itemsPerPage: number;
    totalPages: number;
    currentPage: number;
}

export interface IPaginatedResponse<T> {
    success: boolean;
    message: string;
    data: T[];
    meta: IPaginationMeta;
}
