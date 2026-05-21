/**
 * Bản đồ phân quyền tĩnh: ánh xạ API Route → chuỗi hành động kiểm tra quyền
 * Format: module.action
 */
export const permissionRoutes: Record<string, Record<string, { route: string; method: string }[]>> = {
    patients: {
        view: [
            { route: '/api/v1/patients', method: 'GET' },
            { route: '/api/v1/patients/:id', method: 'GET' },
        ],
        create: [{ route: '/api/v1/patients', method: 'POST' }],
        update: [{ route: '/api/v1/patients/:id', method: 'PATCH' }],
        delete: [{ route: '/api/v1/patients/:id', method: 'DELETE' }],
    },
    scans: {
        view: [
            { route: '/api/v1/scans/:patientId', method: 'GET' },
            { route: '/api/v1/scans/detail/:scanId', method: 'GET' },
        ],
        create: [{ route: '/api/v1/scans/:patientId/upload', method: 'POST' }],
        delete: [{ route: '/api/v1/scans/:scanId', method: 'DELETE' }],
    },
    diagnoses: {
        view: [
            { route: '/api/v1/diagnoses/by-patient/:patientId', method: 'GET' },
            { route: '/api/v1/diagnoses/:id', method: 'GET' },
        ],
        verify: [{ route: '/api/v1/diagnoses/:id/verify', method: 'PATCH' }],
    },
    users: {
        view: [{ route: '/api/v1/users', method: 'GET' }],
        create: [{ route: '/api/v1/users', method: 'POST' }],
        update: [{ route: '/api/v1/users/:id', method: 'PATCH' }],
        delete: [{ route: '/api/v1/users/:id', method: 'DELETE' }],
    },
    research: {
        view: [
            { route: '/api/v1/research/stats', method: 'GET' },
            { route: '/api/v1/research/trends', method: 'GET' },
            { route: '/api/v1/research/heatmap', method: 'GET' },
        ],
    },
};
