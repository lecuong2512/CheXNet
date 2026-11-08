import { baseApi } from '../../../stores/baseApi';

export const researchApi = baseApi.injectEndpoints({
    endpoints: builder => ({
        getResearchStats: builder.query<any, number | void>({
            queryFn: async (weeks = 8, _api, _options, query) => {
                const [stats, trends, locations] = await Promise.all([
                    query('/research/stats?weeks=' + weeks),
                    query('/research/trends?weeks=' + weeks),
                    query('/research/heatmap?weeks=' + weeks),
                ]);
                if (stats.error) return { error: stats.error };
                if (trends.error) return { error: trends.error };
                if (locations.error) return { error: locations.error };
                return { data: { summary: (stats.data as any).data,
                    trends: (trends.data as any).data, locations: (locations.data as any).data } };
            },
            providesTags: ['Patients'],
        }),
    }),
});
export const { useGetResearchStatsQuery } = researchApi;
