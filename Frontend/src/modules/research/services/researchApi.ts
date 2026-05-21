import { baseApi } from "../../../stores/baseApi";

export const researchApi = baseApi.injectEndpoints({
  endpoints: (builder) => ({
    getResearchStats: builder.query<any, void>({
      queryFn: async () => {
        // TODO: Kết nối API backend thực tế khi endpoint sẵn sàng
        return {
          data: {
            totalScans: 0,
            scanTrend: "0%",
            aiAccuracy: 0,
            responseTime: 0,
            responseTimeTrend: "0s",
            pathologyTrends: [],
            clinicalAlerts: [],
            densityGrid: [],
          },
        };
      },
    }),
  }),
  overrideExisting: false,
});

export const { useGetResearchStatsQuery } = researchApi;
