/** Nhãn hiển thị cho modelVersion từ AI service / MongoDB */
export const AI_MODEL_LABELS: Record<string, string> = {
    'densenet-121': 'CheXNet DenseNet-121',
    'convnextv2-large': 'ConvNeXtV2-Large',
    'convnextv2-large-v3': 'ConvNeXtV2-Large',
    'chexnet-unknown': 'CheXNet',
};

export function formatAiModelLabel(modelVersion?: string | null): string {
    if (!modelVersion) return 'CheXNet';
    return AI_MODEL_LABELS[modelVersion] ?? modelVersion;
}
