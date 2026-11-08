export interface HybridPrediction {
    disease: string;
    probability: number;
    heatmap: string;
    lesion_location: string;
    lesion_area_percent: number;
    calibrated_confidence?: number;
    uncertainty?: number;
    attention_quality?: { concentration?: number; contrast?: number; sharpness?: number; quality_score?: number };
    bounding_boxes: unknown[];
}

export interface InferenceMeta {
    contractVersion?: number;
    device?: string;
    detectorStatus?: 'detections' | 'loaded-unconfirmed' | 'unavailable';
}

export interface HybridAnalysis {
    predictions: HybridPrediction[];
    all_probabilities: { disease: string; probability: number }[];
    yolo_detections: {
        disease: string;
        confidence: number;
        cascade_score: number;
        vindr_class?: string;
        bbox?: { x1: number; y1: number; x2: number; y2: number };
    }[];
    co_occurrences: { diseases: string[]; correlation: number; clinical_note: string }[];
    similar_cases: { filename: string; similarity: number; diseases: string[] }[];
    clinical_report: {
        summary: string;
        findings: { disease: string; finding_text: string; disease_vi?: string; severity_vi?: string;
            confidence_level?: string; uncertainty_note?: string }[];
        recommendations: string[];
        warnings: string[];
    };
}
