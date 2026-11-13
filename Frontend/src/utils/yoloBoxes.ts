import type { YoloDetection } from '../types/hybrid';

/** YOLO's xyxy coordinates belong to the original image, never the cropped heatmap. */
export function yoloBoxes(detections: YoloDetection[], width: number, height: number) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return [];
  return detections.flatMap(detection => {
    const box = detection.bbox;
    if (!box || ![box.x1, box.y1, box.x2, box.y2, detection.confidence, detection.cascade_score].every(Number.isFinite)) return [];
    if (detection.confidence < 0 || detection.confidence > 1 || detection.cascade_score < 0 || detection.cascade_score > 1) return [];
    const x1 = Math.max(0, Math.min(width, box.x1));
    const y1 = Math.max(0, Math.min(height, box.y1));
    const x2 = Math.max(0, Math.min(width, box.x2));
    const y2 = Math.max(0, Math.min(height, box.y2));
    return x2 > x1 && y2 > y1 ? [{ ...detection, bbox: { x1, y1, x2, y2 } }] : [];
  });
}
