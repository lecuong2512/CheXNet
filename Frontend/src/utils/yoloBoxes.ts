import type { HybridBox, YoloDetection } from '../types/hybrid';

function clipBox(
  box: { x1: number; y1: number; x2: number; y2: number } | undefined,
  width: number,
  height: number,
) {
  if (!box || ![box.x1, box.y1, box.x2, box.y2].every(Number.isFinite)) return null;
  const x1 = Math.max(0, Math.min(width, box.x1));
  const y1 = Math.max(0, Math.min(height, box.y1));
  const x2 = Math.max(0, Math.min(width, box.x2));
  const y2 = Math.max(0, Math.min(height, box.y2));
  return x2 > x1 && y2 > y1 ? { x1, y1, x2, y2 } : null;
}

/** YOLO's xyxy coordinates belong to the original image, never the cropped heatmap. */
export function yoloBoxes(detections: YoloDetection[], width: number, height: number) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return [];
  return detections.flatMap(detection => {
    if (!Number.isFinite(detection.confidence) || !Number.isFinite(detection.cascade_score)) return [];
    if (detection.confidence < 0 || detection.confidence > 1 || detection.cascade_score < 0 || detection.cascade_score > 1) return [];
    const clipped = clipBox(detection.bbox, width, height);
    return clipped ? [{ ...detection, bbox: clipped }] : [];
  });
}

/** Hybrid fallback boxes are projected to original image coordinates for diseases YOLO did not detect. */
export function hybridFallbackBoxes(boxes: HybridBox[] = [], width: number, height: number) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return [];
  return boxes.flatMap(item => {
    if (!Number.isFinite(item.confidence) || !Number.isFinite(item.probability)) return [];
    if (item.confidence < 0 || item.confidence > 1 || item.probability < 0 || item.probability > 1) return [];
    const clipped = clipBox(item.bbox, width, height);
    return clipped ? [{ ...item, bbox: clipped }] : [];
  });
}

/** Chỉ giữ lại các khung thuộc dấu hiệu/bệnh đang chọn (tránh chồng chéo toàn bộ bệnh cùng lúc). */
export function filterBoxesByDisease<T extends { disease: string }>(
  items: T[],
  selectedDisease?: string,
): T[] {
  if (!selectedDisease) return items;
  const target = selectedDisease.trim().toLowerCase();
  return items.filter(item => item.disease.trim().toLowerCase() === target);
}
