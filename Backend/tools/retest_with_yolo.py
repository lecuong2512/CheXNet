"""Repeat the frozen 200-image test after enabling YOLO, without tuning or resampling."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import evaluate_public_pneumonia as evaluation

ROOT = evaluation.ROOT
BASELINE = ROOT / '.runtime/external-eval-2026-11-08'
DEFAULT_OUTPUT = ROOT / '.runtime/external-eval-yolo-2026-11-09'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def prepare(output):
    if output.resolve() == BASELINE.resolve():
        raise ValueError('Do not overwrite the original evaluation')
    health = evaluation.get(evaluation.API + '/health').json()
    if not health.get('yolo_loaded'):
        raise RuntimeError('YOLO must be loaded for this repeat')
    hashes = {name: digest(ROOT / 'Trainedmodel' / name) for name in ['hybrid_model.pth', 'yolov11m.pt']}
    old_checkpoint = json.loads((BASELINE / 'checkpoint.json').read_text(encoding='utf-8'))
    if hashes['hybrid_model.pth'] != old_checkpoint['Hash'].lower():
        raise RuntimeError('Hybrid weights differ from the baseline')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        frozen = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
        if frozen['checkpoints'] != hashes:
            raise RuntimeError('Checkpoint changed during evaluation; do not resume')
        return
    protocol = json.loads((BASELINE / 'protocol.json').read_text(encoding='utf-8'))
    protocol.update({
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': 'Paired repeat on the same previously examined 200 test images after enabling YOLO',
        'baseline_protocol_sha256': digest(BASELINE / 'protocol.json'),
        'baseline_manifest_sha256': digest(BASELINE / 'manifest.json'),
        'checkpoints': hashes,
        'yolo_required': True,
        'no_tuning': True,
        'comparison_limits': ['Repeated test set, not new independent validation',
                              'Binary NORMAL/PNEUMONIA labels do not provide bounding-box ground truth',
                              'YOLO box counts measure serving coverage, not localization accuracy or mAP'],
    })
    protocol['secondary'] += ['Full classifier findings returned by the current web contract',
                              'YOLO box coverage, class counts and request latency only']
    manifest = json.loads((BASELINE / 'manifest.json').read_text(encoding='utf-8'))
    if len(manifest['images']) != 200:
        raise RuntimeError('Expected exactly the original frozen 200 cases')
    for image in manifest['images']:
        source = BASELINE / image['local_file']
        if digest(source) != image['sha256']:
            raise RuntimeError('Baseline image bytes changed: ' + image['path'])
        image['local_file'] = str(source.resolve())
    evaluation.write_json(output / 'protocol.json', protocol)
    manifest['protocol_sha256'] = digest(output / 'protocol.json')
    evaluation.write_json(output / 'manifest.json', manifest)
    print('Frozen the same 200 images, original Hybrid weights and 0.5 threshold; YOLO is enabled', flush=True)


def compare(output):
    current = read_rows(output / 'results.jsonl')
    baseline = {row['path']: row for row in read_rows(BASELINE / 'results.jsonl')}
    metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
    scored = [row for row in current if row.get('classifier_success')]
    paired = [(row, baseline[row['path']]) for row in scored if baseline[row['path']].get('classifier_success')]
    if len(current) != 200 or {row['path'] for row in current} != set(baseline):
        raise RuntimeError('Incomplete or different sample; do not publish a full comparison')
    if any(row['sha256'] != old['sha256'] or row['label'] != old['label'] for row, old in paired):
        raise RuntimeError('Image/label mismatch between paired runs')
    boxes = [box for row in current for box in row.get('yolo_detections', [])]
    change = max((abs(score - old['probabilities'][name]) for row, old in paired
                  for name, score in row['probabilities'].items()), default=None)
    counts = Counter(box.get('vindr_class', box['disease']) for box in boxes)
    result = {
        'baseline': str(BASELINE), 'repeat': str(output),
        'completed': len(current), 'paired_classifier_cases': len(paired),
        'max_classifier_probability_difference': change,
        'pneumonia_threshold_decision_changes': sum(row['predicted_pneumonia'] != old['predicted_pneumonia'] for row, old in paired),
        'images_with_yolo_boxes': sum(bool(row.get('yolo_detections')) for row in current),
        'images_without_yolo_boxes': sum(not row.get('yolo_detections') for row in current),
        'total_yolo_boxes': len(boxes), 'yolo_boxes_by_vindr_class': dict(counts),
        'boxes_are_not_localization_accuracy': True,
        'classifier_pneumonia': metrics['primary_classifier_pneumonia_threshold_0_5'],
        'current_web_pneumonia': metrics['web_complete_pneumonia_on_accepted_images'],
        'system': metrics['system'],
    }
    evaluation.write_json(output / 'comparison.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.report_only:
        prepare(output)
        evaluation.infer(output)
    evaluation.report(output)
    compare(output)
