"""Replay saved labelled scores and optionally verify representative cases via the live API.

This audits serving decisions, not a new independent model accuracy evaluation.
It never tunes thresholds, changes weights, or alters the saved evaluation corpus.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Backend.ai_service.classification import DISEASE_LABELS, classification_summary


def binary_metrics(labels, predicted):
    tp = sum(label == 1 and prediction for label, prediction in zip(labels, predicted))
    tn = sum(label == 0 and not prediction for label, prediction in zip(labels, predicted))
    fp = sum(label == 0 and prediction for label, prediction in zip(labels, predicted))
    fn = sum(label == 1 and not prediction for label, prediction in zip(labels, predicted))
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'accuracy': (tp + tn) / len(labels),
            'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp)}


def audit(corpus, output, api=None):
    records = [json.loads(line) for line in (corpus / 'results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = []
    for record in records:
        if not record.get('classifier_success') or set(record['probabilities']) != set(DISEASE_LABELS):
            raise ValueError('Every frozen case must have all 15 valid classifier scores')
        assessment = classification_summary([
            {'disease': disease, 'probability': score} for disease, score in record['probabilities'].items()
        ], list(DISEASE_LABELS), visualized=record['native_findings'])
        rows.append((record, assessment))
    labels = [record['label'] for record, _ in rows]
    missing = [record['path'] for record, assessment in rows if 'Pneumonia' in assessment['withoutHeatmap']]
    summary = {
        'method': 'Replay existing frozen scores at the unchanged 0.5 threshold; not a new accuracy experiment',
        'sample_size': len(rows),
        'old_heatmap_list_pneumonia': binary_metrics(labels, ['Pneumonia' in record['native_findings'] for record, _ in rows]),
        'complete_classifier_pneumonia': binary_metrics(labels, [any(item['disease'] == 'Pneumonia' for item in assessment['findings']) for _, assessment in rows]),
        'pneumonia_previously_omitted': missing,
        'normal_label_conflicts_flagged': sum(assessment['conflictingNoFinding'] for _, assessment in rows),
        'cases_with_positive_labels_without_heatmaps': sum(bool(assessment['withoutHeatmap']) for _, assessment in rows),
    }
    if api:
        import requests
        selected = [(record, assessment) for record, assessment in rows if record['path'] in missing]
        selected += [next((row for row in rows if row[0]['label'] == 0 and row[1]['conflictingNoFinding']))]
        selected += [next((row for row in rows if row[0]['label'] == 0 and not row[1]['findings']))]
        checks = []
        for record, assessment in selected:
            with (corpus / 'images' / record['path']).open('rb') as image:
                response = requests.post(api.rstrip('/') + '/v3/predict', files={'file': (Path(record['path']).name, image, 'image/jpeg')}, timeout=120)
            response.raise_for_status()
            result = response.json()
            scores = {item['className']: item['probability'] for item in result['predictions']}
            difference = max(abs(score - scores[name]) for name, score in record['probabilities'].items())
            current = result['inferenceMeta']['classification']
            assert difference < 1e-5, f'Classifier scores changed: {difference}'
            assert [item['disease'] for item in current['findings']] == [item['disease'] for item in assessment['findings']]
            assert current['conflictingNoFinding'] == assessment['conflictingNoFinding']
            checks.append({'path': record['path'], 'http': response.status_code, 'max_probability_difference': difference,
                           'complete_findings': [item['disease'] for item in current['findings']],
                           'without_heatmap': current['withoutHeatmap'], 'conflict': current['conflictingNoFinding']})
            print('Verified ' + record['path'], flush=True)
        summary['live_api_checks'] = checks
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=ROOT / '.runtime/external-eval-2026-11-08')
    parser.add_argument('--output', type=Path, default=ROOT / '.runtime/classification-audit-2026-11-09.json')
    parser.add_argument('--api', help='Optional local AI origin, e.g. http://127.0.0.1:8000')
    args = parser.parse_args()
    audit(args.corpus, args.output, args.api)
