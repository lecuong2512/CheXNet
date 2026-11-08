"""Reproducible external binary evaluation through the running, unchanged AI API.

Downloads public test images only; does not train, tune or modify the model.
Run prepare before run to freeze sampling and the evaluation protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / '.runtime/external-eval-2026-11-08'
REPOSITORY = 'probml/chest_xray_kaggle'
COMMIT = '551b993d5dcaec27a1f1bbbcbca2ff4968a28e07'
SEED = 20261108
PER_CLASS = 100
THRESHOLD = 0.5
API = 'http://127.0.0.1:8000'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def get(url):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def patient_proxy(filename, label):
    # These are filename-derived groups, not independently verified patient IDs.
    pattern = r'(person\d+)' if label else r'((?:NORMAL2-)?IM-\d+)'
    match = re.match(pattern, filename, re.IGNORECASE)
    if not match:
        raise ValueError(f'Cannot group filename: {filename}')
    return ('pneumonia:' if label else 'normal:') + match.group(1).lower()


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / 'manifest.json'
    if manifest_file.exists():
        print('Using frozen manifest', flush=True)
        return
    if (ROOT / 'Results/optimal_thresholds.npy').exists():
        raise RuntimeError('Threshold artifact exists: explicitly review protocol before evaluating')
    tree = get(f'https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1').json()
    if tree.get('truncated'):
        raise RuntimeError('Source file tree is truncated')
    write_json(output / 'source-tree.json', tree)
    selected = []
    populations = {}
    rng = random.Random(SEED)
    for folder, label in [('NORMAL', 0), ('PNEUMONIA', 1)]:
        entries = [e for e in tree['tree'] if e['type'] == 'blob'
                   and e['path'].startswith(f'test/{folder}/')
                   and e['path'].lower().endswith(('.jpeg', '.jpg', '.png'))]
        groups = {}
        for entry in entries:
            proxy = patient_proxy(Path(entry['path']).name, label)
            groups.setdefault(proxy, []).append(entry)
        populations[folder] = {'images': len(entries), 'filename_patient_groups': len(groups)}
        keys = sorted(groups)
        rng.shuffle(keys)
        if len(keys) < PER_CLASS:
            raise RuntimeError(f'Insufficient independent filename groups: {folder}')
        for proxy in keys[:PER_CLASS]:
            entry = rng.choice(sorted(groups[proxy], key=lambda e: e['path']))
            selected.append({'path': entry['path'], 'git_blob_sha1': entry['sha'],
                             'source_bytes': entry['size'], 'label': label,
                             'class': folder, 'patient_proxy': proxy,
                             'url': f'https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{entry["path"]}'})
    rng.shuffle(selected)
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': {'repository': REPOSITORY, 'commit': COMMIT,
                   'original_dataset': 'https://data.mendeley.com/datasets/rscbjbr9sj/2',
                   'license': 'CC BY 4.0 (images)', 'authors': 'Daniel Kermany; Kang Zhang; Michael Goldbaum'},
        'sampling': {'split': 'test', 'per_class': PER_CLASS, 'seed': SEED,
                     'unit': 'one random image per filename-derived patient group',
                     'population': populations},
        'primary': 'Pneumonia probability >= 0.5; binary normal vs pneumonia only',
        'threshold': THRESHOLD,
        'secondary': ['Pneumonia appears in native top-six displayed findings',
                      'No Finding >= 0.5', 'any native pathology on NORMAL images',
                      'No Finding >= 0.5 together with native pathology'],
        'failure_policy': 'Keep every sampled image. Record all HTTP/validation errors. On v3 HTTP 422 only, call legacy /predict to measure classifier separately; never count rejection as a correct web diagnosis.',
        'no_tuning': True,
        'limits': ['Pediatric external dataset; not full 14-pathology evaluation',
                   'Checkpoint training provenance and patient independence cannot be guaranteed from filenames',
                   'Confidence intervals do not include source-label or distribution-shift uncertainty'],
    }
    write_json(output / 'protocol.json', protocol)
    protocol_hash = hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest()
    manifest = {'protocol_sha256': protocol_hash, 'images': selected}
    write_json(manifest_file, manifest)
    print(json.dumps({'frozen_sample': len(selected), 'population': populations,
                      'protocol_sha256': protocol_hash}), flush=True)


def download(output):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    def one(item):
        local = output / 'images' / item['path']
        local.parent.mkdir(parents=True, exist_ok=True)
        data = local.read_bytes() if local.exists() else get(item['url']).content
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != item['git_blob_sha1']:
            raise RuntimeError('Source bytes mismatch: ' + item['path'])
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            dimensions = list(image.size)
        local.write_bytes(data)
        return {**item, 'sha256': hashlib.sha256(data).hexdigest(), 'dimensions': dimensions,
                'local_file': local.relative_to(output).as_posix()}
    items = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(one, item) for item in manifest['images']]
        for future in as_completed(futures):
            items.append(future.result())
            if len(items) % 25 == 0:
                print(f'Downloaded and verified {len(items)}/{len(futures)}', flush=True)
    by_path = {item['path']: item for item in items}
    manifest['images'] = [by_path[item['path']] for item in manifest['images']]
    if len({item['sha256'] for item in manifest['images']}) != len(items):
        raise RuntimeError('Duplicate image bytes in frozen sample; review sampling before inference')
    local_training_names = set()
    for split in ('train', 'val'):
        with (ROOT / f'Dataset/{split}_list.csv').open(encoding='utf-8-sig', newline='') as stream:
            local_training_names.update(Path(row['Image Index']).name for row in csv.DictReader(stream))
    manifest['training_filename_overlap'] = [item['path'] for item in items if Path(item['path']).name in local_training_names]
    write_json(output / 'manifest.json', manifest)
    print(f'Download complete. Filename overlap with local train/val: {len(manifest["training_filename_overlap"])}', flush=True)


def infer(output):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    protocol_hash = hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest()
    if protocol_hash != manifest['protocol_sha256']:
        raise RuntimeError('Protocol changed after sample selection')
    if (ROOT / 'Results/optimal_thresholds.npy').exists():
        raise RuntimeError('Threshold artifact changed; stop evaluation')
    health = get(API + '/health').json()
    if health['status'] != 'healthy':
        raise RuntimeError('AI service is not ready')
    write_json(output / 'health-before.json', health)
    rows_file = output / 'results.jsonl'
    completed = {}
    if rows_file.exists():
        for line in rows_file.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            completed[row['path']] = row
    session = requests.Session()
    started = time.perf_counter()
    for number, item in enumerate(manifest['images'], 1):
        if item['path'] in completed:
            continue
        local = output / item['local_file']
        data = local.read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError('Local image changed: ' + item['path'])
        row = {'path': item['path'], 'label': item['label'], 'class': item['class'],
               'patient_proxy': item['patient_proxy'], 'sha256': item['sha256']}
        tick = time.perf_counter()
        try:
            response = session.post(API + '/v3/predict', files={'file': (local.name, data, 'image/jpeg')}, timeout=(10, 120))
            row['web_http'] = response.status_code
            body = response.json()
            row['web_success'] = response.ok and body.get('success') is True
            row['web_wall_seconds'] = round(time.perf_counter() - tick, 4)
            row['classifier_via_legacy_after_rejection'] = False
            if not row['web_success']:
                row['web_error'] = body
            if row['web_success']:
                analysis = body['hybridAnalysis']
                probabilities = {p['className']: p['probability'] for p in body['predictions']}
                row['processing_seconds'] = body.get('processingTime')
            elif response.status_code == 422:
                row['classifier_via_legacy_after_rejection'] = True
                fallback = session.post(API + '/predict', files={'file': (local.name, data, 'image/jpeg')}, timeout=(10, 120))
                row['legacy_http'] = fallback.status_code
                analysis = fallback.json()
                if not fallback.ok or not analysis.get('success'):
                    raise RuntimeError('Legacy classifier failed: ' + str(analysis)[:500])
                probabilities = {p['disease']: p['probability'] for p in analysis['all_probabilities']}
            else:
                raise RuntimeError('API inference did not succeed')
            if len(probabilities) != 15 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
                raise RuntimeError('Invalid 15-class probability response')
            findings = analysis.get('predictions', [])
            row.update({'classifier_success': True, 'probabilities': probabilities,
                        'predicted_pneumonia': probabilities['Pneumonia'] >= THRESHOLD,
                        'native_findings': [p['disease'] for p in findings],
                        'clinical_summary': analysis.get('clinical_report', {}).get('summary'),
                        'native_findings_detail': [{k: v for k, v in p.items() if k != 'heatmap'} for p in findings],
                        'model_version': body.get('modelVersion'),
                        'inference_meta': body.get('inferenceMeta')})
        except Exception as exc:
            row.update({'classifier_success': False, 'error': str(exc)})
        row['total_wall_seconds'] = round(time.perf_counter() - tick, 4)
        with rows_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
        completed[item['path']] = row
        write_json(output / 'progress.json', {'completed': len(completed), 'planned': len(manifest['images']),
                   'web_success': sum(r.get('web_success', False) for r in completed.values()),
                   'classifier_success': sum(r.get('classifier_success', False) for r in completed.values()),
                   'elapsed_seconds_this_run': round(time.perf_counter() - started, 1)})
        if number % 10 == 0 or number == len(manifest['images']):
            # Progress intentionally omits accuracy to prevent sample-dependent stopping.
            print(f'Inference {number}/{len(manifest["images"])}; web OK {sum(r.get("web_success", False) for r in completed.values())}; elapsed {time.perf_counter()-started:.1f}s', flush=True)
    write_json(output / 'health-after.json', get(API + '/health').json())


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0, center-radius), min(1, center+radius)]


def classification(rows, decision, probability=None):
    from sklearn.metrics import roc_auc_score
    tp = sum(r['label'] == 1 and decision(r) for r in rows)
    tn = sum(r['label'] == 0 and not decision(r) for r in rows)
    fp = sum(r['label'] == 0 and decision(r) for r in rows)
    fn = sum(r['label'] == 1 and not decision(r) for r in rows)
    n = len(rows)
    divide = lambda a, b: a/b if b else None
    sensitivity = divide(tp, tp+fn)
    specificity = divide(tn, tn+fp)
    return {'n': n, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'accuracy': divide(tp+tn, n), 'accuracy_ci95_wilson': wilson(tp+tn, n),
            'sensitivity': sensitivity, 'sensitivity_ci95_wilson': wilson(tp, tp+fn),
            'specificity': specificity, 'specificity_ci95_wilson': wilson(tn, tn+fp),
            'precision': divide(tp, tp+fp), 'f1': divide(2*tp, 2*tp+fp+fn),
            'balanced_accuracy': (sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            'roc_auc': float(roc_auc_score([r['label'] for r in rows], [probability(r) for r in rows])) if probability and len({r['label'] for r in rows}) == 2 else None}


def report(output):
    import numpy as np
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (output/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != len(manifest['images']) or len({r['path'] for r in rows}) != len(rows):
        raise RuntimeError('Report requires the complete, unique frozen sample')
    classified = [r for r in rows if r.get('classifier_success')]
    accepted = [r for r in classified if r.get('web_success')]
    normals = [r for r in classified if not r['label']]
    primary = classification(classified, lambda r:r['predicted_pneumonia'], lambda r:r['probabilities']['Pneumonia'])
    displayed = classification(accepted, lambda r:'Pneumonia' in r['native_findings'])
    no_finding_rule = classification(classified, lambda r:r['probabilities']['No Finding'] < THRESHOLD,
                                     lambda r:1-r['probabilities']['No Finding'])
    native_abnormal = classification(accepted, lambda r:bool(r['native_findings']))
    times = [r['web_wall_seconds'] for r in rows if r.get('web_success')]
    result = {
        'sample': {'planned': len(manifest['images']), 'completed': len(rows),
                   'normal': sum(r['label']==0 for r in rows), 'pneumonia':sum(r['label']==1 for r in rows),
                   'unique_filename_patient_groups': len({r['patient_proxy'] for r in rows}),
                   'unique_image_sha256': len({r['sha256'] for r in rows}),
                   'train_val_filename_overlap':manifest['training_filename_overlap']},
        'primary_classifier_pneumonia_threshold_0_5':primary,
        'web_native_pneumonia_on_accepted_images':displayed,
        'secondary_no_finding_rule_for_abnormality':no_finding_rule,
        'secondary_any_native_finding_for_abnormality':native_abnormal,
        'normal_images': {'n':len(normals),
                          'any_native_pathology':sum(bool(r['native_findings']) for r in normals),
                          'no_finding_ge_0_5':sum(r['probabilities']['No Finding']>=THRESHOLD for r in normals),
                          'contradiction_no_finding_and_pathology':sum(r['probabilities']['No Finding']>=THRESHOLD and bool(r['native_findings']) for r in normals)},
        'system': {'web_accepted':len(accepted), 'classifier_scored':len(classified),
                   'legacy_fallback_count':sum(r.get('classifier_via_legacy_after_rejection',False) for r in rows),
                   'failed_or_rejected':[{k:r.get(k) for k in ('path','class','web_http','web_error','error')} for r in rows if not r.get('web_success')],
                   'correct_web_pneumonia_decisions_over_all_attempts':sum(r['label']==int('Pneumonia' in r['native_findings']) for r in accepted),
                   'all_attempts_denominator':len(rows),
                   'wall_seconds_median':statistics.median(times) if times else None,
                   'wall_seconds_p95':float(np.percentile(times,95)) if times else None,
                   'wall_seconds_sum':sum(r['total_wall_seconds'] for r in rows)},
        'protocol_sha256':manifest['protocol_sha256'],
        'source_commit':COMMIT,
    }
    write_json(output/'metrics.json',result)
    with (output/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        names=['path','class','label','web_http','web_success','classifier_success','pneumonia_probability','no_finding_probability','predicted_pneumonia','native_findings','web_wall_seconds']
        writer=csv.DictWriter(stream,fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row.get(k) for k in names if k not in ('pneumonia_probability','no_finding_probability','native_findings')},
                             'pneumonia_probability':row.get('probabilities',{}).get('Pneumonia'),
                             'no_finding_probability':row.get('probabilities',{}).get('No Finding'),
                             'native_findings':';'.join(row.get('native_findings',[]))})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','download','run','report'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    {'prepare':prepare,'download':download,'run':infer,'report':report}[args.phase](args.output.resolve())
