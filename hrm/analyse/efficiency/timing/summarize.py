#!/usr/bin/env python3
"""Recompute the six recorded timing settings using repository-local data."""
from hrm.paths import analysis_directory
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

ROOT = analysis_directory(__file__)
JOBS = [f'{task}_{model}' for task in ('sudoku', 'maze') for model in ('hrm', 'rt', 'trm')]
ADDITIONAL_WARMUP_CALLS = 1


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def recompute():
    provenance = read_json(ROOT / 'provenance.json')
    for item in provenance['copied_files']:
        assert sha256(ROOT / item['path']) == item['sha256'], item['path']
    assert sha256(ROOT / 'per_call.csv') == provenance['per_call_csv_sha256']
    groups = {job: [] for job in JOBS}
    identities = {job: [] for job in JOBS}
    with (ROOT / 'per_call.csv').open() as stream:
        for row in csv.DictReader(stream):
            job = row['job_id']
            assert int(row['sample_index']) == len(groups[job])
            value = float(row['latency_seconds'])
            assert math.isfinite(value) and value > 0
            groups[job].append(value)
            identities[job].append((row['source_row_index'], row['input_sha256']))
    original = {r['job_id']: r for r in read_json(ROOT / 'sources/archived_summary.json')['rows']}
    rows = []
    for job in JOBS:
        folder = ROOT / 'sources' / job
        train = read_json(folder / 'training_summary.json')
        infer = read_json(folder / 'inference/inference_summary.json')
        status = read_json(folder / 'status.json')
        assert train['status'] == infer['status'] == status['status'] == 'completed'
        assert train['checkpoint_sha256'] == infer['checkpoint_sha256'] == original[job]['checkpoint_sha256']
        ranks = [read_json(folder / p) for p in train['rank_record_sha256']]
        assert len(ranks) == 8
        assert all(r['status'] == 'completed' and r['world_size'] == 8 for r in ranks)
        assert all(r['actual_steps'] == train['optimizer_steps'] for r in ranks)
        measured_train = max(r['phase_totals_seconds']['train'] for r in ranks)
        assert math.isclose(measured_train, train['training_seconds'], rel_tol=1e-12)
        assert train['training_seconds'] == original[job]['training_seconds']
        values = groups[job]
        assert len(values) == infer['measured_examples'] == original[job]['inference_calls']
        assert infer['batch_size'] == 1 and infer['cycles_per_data'] == 16
        all_mean, all_sd = statistics.mean(values), statistics.stdev(values)
        for actual, expected in [(all_mean, infer['batch_latency']['mean_seconds']),
                                 (all_sd, infer['batch_latency']['sample_sd_seconds']),
                                 (all_mean, original[job]['inference_mean_seconds']),
                                 (all_sd, original[job]['inference_sample_sd_seconds'])]:
            assert math.isclose(actual, expected, rel_tol=1e-12), job
        # Apply the same first-call warmup rule to every setting; retain raw data.
        measured = values[ADDITIONAL_WARMUP_CALLS:]
        mean, sd = statistics.mean(measured), statistics.stdev(measured)
        rows.append({
            'job_id': job, 'task': job.split('_')[0], 'model': job.split('_')[1], 'seed': 1,
            'training_seconds': measured_train, 'training_minutes': measured_train / 60,
            'training_process_seconds': train['process_wall_seconds'],
            'training_process_minutes': train['process_wall_seconds'] / 60,
            'training_gpu_hours': train['training_gpu_hours'],
            'recorded_inference_calls': len(values),
            'additional_warmup_calls': ADDITIONAL_WARMUP_CALLS,
            'inference_calls': len(measured), 'inference_mean_ms': mean * 1000,
            'inference_sample_sd_ms': sd * 1000,
            'inference_median_ms': statistics.median(measured) * 1000,
            'inference_p95_ms': percentile(measured, .95) * 1000,
            'additional_warmup_seconds': values[0],
            'calls_over_one_second': sum(t > 1 for t in measured),
            'maximum_inference_ms': max(measured) * 1000,
            'archived_all_calls_mean_ms': all_mean * 1000,
            'archived_all_calls_sample_sd_ms': all_sd * 1000,
            'checkpoint_sha256': train['checkpoint_sha256'],
        })
    for task in ('sudoku', 'maze'):
        assert identities[f'{task}_hrm'] == identities[f'{task}_rt'] == identities[f'{task}_trm']
    assert sum(len(v) for v in groups.values()) == provenance['total_measured_calls'] == 63000
    assert sum(r['inference_calls'] for r in rows) == 62994
    assert original['sat_hrm']['status'] == 'failed'
    assert read_json(ROOT / 'sources/sat_hrm/status.json')['status'] == 'failed'
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify without writing files.')
    args = parser.parse_args()
    rows = recompute()
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    expected = output.getvalue()
    path = ROOT / 'settings.csv'
    if args.check:
        assert path.read_text() == expected, 'settings.csv differs from recomputed statistics'
    else:
        path.write_text(expected)
    print('PASS: 63,000 raw calls verified; six first-call warmups excluded; '
          '62,994 calls summarized; 48 training-rank records and archived hashes verified.')


if __name__ == '__main__':
    main()
