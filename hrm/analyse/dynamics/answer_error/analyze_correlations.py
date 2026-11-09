from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
"""Read-only analysis of frozen trajectories; no HRM/probe fit or intervention."""
from hrm.paths import repo_path
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import csv
import hashlib
import json
import numpy as np
import scipy.stats as stats
import torch
OUT = analysis_directory(__file__)
BASE = OUT.parent
ORIGINAL = Path(str(repo_path('checkpoint/dynamics/cl_agi_hrm_mini/artifacts/internal_critic')))
SOURCES = {'error': ORIGINAL, 'ce': BASE / 'probes/ce', 'mse': BASE / 'probes/mse'}
COHORTS = {'test': (384, 512), 'train': (0, 256), 'validation': (256, 384), 'all_descriptive': (0, 512)}
BOOTSTRAPS = 10000
SEED = 20261012

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write_new(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def correlation(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    a, b = (x - x.mean(), y - y.mean())
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator == 0 else float(np.dot(a, b) / denominator)

def moments(x, y, weights):
    """Exact correlations for whole-question resamples, including new time means."""
    puzzles, events = x.shape
    sx, sy = (weights @ x.sum(1), weights @ y.sum(1))
    sxx = weights @ np.square(x).sum(1)
    syy = weights @ np.square(y).sum(1)
    sxy = weights @ (x * y).sum(1)
    count = puzzles * events
    numerator = sxy - sx * sy / count
    denominator = np.sqrt(np.maximum(sxx - sx * sx / count, 0) * np.maximum(syy - sy * sy / count, 0))
    raw = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    tx, ty = (weights @ x, weights @ y)
    numerator = sxy - (tx * ty).sum(1) / puzzles
    denominator = np.sqrt(np.maximum(sxx - np.square(tx).sum(1) / puzzles, 0) * np.maximum(syy - np.square(ty).sum(1) / puzzles, 0))
    adjusted = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return (raw, adjusted)

def numerical_checks():
    rng = np.random.default_rng(718)
    x = rng.normal(size=(7, 13)) + np.linspace(-3, 3, 13)
    y = 0.4 * x + rng.normal(size=(7, 13))
    indices = rng.integers(0, 7, size=7)
    weights = np.bincount(indices, minlength=7)[None].astype(np.float64)
    raw, adjusted = moments(x, y, weights)
    xx, yy = (x[indices], y[indices])
    np.testing.assert_allclose(raw[0], stats.pearsonr(xx.ravel(), yy.ravel()).statistic, atol=1e-12)
    np.testing.assert_allclose(adjusted[0], correlation(xx - xx.mean(0), yy - yy.mean(0)), atol=1e-12)
    assert correlation(np.ones(7), np.arange(7)) is None
    np.testing.assert_allclose(correlation(x, 2 * x + 3), 1, atol=1e-12)
    return {'passed': True, 'checks': ['Question-count weighted Pearson equals explicit resampled SciPy Pearson.', 'Time-adjusted correlation equals explicit resampling and recomputation of time means.', 'Constant inputs return null; positive affine equivalents correlate at one.']}

def describe(x, y, weights):
    x, y = (np.asarray(x, np.float64), np.asarray(y, np.float64))
    assert x.shape == y.shape and x.ndim == 2 and np.isfinite(x).all() and np.isfinite(y).all()
    raw_parts, adjusted_parts = ([], [])
    for start in range(0, len(weights), 256):
        raw, adjusted = moments(x, y, weights[start:start + 256])
        raw_parts.append(raw)
        adjusted_parts.append(adjusted)
    raw, adjusted = (np.concatenate(raw_parts), np.concatenate(adjusted_parts))

    def interval(samples):
        return np.quantile(samples, [0.025, 0.975]).tolist() if np.isfinite(samples).all() else None
    return {'pearson': correlation(x, y), 'pearson_ci95': interval(raw), 'spearman': correlation(stats.rankdata(x.ravel()), stats.rankdata(y.ravel())), 'pearson_time_centered': correlation(x - x.mean(0), y - y.mean(0)), 'pearson_time_centered_ci95': interval(adjusted), 'bootstrap_undefined_raw': int((~np.isfinite(raw)).sum()), 'bootstrap_undefined_time_centered': int((~np.isfinite(adjusted)).sum())}

def main():
    torch.set_num_threads(4)
    if (OUT / 'results.json').exists():
        raise FileExistsError('Preserve completed results; use a separate analysis directory for changes.')
    manifest = read(BASE / 'collection/manifest.json')
    trajectory_path = BASE / 'collection/trajectory.pt'
    assert manifest['status'] == 'complete' and manifest['split'] == [256, 128, 128]
    assert sha(trajectory_path) == manifest['trajectory_sha256']
    source_files = [trajectory_path, BASE / 'collection/manifest.json']
    for target, directory in SOURCES.items():
        source_files.extend([directory / name for name in ('predictions.npz', 'metrics.csv', 'probes.pt')])
        if target != 'error':
            receipt = read(directory / 'completion.json')
            assert receipt['status'] == 'complete' and receipt['target'] == target
            for name, digest in receipt['files'].items():
                assert sha(directory / name) == digest
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'frozen before new correlation estimates', 'scope': 'Saved true-error correlations and fixed H-only readout analysis; no model/probe fitting or intervention.', 'trajectory': str(trajectory_path), 'primary_cohort': 'test', 'cohorts': COHORTS, 'targets': {'error': 'Original nine-class normalized row/column/box probability-mass excess E, not error_10class.', **manifest['target_definitions']}, 'readout': 'Uncommitted lm_head(H_level(H+L)) before each of 192 L updates per puzzle; all targets share it.', 'features': 'Original mean-pooled L and H; H-only probes use last512 features. Original separately fitted linear_H probes are retained.', 'primary_statistics': ['pooled Pearson', 'pooled Spearman'], 'time_adjustment': 'Subtract each event-index cross-puzzle mean from each raw variable, then compute pooled Pearson. Recompute means inside every bootstrap draw.', 'bootstrap': {'count': BOOTSTRAPS, 'seed': SEED, 'unit': 'whole puzzle with all192 states', 'interval': 'pointwise95%percentile for Pearson only; not simultaneous', 'spearman_ci': False, 'probe_uncertainty': 'conditional on the already fitted fixed probes; no probe refitting', 'undefined_draws': 'Count and report; do not drop silently. Affected interval is null.'}, 'time_index_correlations': 'Descriptive Pearson at each of192 time indices. Null if either variable is constant; no filtering of subjects.', 'H_limitations': 'H is unchanged within each six-L block; predicting a time-varying diagnostic loss from it also tests update frequency and pooled readout, not only encoded information.', 'all512': 'Descriptive sensitivity only; train/validation states are not new held-out subjects.', 'source_sha256': {str(p): sha(p) for p in source_files}, 'script_sha256': sha(__file__)}
    data = torch.load(trajectory_path, map_location='cpu', mmap=True, weights_only=True)
    errors = {k: data[k].numpy().astype(np.float64) for k in SOURCES}
    assert all((x.shape == (512, 192) for x in errors.values()))
    ids = data['ids'].numpy()
    predictions, old_metrics = ({}, {})
    for target, directory in SOURCES.items():
        with np.load(directory / 'predictions.npz') as p:
            np.testing.assert_array_equal(p['row_ids'], ids)
            np.testing.assert_array_equal(p['true_error'], errors[target])
            split = p['split']
            assert split.shape == (512,)
            assert (split[:256] == 0).all() and (split[256:384] == 1).all() and (split[384:] == 2).all()
            predictions[target] = p['linear_H'].astype(np.float64)
        with (directory / 'metrics.csv').open() as stream:
            old_metrics[target] = list(csv.DictReader(stream))
    h = data['features'][:, :, 512:]
    assert torch.equal(h.reshape(512, 32, 6, 512), h[:, ::6, None, :].expand(-1, -1, 6, -1))
    write_new(OUT / 'protocol.json', protocol)
    write_new(OUT / 'numerical_checks.json', numerical_checks())
    results = {'status': 'complete', 'protocol': protocol, 'cohorts': {}, 'H_block_repeat_bitwise_verified': True}
    for cohort, (start, stop) in COHORTS.items():
        n = stop - start
        weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=BOOTSTRAPS).astype(np.float64)
        item = {'puzzles': n, 'events_per_puzzle': 192, 'ids': ids[start:stop].tolist(), 'error_pairs': [], 'h_probes': []}
        for xkey, ykey in combinations(SOURCES, 2):
            x, y = (errors[xkey][start:stop], errors[ykey][start:stop])
            row = {'x': xkey, 'y': ykey, **describe(x, y, weights)}
            row['per_time_pearson'] = [correlation(x[:, t], y[:, t]) for t in range(192)]
            item['error_pairs'].append(row)
        for target in SOURCES:
            pred, truth = (predictions[target][start:stop], errors[target][start:stop])
            row = {'target': target, **describe(pred, truth, weights)}
            row['r2'] = float(1 - np.square(pred - truth).sum() / np.square(truth - truth.mean()).sum())
            if cohort != 'all_descriptive':
                previous = next((r for r in old_metrics[target] if r['method'] == 'linear_H' and r['split'] == cohort))
                np.testing.assert_allclose(row['pearson'], float(previous['pearson_error']), atol=1e-12)
                np.testing.assert_allclose(row['pearson_time_centered'], float(previous['pearson_error_within_time']), atol=1e-12)
                row['reference_L_pearson'] = float(next((r for r in old_metrics[target] if r['method'] == 'linear_L' and r['split'] == cohort))['pearson_error'])
            else:
                row['reference_L_pearson'] = None
            item['h_probes'].append(row)
        results['cohorts'][cohort] = item
        print(json.dumps({'completed_cohort': cohort, 'puzzles': n}), flush=True)
    write_new(OUT / 'results.json', results)
    write_new(OUT / 'completion.json', {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(), 'results_sha256': sha(OUT / 'results.json'), 'protocol_sha256': sha(OUT / 'protocol.json'), 'script_sha256': sha(__file__), 'model_forward': False, 'model_or_probe_training': False})
    print(json.dumps({'status': 'complete', 'results': str(OUT / 'results.json')}), flush=True)
if __name__ == '__main__':
    main()
