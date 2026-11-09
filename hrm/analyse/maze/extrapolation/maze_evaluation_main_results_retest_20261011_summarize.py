"""Publish the nine frozen Maze retests only after CPU verification of every prediction.

No model loading, inference, training, network requests, or GPU use. A partial
campaign produces summary_status.json and exits 2; it never emits guessed scores.
"""
from __future__ import annotations
from hrm.paths import analysis_directory
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import numpy as np
ROOT = analysis_directory(__file__)
PROJECT = ROOT.parent.parent
pass
from hrm.analyse.maze.extrapolation.scoring import METRIC_NAME, METRIC_VERSION, score_predictions
DATA_ROOT = PROJECT / 'revision_packages/comment_2_10/maze_size_generalization/data_20261007'
DATASETS = {'maze_15': 'maze-15x15-hard-1k/test.csv', 'maze_20': 'maze-20x20-hard-1k/test.csv', 'maze_25': 'maze-25x25-hard-1k/test.csv', 'maze_30_original': 'maze-30x30-hard-1k/test.csv', 'maze_30_control': 'maze-30x30-reconstructed-control/test.csv', 'maze_32': 'maze-32x32-hard-1k/test.csv', 'maze_35': 'maze-35x35-hard-1k/test.csv', 'maze_40': 'maze-40x40-hard-1k/test.csv'}
VARIANTS = ('HRM_1D', 'RT_1D', 'D_2D')
FLAGS = ('shortest_path_correct', 'valid_path_correct', 'canonical_correct', 'format_correct')
VOCAB = {character: index for index, character in enumerate('~# SGo')}

class Pending(Exception):
    pass

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def atomic_write(path, text):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)

def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')

def load_data_sources(manifest):
    """Verify all eight immutable test sources, including reference label identity."""
    bundle_path = DATA_ROOT / 'manifest.json'
    require(sha256(bundle_path) == manifest['d_data_manifest_sha256'], 'Frozen D data manifest changed')
    bundle = json.loads(bundle_path.read_text())
    entries = {entry['path']: entry for entry in bundle['datasets']}
    sources, arrays = ({}, {})
    manifest_sources = manifest.get('datasets', {})
    require(isinstance(manifest_sources, dict), 'manifest.datasets must be a name->source mapping')
    for name, relative in DATASETS.items():
        declared = manifest_sources.get(name, {})
        path = Path(declared.get('path', DATA_ROOT / relative)).resolve()
        expected_hash = declared.get('sha256', entries[relative]['sha256'])
        require(path.is_file(), f'Missing frozen dataset: {path}')
        digest = sha256(path)
        require(digest == expected_hash, f'Dataset hash mismatch: {name}')
        require(digest == entries[relative]['sha256'], f'Dataset differs from the original eight-test bundle: {name}')
        with path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 1000, f'{name}: expected all 1000 test examples')
        questions = np.asarray([[0] + [VOCAB[c] for c in row['question']] for row in rows], dtype=np.int32)
        targets = np.asarray([[0] + [VOCAB[c] for c in row['answer']] for row in rows], dtype=np.int32)
        require(len({row.tobytes() for row in questions}) == 1000, f'Duplicate question in frozen source: {name}')
        require(score_predictions(questions, targets, targets)['shortest_path_correct'].all(), f'A frozen reference answer is not a legal shortest path: {name}')
        arrays[name] = (questions, targets)
        sources[name] = {'path': str(path), 'sha256': digest, 'examples': 1000, 'seq_len': int(questions.shape[1])}
    original30 = Path(manifest['original30_test_path']).resolve()
    require(sha256(original30) == manifest['original30_test_sha256'] == sources['maze_30_original']['sha256'], 'Original30 baseline and D tests differ')
    return (sources, arrays)

def scalar_text(values, key):
    require(key in values, f'Missing prediction metadata {key}')
    return str(np.asarray(values[key]).item())

def load_prediction_parts(paths, require_indices):
    data = {key: [] for key in ('inputs', 'targets', 'predictions')}
    stored_flags = {key: [] for key in FLAGS}
    indices, ranks, hashes = ([], [], [])
    for path in paths:
        with np.load(path, allow_pickle=False) as values:
            require(scalar_text(values, 'metric_version') == METRIC_VERSION, f'Wrong metric version in {path}')
            for key in data:
                require(key in values, f'Missing {key}: {path}')
                value = values[key]
                require(value.ndim == 2 and np.issubdtype(value.dtype, np.integer), f'{path}: {key} must be integer [batch,tokens]')
                data[key].append(value.copy())
            count = len(data['inputs'][-1])
            require(count > 0, f'Empty prediction shard: {path}')
            for key in FLAGS:
                if key in values:
                    require(values[key].shape == (count,), f'Invalid stored flag shape: {path}/{key}')
                    stored_flags[key].append(values[key].copy())
            if require_indices:
                require('indices' in values and 'rank' in values and ('world_size' in values), f'D shard lacks original indices/rank/world size: {path}')
                idx = values['indices']
                rank, world = (int(values['rank']), int(values['world_size']))
                require(idx.shape == (count,) and np.issubdtype(idx.dtype, np.integer), f'Invalid D indices: {path}')
                require(world > 0 and 0 <= rank < world, f'Invalid D rank/world size: {path}')
                require(np.array_equal(idx, np.arange(rank, 1000, world)), f'D shard indices do not match frozen rank-strided coverage: {path}')
                indices.append(idx.copy())
                ranks.append((rank, world))
        hashes.append({'path': str(path.resolve()), 'sha256': sha256(path)})
    if require_indices:
        require(len({world for _, world in ranks}) == 1, 'D shards have different world sizes')
        world = ranks[0][1]
        require(sorted((rank for rank, _ in ranks)) == list(range(world)), 'D shards contain missing or duplicate rank attempts; refusing to choose one silently')
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in data.items()}
    flag_arrays = {}
    for key, parts in stored_flags.items():
        if parts:
            require(len(parts) == len(paths), f'Only some shards contain {key}')
            flag_arrays[key] = np.concatenate(parts)
    return (arrays, flag_arrays, None if not indices else np.concatenate(indices), hashes)

def assert_metric(value, expected, description):
    require(type(value) in (int, float) and math.isfinite(value) and math.isclose(value, expected, rel_tol=0, abs_tol=1e-09), f'{description}: saved {value!r}, CPU replay {expected!r}')

def verify_dataset(job, name, metrics, paths, expected_arrays):
    arrays, saved_flags, indices, hashes = load_prediction_parts(paths, job['variant'] == 'D_2D')
    inputs, targets, predictions = (arrays[key] for key in ('inputs', 'targets', 'predictions'))
    require(len(inputs) == 1000, f"{job['id']}/{name}: not all 1000 test rows")
    require(inputs.shape == targets.shape == predictions.shape, f"Mismatched prediction shapes: {job['id']}/{name}")
    expected_inputs, expected_targets = expected_arrays
    require(inputs.shape == expected_inputs.shape, f"Incorrect test shape: {job['id']}/{name}")
    lookup = {tuple(row): index for index, row in enumerate(expected_inputs)}
    unique_inputs = {tuple(row) for row in inputs}
    require(len(unique_inputs) == 1000, f"Duplicate questions in {job['id']}/{name}")
    require(unique_inputs == set(lookup), f"Questions differ from frozen test: {job['id']}/{name}")
    row_indices = np.asarray([lookup[tuple(row)] for row in inputs], dtype=np.int64)
    if indices is not None:
        require(np.array_equal(np.sort(indices), np.arange(1000)), f"D index coverage incomplete: {job['id']}/{name}")
        require(np.array_equal(row_indices, indices), f"D indices mismatch original questions: {job['id']}/{name}")
    require(np.array_equal(targets, expected_targets[row_indices]), f"Reference labels differ: {job['id']}/{name}")
    scores = score_predictions(inputs, predictions, targets)
    for key, values in saved_flags.items():
        require(np.array_equal(values, scores[key]), f"Saved {key} differs from CPU scorer: {job['id']}/{name}")
    shortest = scores['shortest_path_correct']
    valid = scores['valid_path_correct']
    canonical = scores['canonical_correct']
    format_correct = scores['format_correct']
    board_same = np.all(predictions[:, 1:] == targets[:, 1:], axis=1)
    alternatives = shortest & ~board_same
    bos_only = shortest & board_same & ~canonical
    require(not np.any(canonical & ~shortest), 'Exact valid references must imply shortest-path correctness')
    require(np.count_nonzero(canonical) + np.count_nonzero(alternatives) + np.count_nonzero(bos_only) == np.count_nonzero(shortest), 'Shortest-path score partition failed')
    require(metrics.get('metric_version') == METRIC_VERSION, f"Wrong result metric version: {job['id']}/{name}")
    assert_metric(metrics.get('total'), 1000, 'total')
    assert_metric(metrics.get('correct'), int(shortest.sum()), 'correct')
    assert_metric(metrics.get(METRIC_NAME), float(shortest.mean()), METRIC_NAME)
    assert_metric(metrics.get('exact_match'), float(canonical.mean()), 'canonical exact-match')
    for key, expected in (('accuracy', shortest.mean()), ('valid_path_accuracy', valid.mean()), ('valid_path_rate', valid.mean()), ('format_accuracy', format_correct.mean()), ('format_correct_rate', format_correct.mean())):
        if key in metrics:
            assert_metric(metrics[key], float(expected), key)
    old_exact = job['old_exact_match'][name]
    require(type(old_exact) in (int, float) and 0 <= old_exact <= 1, f"Invalid old exact score: {job['id']}/{name}")
    old_correct = round(1000 * old_exact)
    require(math.isclose(1000 * old_exact, old_correct, abs_tol=1e-08), 'Old score does not represent 1000 rows')
    return {'dataset': name, 'seed': job['seed'], 'job_id': job['id'], 'variant': job['variant'], 'metric_name': METRIC_NAME, 'metric_version': METRIC_VERSION, 'total': 1000, 'shortest_path_correct': int(shortest.sum()), 'shortest_path_accuracy': float(shortest.mean()), 'valid_path_correct': int(valid.sum()), 'valid_nonshortest_count': int((valid & ~shortest).sum()), 'invalid_path_count': int((~valid).sum()), 'canonical_correct': int(canonical.sum()), 'canonical_accuracy': float(canonical.mean()), 'old_canonical_correct': old_correct, 'canonical_delta_vs_old_count': int(canonical.sum()) - old_correct, 'canonical_replay_matches_old': int(canonical.sum()) == old_correct, 'alternative_shortest_count': int(alternatives.sum()), 'reference_route_wrong_bos_count': int(bos_only.sum()), 'bos_only_reference_mismatch': int(bos_only.sum()), 'format_correct_count': int(format_correct.sum()), 'coverage': {'unique_questions': 1000, 'matches_frozen_source': True, 'D_indices_complete': indices is not None}, 'prediction_files': hashes}

def verify_job(root, job, data_arrays):
    directory = root / 'results' / job['id']
    result_path = directory / 'metrics.json'
    if not result_path.is_file():
        raise Pending(f"{job['id']}: metrics.json pending")
    result = json.loads(result_path.read_text())
    checkpoint = Path(job['checkpoint']).resolve()
    require(checkpoint.is_file(), f'Checkpoint missing: {checkpoint}')
    require(sha256(checkpoint) == job['checkpoint_sha256'], f"Checkpoint changed: {job['id']}")
    require(Path(result['checkpoint']).resolve() == checkpoint, f"Wrong result checkpoint: {job['id']}")
    expected_names = set(DATASETS) if job['variant'] == 'D_2D' else {'maze_30_original'}
    require(set(job['old_exact_match']) == expected_names, f"Old result inventory incomplete: {job['id']}")
    if job['variant'] == 'D_2D':
        require(set(result['metrics']) == expected_names, f"D result datasets incomplete: {job['id']}")
        require(result.get('epoch') == 19 and result.get('weights') == 'EMA' and (result.get('seed') == job['seed']), f"Wrong D checkpoint epoch/weights/seed: {job['id']}")
        metrics = result['metrics']
    else:
        require(result.get('T') == 16 and result.get('split') == 'test', f"Wrong baseline T/split: {job['id']}")
        metrics = {'maze_30_original': result}
    rows = []
    for name in DATASETS:
        if name not in metrics:
            continue
        if job['variant'] == 'D_2D':
            paths = sorted((directory / 'metrics_predictions' / name).glob('rank_*.npz'))
        else:
            paths = [directory / 'predictions.npz']
        if not paths or any((not path.is_file() for path in paths)):
            raise Pending(f"{job['id']}/{name}: raw predictions pending")
        rows.append(verify_dataset(job, name, metrics[name], paths, data_arrays[name]))
    return {'id': job['id'], 'variant': job['variant'], 'seed': job['seed'], 'checkpoint': str(checkpoint), 'checkpoint_sha256': job['checkpoint_sha256'], 'metrics_file': str(result_path.resolve()), 'metrics_sha256': sha256(result_path), 'datasets': rows}

def aggregate(rows):
    require(len(rows) == 3 and sorted((row['seed'] for row in rows)) == [1, 2, 3], 'Expected exactly three independent seeds')
    output = {'seeds': [1, 2, 3], 'questions_per_seed': 1000, 'metric_name': METRIC_NAME, 'metric_version': METRIC_VERSION}
    for metric in ('shortest_path_accuracy', 'canonical_accuracy', 'valid_nonshortest_count', 'alternative_shortest_count', 'reference_route_wrong_bos_count', 'invalid_path_count'):
        values = [row[metric] for row in sorted(rows, key=lambda row: row['seed'])]
        output[metric] = {'per_seed': values, 'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
    output['canonical_delta_vs_old_count'] = [row['canonical_delta_vs_old_count'] for row in sorted(rows, key=lambda row: row['seed'])]
    return output

def render_report(result):
    lines = ['Maze main-result reevaluation: any valid shortest path', f'Metric version: {METRIC_VERSION}', 'Fixed epoch-20 EMA, T16, seeds 1/2/3; all 1,000 instances per dataset and seed.', 'Training is unchanged; only scoring changes. Earlier LR1e-4 D and cloud timing experiments are excluded.', 'SD denotes sample standard deviation across three seeds.', 'Reference exact match in the main table uses the same predictions and can be compared directly with shortest-path accuracy; historical exact match is compared only in the final table.', '', 'Original 30×30 main comparison (%)', 'Model          Shortest-path mean±SD    seed1 / seed2 / seed3    Reference exact-match mean±SD']
    for variant in VARIANTS:
        row = result['main_30_comparison'][variant]
        shortest, canonical = (row[METRIC_NAME], row['canonical_accuracy'])
        seeds = ' / '.join((f'{100 * value:.1f}' for value in shortest['per_seed']))
        lines.append(f"{variant:<12} {100 * shortest['mean']:.2f}±{100 * shortest['sample_std']:.2f}    {seeds}    {100 * canonical['mean']:.2f}±{100 * canonical['sample_std']:.2f}")
    lines += ['', 'D evaluation at eight sizes (%)', 'Dataset             Shortest-path mean±SD    seed1 / seed2 / seed3']
    for name in DATASETS:
        shortest = result['D_eight_datasets'][name][METRIC_NAME]
        seeds = ' / '.join((f'{100 * value:.1f}' for value in shortest['per_seed']))
        lines.append(f"{name:<20} {100 * shortest['mean']:.2f}±{100 * shortest['sample_std']:.2f}    {seeds}")
    lines += ['', 'Per-seed scoring breakdown (counts; 1,000 instances per row)', 'run / dataset: Shortest path | Current reference exact match | Historical reference exact match | Alternative shortest path | Reference path with BOS error only | Valid non-shortest | Invalid | Current minus historical exact match']
    for job in result['verified_jobs']:
        for row in job['datasets']:
            lines.append(f"{job['id']} / {row['dataset']}: {row['shortest_path_correct']} | {row['canonical_correct']} | {row['old_canonical_correct']} | {row['alternative_shortest_count']} | {row['reference_route_wrong_bos_count']} | {row['valid_nonshortest_count']} | {row['invalid_path_count']} | {row['canonical_delta_vs_old_count']:+d}")
    lines += ['', 'Alternative shortest paths must differ from the reference board path. BOS-only errors are counted separately and are not alternative paths.', 'Valid non-shortest and invalid predictions are counted separately. BFS only verifies shortest distance; it does not modify or repair model outputs.', 'Current versus historical exact-match differences are reported separately; numerical differences are not all attributed to the scoring change.', 'All NPZ outputs are rescored on CPU, checking completeness and uniqueness of 1,000 instances, original CSV puzzles/labels and checkpoint SHA256. D also checks complete original row-index coverage.', 'Original results and checkpoints are not overwritten. See final_results.json for complete per-instance paths and hashes.']
    return '\n'.join(lines) + '\n'

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'summary_status.json'
    try:
        manifest_path = root / 'manifest.json'
        if not manifest_path.is_file():
            raise Pending('Frozen manifest.json has not been published')
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest['jobs']
        require(len(jobs) == 9 and len({job['id'] for job in jobs}) == 9, 'Expected nine unique jobs')
        require({(job['variant'], job['seed']) for job in jobs} == {(variant, seed) for variant in VARIANTS for seed in (1, 2, 3)}, 'Wrong main-result run matrix')
        require(manifest.get('metric_version') == METRIC_VERSION, 'Manifest metric version mismatch')
        require(manifest.get('cycles') == 16 and manifest.get('checkpoint_epoch_zero_based') == 19 and (manifest.get('weights') == 'EMA'), 'Manifest frozen inference protocol changed')
        source_root = Path(manifest['project']).resolve()
        for relative, expected_hash in manifest['source_sha256'].items():
            require(sha256(source_root / relative) == expected_hash, f'Frozen inference source changed: {relative}')
        scorer_hash = sha256(ROOT.parent / 'scoring.py')
        if 'scorer_sha256' in manifest:
            require(manifest['scorer_sha256'] == scorer_hash, 'Scorer changed since freeze')
        sources, data_arrays = load_data_sources(manifest)
        verified, pending = ([], [])
        for job in jobs:
            try:
                verified.append(verify_job(root, job, data_arrays))
            except Pending as error:
                pending.append(str(error))
        if pending:
            write_json(status_path, {'status': 'pending', 'verified_jobs': [job['id'] for job in verified], 'pending': pending, 'final_results_published': False})
            print(json.dumps({'status': 'pending', 'verified': len(verified), 'total': 9, 'pending': pending}, ensure_ascii=False))
            return 2
        rows = [row for job in verified for row in job['datasets']]
        main_table = {variant: aggregate([row for row in rows if row['variant'] == variant and row['dataset'] == 'maze_30_original']) for variant in VARIANTS}
        d_table = {name: aggregate([row for row in rows if row['variant'] == 'D_2D' and row['dataset'] == name]) for name in DATASETS}
        result = {'status': 'completed', 'metric_name': METRIC_NAME, 'metric_version': METRIC_VERSION, 'manifest_sha256': sha256(manifest_path), 'scorer_sha256': scorer_hash, 'd_data_manifest_sha256': manifest['d_data_manifest_sha256'], 'source_sha256': manifest['source_sha256'], 'verification_device': 'CPU', 'T': 16, 'checkpoint_epoch_zero_based': 19, 'weights': 'EMA', 'datasets': sources, 'verified_jobs': verified, 'main_30_comparison': main_table, 'D_eight_datasets': d_table}
        output = root / 'final_results.json'
        if output.exists():
            require(json.loads(output.read_text()) == result, 'Existing final_results.json differs; refusing to replace a published result')
        else:
            write_json(output, result)
        report = render_report(result)
        report_path = root / 'report.txt'
        if report_path.exists():
            require(report_path.read_text() == report, 'Existing report differs; refusing to replace it')
        else:
            atomic_write(report_path, report)
        write_json(status_path, {'status': 'completed', 'verified_jobs': 9, 'verified_dataset_evaluations': len(rows), 'final_results_published': True, 'final_results': str(output)})
        print(report)
        return 0
    except Pending as error:
        write_json(status_path, {'status': 'pending', 'reason': str(error), 'final_results_published': False})
        print(f'PENDING: {error}')
        return 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        write_json(status_path, {'status': 'verification_failed', 'error': f'{type(error).__name__}: {error}', 'final_results_published': False})
        print(f'VERIFICATION FAILED: {error}', file=sys.stderr)
        return 1
if __name__ == '__main__':
    raise SystemExit(main())
