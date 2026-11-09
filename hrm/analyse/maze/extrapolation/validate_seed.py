"""CPU-only acceptance of one complete HRM-D seed and all 12 frozen Maze tests.

Only a successful full audit publishes runs/D_2D_seedN/verification.json.
No inference, GPU initialization, output repair, or historical metric migration.
"""
from __future__ import annotations
from hrm.paths import analysis_directory
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
EXECUTION = analysis_directory(__file__)
PROJECT = EXECUTION.parent
WORKSPACE = PROJECT.parent
RUNTIME = PROJECT / 'runtime'
pass
from hrm.analyse.maze.extrapolation.experiment_config import resolved_config, run_identity
from . import summarize as audit
HELPER = Path(audit.__file__)
require, sha256, write_json = (audit.require, audit.sha256, audit.write_json)
METRIC_NAME, METRIC_VERSION = (audit.METRIC_NAME, audit.METRIC_VERSION)
SMALL_NAMES, LARGE_NAMES = (tuple(audit.parent.DATASETS), tuple(audit.LARGE_NAMES))
ALL_NAMES = SMALL_NAMES + LARGE_NAMES

def load_sources():
    manifest = json.loads((EXECUTION / 'large_manifest.json').read_text())
    require(manifest['cycles'] == 16 and manifest['weights'] == 'EMA' and (manifest['checkpoint_epoch_zero_based'] == 19) and (manifest['metric_version'] == METRIC_VERSION), 'Wrong inference protocol')
    for path, expected in manifest['provenance_sha256'].items():
        require(sha256(path) == expected, f'Frozen evaluation dependency changed: {path}')
    require(sha256(WORKSPACE / 'maze_evaluation/scoring.py') == manifest['scorer_sha256'], 'Frozen shortest-path scorer changed')
    small_manifest = dict(manifest, datasets=manifest['small_datasets'])
    small_sources, small_arrays = audit.parent.load_data_sources(small_manifest)
    large_sources, large_arrays = audit.load_large_sources(manifest)
    return (manifest, {**small_sources, **large_sources}, {**small_arrays, **large_arrays})

def verify_prediction_set(job, name, metrics, directory, expected):
    """Require all four saved flags and exactly one shard per eight ranks."""
    paths = sorted(Path(directory).glob('rank_*.npz'))
    require(len(paths) == 8, f"{job['id']}/{name}: need precisely eight committed shards")
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            require(str(shard['metric_name'].item()) == METRIC_NAME, f'Wrong metric: {path}')
            for flag in audit.parent.FLAGS:
                require(flag in shard and shard[flag].dtype == np.bool_, f'Missing or non-boolean saved flag {flag}: {path}')
    row = audit.verify_large_dataset(job, name, metrics, paths, expected)
    require(set(metrics) >= {'accuracy', 'valid_path_accuracy', 'format_accuracy', 'canonical_correct'}, f"Incomplete committed metrics: {job['id']}/{name}")
    audit.parent.assert_metric(metrics['canonical_correct'], row['canonical_correct'], 'canonical count')
    row.update(correct=row['shortest_path_correct'], valid_path_accuracy=row['valid_path_correct'] / 1000, exact_match=row['canonical_accuracy'])
    return row

def verify_training(run_dir, seed, config, identity):
    metadata = json.loads((run_dir / 'run_metadata.json').read_text())
    final = json.loads((run_dir / 'final_metrics.json').read_text())
    require(metadata['identity'] == final['identity'] == identity, 'Strict current runtime/training/final identity mismatch')
    require(json.loads((run_dir / 'resolved_config.json').read_text()) == config, 'Saved training configuration differs from the frozen runtime')
    require(all((final.get(k) == v for k, v in {'experiment': 'D', 'seed': seed, 'epoch': 19, 'step': 10240, 'weights': 'EMA'}.items())), 'Training did not complete the final 20-epoch EMA protocol')
    require(metadata.get('experiment') == 'D' and metadata.get('seed') == seed, 'Training metadata belongs to a different experiment/seed')
    data = metadata['training_data']
    require(data['training_examples'] == 1000 and data['training_examples_by_size'] == {'15': 250, '20': 250, '25': 250, '30': 250}, 'Training selection changed')
    require(data['selection_sha256'] == identity['data_selection_sha256'] and data['batches_per_epoch'] == 32 and (data['repeat'] == 25) and (data['global_batch_size'] == 768), 'Training sampling/batch protocol changed')
    require(set(final['metrics']) == set(SMALL_NAMES), 'Expected all eight final training-time test sets')
    log_path = run_dir / 'metrics.jsonl'
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    train = [row for row in records if row['kind'] == 'train']
    evaluation = [row for row in records if row['kind'] == 'eval']
    require(len(records) == len(train) + len(evaluation), 'Unknown training log record kind')
    require(len(train) == 640 and len(evaluation) == 27, 'Incomplete or duplicate train/eval log records')
    for index, row in enumerate(train):
        require((row['epoch'], row['batch'], row['step']) == (index // 32, index % 32, (index + 1) * 16), 'Training log does not cover exactly 20 epochs / 10240 updates')
        require(row['seed'] == seed and row['experiment'] == 'D', 'Wrong training log identity')
        require(math.isfinite(row['loss']) and math.isfinite(row['lr']), 'Non-finite training statistics')
    for epoch in range(20):
        require((run_dir / 'checkpoints' / f'epoch_{epoch}.pt').is_file(), 'Missing epoch EMA checkpoint')
        rows = [row for row in evaluation if row['epoch'] == epoch]
        names = SMALL_NAMES if epoch == 19 else ('maze_30_original',)
        require(len(rows) == len(names) and {row['dataset'] for row in rows} == set(names), f'Missing/duplicate epoch{epoch} evaluation')
        for row in rows:
            require(row['weights'] == 'EMA' and row['step'] == (epoch + 1) * 512 and (row['seed'] == seed) and (row['experiment'] == 'D') and (row['metric_version'] == METRIC_VERSION) and (row['total'] == 1000), 'Wrong online evaluation identity, weights, metric, or coverage')
            if epoch == 19:
                require(all((row.get(k) == v for k, v in final['metrics'][row['dataset']].items())), 'Final metrics differ from the committed epoch19 log')
    import torch
    state = torch.load(run_dir / 'checkpoints/latest_train.pt', map_location='cpu', weights_only=True)
    require(state['identity'] == identity and state['epoch'] == 19 and (state['step'] == 10240) and (state['total_steps'] == 10240) and (state['world_size'] == 8) and (state['weights'] == 'non_ema_training') and (state['format_version'] == 1), 'Latest committed training state does not match the completed run')
    require(state['final_metrics'] == final['metrics'], 'Training boundary/final evaluation mismatch')
    del state
    return (final, dict(epochs=20, optimizer_steps=10240, train_records=640, evaluation_records=27))

def verify_seed(seed):
    manifest, sources, arrays = load_sources()
    require(seed in manifest['seeds'], 'Only the new seeds4--10 can publish verification receipts')
    run_dir = PROJECT / 'runs' / f'D_2D_seed{seed}'
    config, identity = (resolved_config('D', seed), run_identity('D', seed=seed))
    final, training = verify_training(run_dir, seed, config, identity)
    checkpoint = run_dir / 'checkpoints/epoch_19.pt'
    job = dict(id=f'D_2D_seed{seed}', seed=seed, variant='D_2D', run_dir=str(run_dir), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    directory = run_dir / 'large_evaluation'
    large_path, meta_path = (directory / 'metrics.json', directory / 'run_metadata.json')
    large, meta = (json.loads(large_path.read_text()), json.loads(meta_path.read_text()))
    require(all((large.get(k) == v for k, v in {'experiment': 'D', 'seed': seed, 'epoch': 19, 'weights': 'EMA', 'T': 16, 'checkpoint': str(checkpoint), 'checkpoint_sha256': job['checkpoint_sha256'], 'metric_name': METRIC_NAME, 'metric_version': METRIC_VERSION}.items())), 'Large evaluation checkpoint or inference protocol differs')
    require(set(large['metrics']) == set(LARGE_NAMES), 'Large evaluation inventory incomplete')
    extended = dict(config['architecture'])
    extended['allowed_sizes'] = sorted(set(extended['allowed_sizes']) | {45, 50, 60, 80})
    require(meta['training_identity'] == meta['evaluation_identity'] == identity and meta['original_architecture'] == config['architecture'] and (meta['architecture'] == extended) and (meta['world_size'] == 8) and (meta['seed'] == seed) and (meta['rope_variant'] == 'original') and (meta['checkpoint'] == str(checkpoint)) and (meta['checkpoint_sha256'] == job['checkpoint_sha256']) and (meta['manifest_sha256'] == sha256(EXECUTION / 'large_manifest.json')) and (meta['evaluator_sha256'] == sha256(EXECUTION / 'evaluate_large.py')), 'Large evaluation metadata does not bind the unchanged model and original 2D coordinates')
    rows = {}
    for name in ALL_NAMES:
        if name in SMALL_NAMES:
            metrics = final['metrics'][name]
            prediction_dir = run_dir / 'evaluations' / METRIC_VERSION / 'epoch_19' / name
        else:
            metrics = large['metrics'][name]
            prediction_dir = directory / 'metrics_predictions' / name
            frozen = manifest['datasets'][name]
            require(metrics['local_batch_size'] == frozen['local_batch_size'] and metrics['world_size'] == 8 and (metrics['cycles'] == 16), 'Large inference batching or T changed')
            require(json.loads((directory / f'{name}_metrics.json').read_text()) == metrics, 'Per-dataset large metrics differ from the committed combined result')
        rows[name] = verify_prediction_set(job, name, metrics, prediction_dir, arrays[name])
        print(f"VERIFIED seed={seed} dataset={name} shortest={rows[name]['correct']}/1000", flush=True)
    result_files = [run_dir / name for name in ('run_metadata.json', 'resolved_config.json', 'metrics.jsonl', 'final_metrics.json')]
    result_files += [large_path, meta_path, run_dir / 'checkpoints/latest_train.pt']
    result_files += [directory / f'{name}_metrics.json' for name in LARGE_NAMES]
    source_files = dict(manifest['provenance_sha256'])
    source_files.update({str(RUNTIME / name): digest for name, digest in identity['source_sha256'].items()})
    source_files.update({str(path): sha256(path) for path in (RUNTIME / 'protocol.json', RUNTIME / 'data_selection.json', EXECUTION / 'evaluate_large.py', EXECUTION / 'large_manifest.json', Path(__file__))})
    return dict(passed=True, status='completed', seed=seed, experiment='D', variant='D_2D', epoch=19, weights='EMA', T=16, metric_name=METRIC_NAME, metric_version=METRIC_VERSION, checkpoint=str(checkpoint), checkpoint_sha256=job['checkpoint_sha256'], identity=identity, training=training, metrics=rows, sources=sources, manifest_sha256=sha256(EXECUTION / 'large_manifest.json'), validator_sha256=sha256(__file__), source_files_sha256=source_files, result_files_sha256={str(p): sha256(p) for p in result_files}, verified_utc=datetime.now(timezone.utc).isoformat(), verification_device='CPU')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=range(4, 11))
    args = parser.parse_args()
    result = verify_seed(args.seed)
    path = PROJECT / 'runs' / f'D_2D_seed{args.seed}' / 'verification.json'
    if path.exists():
        previous = json.loads(path.read_text())
        require({k: v for k, v in previous.items() if k != 'verified_utc'} == {k: v for k, v in result.items() if k != 'verified_utc'}, 'Refusing to replace a different already accepted result')
    else:
        write_json(path, result)
    print(json.dumps({'passed': True, 'seed': args.seed, 'datasets': 12, 'predictions': 12000, 'verification': str(path)}), flush=True)
if __name__ == '__main__':
    main()
