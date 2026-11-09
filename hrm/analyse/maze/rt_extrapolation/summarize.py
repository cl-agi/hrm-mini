"""CPU audit of RT-D seeds 4--10 and combination with verified seeds 1--3.

Reuse the frozen HRM-D audit helpers; never load or execute a neural model.
Only original 2D RoPE is included. Publish after all seven new seeds complete.
"""
from __future__ import annotations
from hrm.paths import analysis_directory
import argparse
import csv
import importlib.util
import io
import json
from pathlib import Path
import statistics
import sys
import numpy as np
ROOT = analysis_directory(__file__)
from hrm.analyse.maze.extrapolation import summarize as large
HELPER = Path(large.__file__)
parent = large.parent
require, sha256, write_json = (parent.require, parent.sha256, parent.write_json)
Pending, score_predictions = (parent.Pending, parent.score_predictions)
METRIC_NAME, METRIC_VERSION = (parent.METRIC_NAME, parent.METRIC_VERSION)
pass
from hrm.analyse.maze.rt_extrapolation.experiment_config import resolved_config, run_identity

def read_manifest(require_freeze=True):
    manifest = json.loads((ROOT / 'eval_manifest.json').read_text())
    require(manifest['seeds'] == list(range(4, 11)) and manifest['variants'] == ['original'], 'Wrong frozen seed/variant matrix')
    require(manifest['cycles'] == 16 and manifest['weights'] == 'EMA' and (manifest['checkpoint_epoch_zero_based'] == 19) and (manifest['world_size'] == 8) and (manifest['metric_version'] == METRIC_VERSION), 'Evaluation settings changed')
    require(sha256(ROOT / 'runtime/maze_metrics.py') == manifest['scorer_sha256'], 'Scorer changed')
    for path, digest in manifest['frozen_files_sha256'].items():
        require(sha256(path) == digest, f'Frozen dependency changed: {path}')
    if require_freeze:
        freeze = json.loads((ROOT / 'launch_freeze.json').read_text())
        for path, digest in freeze['sha256'].items():
            require(sha256(path) == digest, f'Campaign source changed: {path}')
    return manifest

def load_sources(manifest):
    require(len(manifest['datasets']) == 12, 'Expected twelve full test sets')
    sources, arrays = ({}, {})
    for name, item in manifest['datasets'].items():
        path = Path(item['path'])
        require(path.resolve() == (Path(item['data_root']) / item['relative_path']).resolve(), f'Dataset path differs from bundle: {name}')
        require(sha256(path) == item['sha256'], f'Dataset changed: {name}')
        bundle = json.loads((Path(item['data_root']) / 'manifest.json').read_text())
        entry, = [row for row in bundle['datasets'] if row['path'] == item['relative_path']]
        require(entry['sha256'] == item['sha256'] and entry['size'] == item['size'], f'Bundle disagrees with frozen test: {name}')
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == item['count'] == 1000 and len({row['question'] for row in rows}) == 1000, f'Expected 1000 distinct source questions: {name}')
        require(all((len(row['question']) == len(row['answer']) == item['size'] ** 2 for row in rows)), f'Wrong source board dimensions: {name}')
        x = np.asarray([[0] + [parent.VOCAB[c] for c in row['question']] for row in rows], dtype=np.int32)
        y = np.asarray([[0] + [parent.VOCAB[c] for c in row['answer']] for row in rows], dtype=np.int32)
        require(score_predictions(x, y, y, size=item['size'])['shortest_path_correct'].all(), f'Invalid reference shortest route: {name}')
        arrays[name] = (x, y)
        sources[name] = dict(item, unique_questions=1000, seq_len=item['size'] ** 2 + 1)
    return (sources, arrays)

def load_baselines(manifest):
    baselines = {}
    for variant, entry in manifest['HRM_D_baselines'].items():
        require(sha256(entry['path']) == entry['sha256'], f'HRM-D {variant} report changed')
        result = json.loads(Path(entry['path']).read_text())
        require(result['status'] == 'completed' and result['metric_version'] == METRIC_VERSION and (result['scorer_sha256'] == manifest['scorer_sha256']) and (result['T'] == 16) and (result['weights'] == 'EMA') and (result['checkpoint_epoch_zero_based'] == 19), 'HRM-D comparison uses a different evaluation protocol')
        table = result[entry['aggregate_key']]
        require(set(table) == set(manifest['datasets']), 'HRM-D comparison lacks a test set')
        require(all((row['seeds'] == [1, 2, 3] and row['questions_per_seed'] == 1000 for row in table.values())), 'HRM-D comparison uses a different seed/count inventory')
        if variant == 'bandwise':
            for name, source in result['datasets'].items():
                require(source['sha256'] == manifest['datasets'][name]['sha256'], f'HRM-D/RT-D comparison datasets differ: {name}')
        baselines[variant] = table
    return baselines

def verify_job(seed, variant, manifest, expected_arrays):
    run = ROOT / 'runs' / f'RT_D_seed{seed}'
    output = ROOT / 'evaluations' / variant / f'seed{seed}'
    completion_path = output / 'completion.json'
    if not completion_path.exists():
        raise Pending(f'{variant}/seed{seed}: final evaluation completion pending')
    completion = json.loads(completion_path.read_text())
    require(completion['status'] == 'completed' and completion['seed'] == seed and (completion['rope_variant'] == variant), 'Wrong evaluation completion')
    actual_files = {str(path.relative_to(output)) for path in output.rglob('*') if path.is_file() and path != completion_path}
    require(set(completion['files']) == actual_files, 'Missing or unexpected evaluation artifacts')
    for relative, digest in completion['files'].items():
        require(sha256(output / relative) == digest, f'Evaluation artifact changed: {output / relative}')
    config = resolved_config('D', seed)
    identity = run_identity('D', seed=seed)
    require(json.loads((run / 'resolved_config.json').read_text()) == config, 'Training config changed')
    require(json.loads((run / 'run_metadata.json').read_text())['identity'] == identity, 'Training identity changed; no migration allowed')
    metadata_path, result_path = (output / 'run_metadata.json', output / 'metrics.json')
    metadata = json.loads(metadata_path.read_text())
    result = json.loads(result_path.read_text())
    manifest_sha = sha256(ROOT / 'eval_manifest.json')
    require(completion['manifest_sha256'] == metadata['manifest_sha256'] == result['manifest_sha256'] == manifest_sha, 'Evaluation manifest identity changed')
    require(metadata['launch_freeze_sha256'] == sha256(ROOT / 'launch_freeze.json'), 'Evaluation launch freeze differs')
    require(metadata['training_identity'] == metadata['evaluation_identity'] == identity, 'Strict training/evaluation identity mismatch')
    require(metadata['original_architecture'] == config['architecture'], 'Original RT architecture differs')
    architecture = dict(config['architecture'])
    architecture['allowed_sizes'] = sorted(set(architecture['allowed_sizes']) | {45, 50, 60, 80})
    require(metadata['architecture'] == architecture and metadata['world_size'] == 8, 'Unauthorized evaluation architecture/layout change')
    for name, digest in metadata['evaluation_source_sha256'].items():
        require(sha256(ROOT / name) == digest, f'Evaluation source changed: {name}')
    require(set(metadata['evaluation_source_sha256']) == {'evaluate_sizes.py', 'rope_scaling.py'}, 'Evaluation source inventory differs')
    require(result['experiment'] == 'RT_D' and result['architecture_name'] == 'RT' and (result['seed'] == seed) and (result['epoch'] == 19) and (result['weights'] == 'EMA') and (result['T'] == 16) and (result['rope_variant'] == metadata['rope_variant'] == variant), 'Wrong final checkpoint or variant')
    checkpoint = run / 'checkpoints/epoch_19.pt'
    digest = sha256(checkpoint)
    require(Path(result['checkpoint']).resolve() == Path(metadata['checkpoint']).resolve() == checkpoint.resolve() and result['checkpoint_sha256'] == metadata['checkpoint_sha256'] == completion['checkpoint_sha256'] == digest, 'Checkpoint identity mismatch')
    require(set(result['metrics']) == set(expected_arrays), 'Incomplete twelve-dataset evaluation')
    job = dict(id=f'RT_D_seed{seed}', variant='RT_D', seed=seed)
    rows = []
    for name, item in manifest['datasets'].items():
        metric = result['metrics'][name]
        require(metric['cycles'] == 16 and metric['world_size'] == 8 and (metric['local_batch_size'] == item['local_batch_size']), 'Evaluation batch or T differs')
        require(json.loads((output / f'{name}_metrics.json').read_text()) == metric, 'Dataset result differs from final result')
        paths = sorted((output / 'metrics_predictions' / name).glob('rank_*.npz'))
        require(len(paths) == 8, f'Missing or duplicate ranks: {variant}/seed{seed}/{name}')
        row = large.verify_large_dataset(job, name, metric, paths, expected_arrays[name])
        row['variant'], row['rope_variant'] = ('RT_D', variant)
        for path in paths:
            with np.load(path, allow_pickle=False) as shard:
                require(str(shard['rope_variant'].item()) == variant and int(shard['cycles']) == 16, 'Prediction shard variant/budget mismatch')
                scores = score_predictions(shard['inputs'], shard['predictions'], shard['targets'], size=item['size'])
                for key in (*parent.FLAGS, 'shortest_distance', 'predicted_steps'):
                    require(key in shard and np.array_equal(shard[key], scores[key]), f'Raw prediction audit mismatch: {path}/{key}')
        rows.append(row)
    return dict(id=job['id'], seed=seed, variant='RT_D', rope_variant=variant, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest, metrics_file=str(result_path), metrics_sha256=sha256(result_path), completion_file=str(completion_path), completion_sha256=sha256(completion_path), datasets=rows)

def stats(values):
    return dict(per_seed=values, mean=statistics.mean(values), sample_std=statistics.stdev(values))

def aggregate_rows(rows, seeds):
    rows = sorted(rows, key=lambda row: row['seed'])
    require([row['seed'] for row in rows] == seeds, 'Missing or duplicate training seeds')
    require(len({row['dataset'] for row in rows}) == 1, 'Mixed datasets in one aggregate')
    for row in rows:
        require(row['total'] == 1000 and row['metric_name'] == METRIC_NAME and (row['metric_version'] == METRIC_VERSION) and (row['variant'] == 'RT_D') and (row['rope_variant'] == 'original'), 'Wrong row protocol or RoPE variant')
        require(row['coverage'] == dict(unique_questions=1000, matches_frozen_source=True, D_indices_complete=True), 'Unverified question coverage')
        counts = ('shortest_path_correct', 'valid_path_correct', 'canonical_correct', 'valid_nonshortest_count', 'invalid_path_count', 'alternative_shortest_count', 'reference_route_wrong_bos_count', 'format_correct_count')
        require(all((isinstance(row[key], int) and 0 <= row[key] <= 1000 for key in counts)), 'Invalid per-seed answer count')
        require(row['shortest_path_correct'] + row['valid_nonshortest_count'] == row['valid_path_correct'] and row['valid_path_correct'] + row['invalid_path_count'] == 1000 and (row['canonical_correct'] + row['alternative_shortest_count'] + row['reference_route_wrong_bos_count'] == row['shortest_path_correct']), 'Answer-count partitions disagree')
        require(row['shortest_path_accuracy'] == row['shortest_path_correct'] / 1000 and row['canonical_accuracy'] == row['canonical_correct'] / 1000, 'Per-seed accuracy disagrees with verified counts')
    output = dict(seeds=seeds, questions_per_seed=1000, metric_name=METRIC_NAME, metric_version=METRIC_VERSION)
    for metric in ('shortest_path_accuracy', 'canonical_accuracy', 'valid_nonshortest_count', 'alternative_shortest_count', 'reference_route_wrong_bos_count', 'invalid_path_count'):
        output[metric] = stats([row[metric] for row in rows])
    output['canonical_delta_vs_old_count'] = [row['canonical_delta_vs_old_count'] for row in rows]
    output['valid_path_accuracy'] = stats([row['valid_path_correct'] / 1000 for row in rows])
    return output

def aggregate_jobs(jobs, datasets, seeds):
    require(len(jobs) == len(seeds) and sorted((job['seed'] for job in jobs)) == seeds, 'Missing or duplicate original-RoPE jobs')
    for job in jobs:
        require(job['id'] == f"RT_D_seed{job['seed']}" and job['variant'] == 'RT_D' and (job['rope_variant'] == 'original'), 'Wrong original-RoPE job identity')
        require(len(job['datasets']) == len(datasets) and {row['dataset'] for row in job['datasets']} == set(datasets) and all((row['seed'] == job['seed'] and row['job_id'] == job['id'] for row in job['datasets'])), 'Incomplete or mismatched per-job datasets')
    return {'original': {name: aggregate_rows([row for job in jobs for row in job['datasets'] if row['dataset'] == name], seeds) for name in datasets}}

def load_historical_rt(manifest):
    entry = manifest['historical_RT']
    require(entry['seeds'] == [1, 2, 3], 'Historical RT seed inventory changed')
    snapshot, root = (ROOT / entry['snapshot'], Path(entry['root']))
    require(sha256(snapshot) == entry['sha256'] == sha256(root / 'final_results.json'), 'Historical RT result differs from the frozen snapshot')
    status = json.loads((root / 'summary_status.json').read_text())
    require(status['status'] == 'completed' and status['final_results_sha256'] == entry['sha256'], 'Historical RT summary has not completed verification')
    old = json.loads(snapshot.read_text())
    require(old['status'] == 'completed' and old['verification_device'] == 'CPU' and (old['metric_name'] == METRIC_NAME) and (old['metric_version'] == METRIC_VERSION) and (old['scorer_sha256'] == manifest['scorer_sha256']) and (old['T'] == 16) and (old['weights'] == 'EMA') and (old['checkpoint_epoch_zero_based'] == 19), 'Historical RT evaluation protocol differs')
    require(old['manifest_sha256'] == sha256(ROOT / 'baseline/eval_manifest.json') and old['launch_freeze_sha256'] == sha256(ROOT / 'baseline/launch_freeze.json'), 'Historical RT manifest/freeze differs from its audited report')
    old_manifest = json.loads((ROOT / 'baseline/eval_manifest.json').read_text())
    require(set(old['datasets']) == set(old_manifest['datasets']) == set(manifest['datasets']), 'Historical RT must contain the same twelve test datasets')
    for name, source in manifest['datasets'].items():
        for previous in (old['datasets'][name], old_manifest['datasets'][name]):
            require(previous['sha256'] == source['sha256'] and previous['size'] == source['size'] and (previous['count'] == source['count'] == 1000), f'Historical RT dataset differs: {name}')
    jobs = sorted((job for job in old['verified_jobs'] if job['rope_variant'] == 'original'), key=lambda job: job['seed'])
    table = aggregate_jobs(jobs, manifest['datasets'], [1, 2, 3])
    require(table['original'] == old['aggregates']['original'], 'Historical RT aggregates disagree with their verified per-seed rows')
    for job in jobs:
        require(sha256(job['metrics_file']) == job['metrics_sha256'] and sha256(job['completion_file']) == job['completion_sha256'], f"Historical RT committed artifacts changed: seed{job['seed']}")
        completion = json.loads(Path(job['completion_file']).read_text())
        require(completion['status'] == 'completed' and completion['seed'] == job['seed'] and (completion['rope_variant'] == 'original') and (completion['checkpoint_sha256'] == job['checkpoint_sha256']), 'Historical RT completion identity differs')
    return jobs

def render_report(result):
    lines = ['Complete ten-seed RT-D evaluation: original 2D RoPE and any valid shortest path', 'Training seeds 4–10 are new; seeds 1–3 reuse completed CPU-verified results without retraining.', 'Fixed final epoch-20 EMA, T16; all 1,000 instances per test set and seed.', 'Training sizes 15/20/25/30 with 250 instances each, 1,000 total; sizes 32 and above are out of distribution by size; LR = 1e-4, WD = 0.5.', 'This summary includes original 2D RoPE only, not Bandwise.', '± denotes sample SD across ten training seeds (ddof = 1).', '', 'Dataset | Shortest-path success (%, mean±SD) | Individual success rates for seeds 1–10 (%)']
    for name, row in result['aggregates']['original'].items():
        value = row[METRIC_NAME]
        lines.append(f"{name} | {100 * value['mean']:.2f}±{100 * value['sample_std']:.2f} | " + ' / '.join((f'{100 * v:.1f}' for v in value['per_seed'])))
    lines += ['', 'Independent results for the seven new seeds are in final_results.json; combined ten-seed results are in combined_results.json.', 'All new NPZ outputs are rescored on CPU, verifying original CSV/SHA, 1,000 complete unique instances, per-instance indices, targets and checkpoints.', 'The three historical seeds are reused after cross-checking frozen reports, completion status, scorer, T/EMA, all data hashes and per-seed summaries.']
    return '\n'.join(lines) + '\n'

def write_combined_csv(result):
    per_seed = []
    for job in sorted(result['verified_jobs'], key=lambda job: job['seed']):
        for row in job['datasets']:
            per_seed.append(dict(dataset=row['dataset'], size=result['datasets'][row['dataset']]['size'], seed=job['seed'], rope_variant='original', source_group='historical_1_3' if job['seed'] <= 3 else 'new_4_10', questions=1000, shortest_path_accuracy_percent=100 * row['shortest_path_accuracy'], canonical_accuracy_percent=100 * row['canonical_accuracy'], valid_path_accuracy_percent=row['valid_path_correct'] / 10))
    summary = []
    for name, row in result['aggregates']['original'].items():
        summary.append(dict(dataset=name, size=result['datasets'][name]['size'], rope_variant='original', n_seeds=len(row['seeds']), questions_per_seed=1000, shortest_path_accuracy_mean_percent=100 * row[METRIC_NAME]['mean'], shortest_path_accuracy_sample_sd_percent=100 * row[METRIC_NAME]['sample_std']))
    for name, rows in [('combined_per_seed.csv', per_seed), ('combined_summary.csv', summary)]:
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        parent.atomic_write(ROOT / name, stream.getvalue())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data', action='store_true', help='CPU data/reference/baseline audit only; no output files')
    args = parser.parse_args()
    manifest = read_manifest(require_freeze=not args.check_data)
    sources, arrays = load_sources(manifest)
    baselines = load_baselines(manifest)
    historical_jobs = load_historical_rt(manifest)
    if args.check_data:
        print(json.dumps(dict(status='passed', datasets=len(sources), unique_rows=12000, shortest_reference_rows=12000, frozen_baselines=list(baselines), historical_RT_seeds=[job['seed'] for job in historical_jobs], new_seeds=manifest['seeds'], variants=manifest['variants'])))
        return 0
    missing = [f'{v}/seed{s}' for v in manifest['variants'] for s in manifest['seeds'] if not (ROOT / 'evaluations' / v / f'seed{s}' / 'completion.json').is_file()]
    if missing:
        write_json(ROOT / 'summary_status.json', dict(status='pending', pending=missing))
        print(json.dumps(dict(status='pending', pending=missing)))
        return 2
    jobs = [verify_job(s, v, manifest, arrays) for v in manifest['variants'] for s in manifest['seeds']]
    aggregates = aggregate_jobs(jobs, sources, manifest['seeds'])
    result = dict(status='completed', metric_name=METRIC_NAME, metric_version=METRIC_VERSION, T=16, weights='EMA', checkpoint_epoch_zero_based=19, scorer_sha256=manifest['scorer_sha256'], manifest_sha256=sha256(ROOT / 'eval_manifest.json'), launch_freeze_sha256=sha256(ROOT / 'launch_freeze.json'), summarizer_sha256=sha256(__file__), verification_device='CPU', datasets=sources, verified_jobs=jobs, aggregates=aggregates, training_seeds=manifest['seeds'], rope_variants=['original'], HRM_D_comparison=baselines, baseline_sources=manifest['HRM_D_baselines'], comparison_caveat='RT LR1e-4 vs historical HRM-D LR6e-5; not an architecture-only ablation')
    combined_jobs = historical_jobs + jobs
    combined = dict(result, verified_jobs=combined_jobs, aggregates=aggregate_jobs(combined_jobs, sources, list(range(1, 11))), training_seeds=list(range(1, 11)), new_training_seeds=manifest['seeds'], historical_RT=manifest['historical_RT'], new_campaign_root=str(ROOT), historical_verification='Reuse CPU-audited frozen original-RoPE results; protocol, dataset hashes, committed artifacts and per-seed aggregate consistency rechecked')
    for name, payload in [('final_results.json', result), ('combined_results.json', combined)]:
        path = ROOT / name
        require(not path.exists() or json.loads(path.read_text()) == payload, f'Refusing to replace different published results: {name}')
        write_json(path, payload)
    write_combined_csv(combined)
    parent.atomic_write(ROOT / 'report.txt', render_report(combined))
    write_json(ROOT / 'summary_status.json', dict(status='completed', prediction_evaluations=84000, combined_prediction_evaluations=120000, historical_prediction_evaluations=36000, final_results_sha256=sha256(ROOT / 'final_results.json'), combined_results_sha256=sha256(ROOT / 'combined_results.json')))
    print(json.dumps(dict(status='completed', prediction_evaluations=84000, combined_prediction_evaluations=120000)))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
