"""Evaluate one frozen best checkpoint at all TTS budgets in a single pass."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

if __package__:
    from .evaluate_distributed import atomic_json, validate_complete_results
else:
    from hrm.analyse.sudoku.stable16.evaluate_distributed import atomic_json, validate_complete_results


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_curve_summary(output, checkpoint, protocol, verify_shards=False):
    """Validate every budget; read full shards once before certifying completion."""
    summary = json.loads((output / 'summary.json').read_text())
    cycles, samples, workers = protocol['cycles'], protocol['test_size'], protocol['world_size']
    expected = {'schema_version': 2, 'complete': True, 'checkpoint': str(checkpoint.resolve()),
                'cycle_budgets': cycles, 'stable_cycles': protocol['stable_cycles'], 'shared_inference': True}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError('Combined evaluation summary does not match the campaign')
    if set(summary.get('results', {})) != {str(limit) for limit in cycles}:
        raise ValueError('Combined evaluation summary has missing or extra cycle budgets')
    metric_keys = ('samples', 'correct', 'exact_match', 'total_cycles', 'mean_cycles', 'min_cycles', 'max_cycles')
    for limit in cycles:
        directory = output / f'cycles_{limit}'
        result = json.loads((directory / 'summary.json').read_text())
        expected = {'schema_version': 1, 'complete': True, 'checkpoint': str(checkpoint.resolve()),
                    'requested_max_cycles': limit, 'stable_cycles': protocol['stable_cycles'],
                    'samples': samples, 'expected_samples': samples, 'world_size': workers,
                    'predictions_saved': True, 'split': 'test', 'shared_inference_max_cycles': cycles[-1]}
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Cycle {limit} summary does not match the campaign')
        correct, total = result.get('correct'), result.get('total_cycles')
        if (type(correct) is not int or not 0 <= correct <= samples or
                not abs(result['exact_match'] - correct / samples) <= 1e-12 or
                type(total) is not int or not samples <= total <= samples * limit or
                not abs(result['mean_cycles'] - total / samples) <= 1e-12 or
                not 1 <= result['min_cycles'] <= result['mean_cycles'] <= result['max_cycles'] <= limit):
            raise ValueError(f'Cycle {limit} has inconsistent counts or accuracy')
        entry = summary['results'][str(limit)]
        if (entry.get('summary') != f'cycles_{limit}/summary.json' or
                any(entry.get(key) != result[key] for key in metric_keys)):
            raise ValueError(f'Cycle {limit} differs from the combined summary')
        paths = [directory / f'rank_{rank:05d}.npz' for rank in range(workers)]
        if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
            raise ValueError(f'Cycle {limit} has missing or empty result shards')
        if verify_shards:
            import numpy as np
            shards = []
            for rank, path in enumerate(paths):
                with np.load(path, allow_pickle=False) as saved:
                    shard = {key: saved[key] for key in ('indices', 'correctness', 'cycles')}
                    predictions, inputs = saved['predictions'], saved['inputs']
                    if (predictions.shape != inputs.shape or predictions.ndim != 2 or
                            len(predictions) != len(range(rank, samples, workers)) or
                            not np.issubdtype(predictions.dtype, np.integer)):
                        raise ValueError(f'Cycle {limit} has invalid saved predictions')
                    shards.append(shard)
            measured = validate_complete_results(shards, samples, limit, protocol['stable_cycles'])
            if any(measured[key] != result[key] for key in metric_keys):
                raise ValueError(f'Cycle {limit} summary differs from persisted shards')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('run_id')
    args = parser.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    run = next(run for run in manifest['runs'] if run['run_id'] == args.run_id)
    protocol, source = manifest['protocol'], root / 'source'
    work = root / 'runs' / args.run_id
    work.mkdir(parents=True, exist_ok=True)
    output, checkpoint = work / 'eval', Path(run['checkpoint']).resolve()
    status = {'run_id': args.run_id, 'task_id': os.environ.get('MLP_TASK_ID')}

    def stage(name):
        status.update(stage=name, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(work / 'status.json', status)
        print(f'RUN {args.run_id}: {name}', flush=True)

    try:
        stage('preflight')
        hashes = json.loads((root / 'source_sha256.json').read_text())
        if not hashes or any(sha256(source / name) != digest for name, digest in hashes.items()):
            raise ValueError('Frozen evaluation source checksum mismatch')
        if sha256(checkpoint) != run['checkpoint_sha256']:
            raise ValueError('Selected training checkpoint checksum mismatch')
        import torch
        if (protocol['world_size'] != 8 or int(os.environ.get('MLP_WORKER_GPU', '8')) != 8 or
                torch.cuda.device_count() != 8):
            raise RuntimeError('TTS evaluation requires exactly eight GPUs')
        cached = False
        if (output / 'summary.json').is_file():
            try:
                receipt = json.loads((work / 'result.json').read_text()) if (work / 'result.json').is_file() else {}
                verified = receipt.get('verified_shards') is True and receipt.get('checkpoint_sha256') == run['checkpoint_sha256']
                validate_curve_summary(output, checkpoint, protocol, verify_shards=not verified)
                cached = True
            except (ValueError, TypeError, KeyError, OSError) as error:
                print(f'Preserving invalid cached evaluation before retry: {error}', flush=True)
        if not cached:
            if output.exists() and any(output.iterdir()):
                output.rename(work / ('eval.partial.' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
            stage('evaluating')
            env = dict(os.environ, PYTHONPATH=str(source), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONUNBUFFERED='1', TORCHINDUCTOR_COMPILE_THREADS='2')
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                       '--nproc_per_node=8', str(source / 'evaluate_distributed.py'), '--ckpt', str(checkpoint),
                       '--output-dir', str(output), '--max-cycles', *map(str, protocol['cycles']),
                       '--stable-cycles', str(protocol['stable_cycles']), '--expected-samples',
                       str(protocol['test_size']), '--save-predictions']
            print('COMMAND', ' '.join(command), flush=True)
            with (work / 'eval.log').open('a') as log:
                subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            validate_curve_summary(output, checkpoint, protocol, verify_shards=True)
        atomic_json(work / 'result.json', {'run': run, 'checkpoint': str(checkpoint),
                    'checkpoint_sha256': run['checkpoint_sha256'], 'summary': str(output / 'summary.json'),
                    'verified_shards': True})
        stage('complete')
    except BaseException as error:
        signature = f'{type(error).__name__}: {error}'
        if isinstance(error, subprocess.CalledProcessError) and (work / 'eval.log').is_file():
            with (work / 'eval.log').open('rb') as log:
                log.seek(max(0, log.seek(0, 2) - 65536))
                errors = re.findall(r'(?:^|\n)(?:\[rank\d+\]:\s*)?([\w.]+(?:Error|Exception):[^\n]+)', log.read().decode(errors='replace'))
            signature = next((line for line in reversed(errors) if 'ChildFailedError:' not in line), signature)
        status.update(error=f'{type(error).__name__}: {error}', failure_signature=signature,
                      failed_stage=status.get('stage'))
        stage('failed')
        raise


if __name__ == '__main__':
    main()
