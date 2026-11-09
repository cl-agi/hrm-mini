"""Evaluate the existing D checkpoints at 45/50/60/80; no model or training edits."""
from hrm.paths import analysis_directory
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
ROOT = analysis_directory(__file__)
WORKSPACE = ROOT.parent.parent
RUNTIME = WORKSPACE / 'Maze-lr6e5-wd05-3seed/d_runtime'
pass
from hrm.models.hrm import HRM
from hrm.analyse.maze.extrapolation.evaluation import configure_compiler
from hrm.analyse.maze.extrapolation.experiment_config import resolved_config, run_identity, validate_evaluation_migration, write_json
from hrm.analyse.maze.extrapolation.maze_dataset import RepeatedMazeDataset, _read_verified_csv, _loader_kwargs
from hrm.analyse.maze.rt_extrapolation.maze_metrics import METRIC_NAME, METRIC_VERSION, score_predictions

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_context(seed):
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if manifest['cycles'] != 16 or manifest['metric_version'] != METRIC_VERSION:
        raise ValueError('Wrong frozen cycle budget or scoring protocol')
    if sha256(ROOT / 'checks.json') != manifest['checks_sha256']:
        raise ValueError('CPU extension check receipt changed')
    if sha256(Path(manifest['large_data_root']) / 'manifest.json') != manifest['large_data_manifest_sha256']:
        raise ValueError('Large data bundle manifest changed')
    for relative, digest in manifest['source_sha256'].items():
        if sha256(Path(manifest['project']) / relative) != digest:
            raise ValueError(f'Runtime changed: {relative}')
    for relative, digest in manifest['extension_source_sha256'].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f'Evaluation extension changed: {relative}')
    parent = manifest['parent_results']
    if sha256(parent['path']) != parent['sha256']:
        raise ValueError('Previously verified primary results changed')
    job, = [job for job in manifest['jobs'] if job['seed'] == seed]
    if sha256(job['checkpoint']) != job['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    config = resolved_config('D', seed)
    metadata = json.loads((Path(job['run_dir']) / 'run_metadata.json').read_text())
    identity = run_identity('D', seed=seed)
    migration = validate_evaluation_migration(metadata['identity'], identity)
    if config['training']['cycles_per_data'] != 16 or config['architecture']['rope_type'] != '2d':
        raise ValueError('Expected the original D two-dimensional RoPE and T16')
    architecture = dict(config['architecture'])
    architecture['allowed_sizes'] = sorted(set(architecture['allowed_sizes']) | {45, 50, 60, 80})
    return (manifest, job, config, architecture, identity, migration)

@torch.inference_mode()
def evaluate(model, manifest, config, output, rank, world_size):
    results = {}
    for name, spec in manifest['datasets'].items():
        started = time.monotonic()
        path = Path(spec['path'])
        if sha256(path) != spec['sha256']:
            raise ValueError(f'Dataset changed: {name}')
        rows, _ = _read_verified_csv(Path(manifest['large_data_root']), spec['relative_path'])
        size = spec['size']
        indices = np.arange(rank, 1000, world_size)
        loader = DataLoader(RepeatedMazeDataset(rows, [size] * len(rows)), batch_size=spec['local_batch_size'], sampler=indices.tolist(), drop_last=False, **_loader_kwargs(config['runtime']['num_workers']))
        saved = {key: [] for key in ('inputs', 'targets', 'predictions', 'shortest_path_correct', 'valid_path_correct', 'canonical_correct', 'format_correct')}
        counts = np.zeros(5, dtype=np.int64)
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            print(f"LARGE_START dataset={name} T=16 local_batch={spec['local_batch_size']} examples=1000", flush=True)
        for batch, (x, y) in enumerate(loader):
            xs, ys = (x.numpy(), y.numpy())
            x = x.cuda()
            carry = model.initial_carry
            finite = torch.ones((), dtype=torch.bool, device='cuda')
            for _ in range(16):
                carry, logits = model(carry, x)
                finite &= torch.isfinite(logits).all()
                prediction = logits.argmax(dim=-1)
            if not finite.item():
                raise FloatingPointError(f'Nonfinite logits: {name}, rank {rank}, batch {batch}')
            predictions = prediction.cpu().numpy()
            scores = score_predictions(xs, predictions, ys, size=size)
            for key, value in {'inputs': xs, 'targets': ys, 'predictions': predictions, **scores}.items():
                if key in saved:
                    saved[key].append(value)
            counts += [scores['shortest_path_correct'].sum(), scores['valid_path_correct'].sum(), scores['canonical_correct'].sum(), scores['format_correct'].sum(), len(ys)]
            del x, y, carry, logits, prediction
            if rank == 0:
                print(f'LARGE_PROGRESS dataset={name} batch={batch + 1}/{len(loader)} elapsed_seconds={time.monotonic() - started:.1f}', flush=True)
        destination = output / 'metrics_predictions' / name
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / f'rank_{rank}.npz').open('xb') as stream:
            np.savez_compressed(stream, indices=indices, rank=rank, world_size=world_size, metric_version=METRIC_VERSION, metric_name=METRIC_NAME, **{key: np.concatenate(values) for key, values in saved.items()})
        totals = torch.as_tensor(counts, dtype=torch.long, device='cuda')
        dist.all_reduce(totals)
        correct, valid, canonical, formatting, total = totals.cpu().tolist()
        if total != 1000:
            raise ValueError(f'Test coverage must be exactly 1000: {name}/{total}')
        memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], dtype=torch.long, device='cuda')
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        results[name] = dict(correct=correct, total=total, accuracy=correct / total, shortest_path_accuracy=correct / total, valid_path_accuracy=valid / total, canonical_correct=canonical, exact_match=canonical / total, format_accuracy=formatting / total, metric_name=METRIC_NAME, metric_version=METRIC_VERSION, size=size, seq_len=size * size + 1, local_batch_size=spec['local_batch_size'], world_size=world_size, cycles=16, elapsed_seconds=time.monotonic() - started, peak_allocated_bytes_max_rank=int(memory[0].item()), peak_reserved_bytes_max_rank=int(memory[1].item()))
        if rank == 0:
            write_json(output / f'{name}_metrics.json', results[name])
            print(f'LARGE_RESULT dataset={name} shortest={correct}/1000 canonical={canonical}/1000', flush=True)
        del loader, saved
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    context = load_context(args.seed)
    manifest, job, config, architecture, identity, migration = context
    if os.environ.get('WORLD_SIZE') != '8':
        raise ValueError('The established evaluation layout requires eight ranks')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    rank, world = (dist.get_rank(), dist.get_world_size())
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / 'run_metadata.json', dict(seed=args.seed, checkpoint=job['checkpoint'], checkpoint_sha256=job['checkpoint_sha256'], architecture=architecture, original_architecture=config['architecture'], evaluation_identity=identity, training_identity=json.loads((Path(job['run_dir']) / 'run_metadata.json').read_text())['identity'], evaluation_migration=migration, manifest_sha256=sha256(ROOT / 'manifest.json'), created_utc=datetime.now(timezone.utc).isoformat(), torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), world_size=world, float32_matmul_precision=torch.get_float32_matmul_precision(), cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))
        dist.barrier()
        configure_compiler(config['runtime'])
        with torch.device('cuda'):
            model = HRM(architecture)
        model.load_state_dict(torch.load(job['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        model.eval()
        model = torch.compile(model, dynamic=False, fullgraph=True)
        metrics = evaluate(model, manifest, config, args.output, rank, world)
        if rank == 0:
            write_json(args.output / 'metrics.json', dict(experiment='D', seed=args.seed, epoch=19, weights='EMA', T=16, checkpoint=job['checkpoint'], checkpoint_sha256=job['checkpoint_sha256'], metric_version=METRIC_VERSION, metric_name=METRIC_NAME, metrics=metrics))
        dist.barrier()
    finally:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
