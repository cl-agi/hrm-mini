"""Final RT-D EMA evaluation on twelve frozen Maze sets, with fixed RoPE variants."""
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
WORKSPACE = ROOT.parent
RUNTIME = ROOT / 'runtime'
pass
from hrm.models.rt import RecurrentTransformer
from hrm.analyse.maze.rt_extrapolation.evaluation import configure_compiler
from hrm.analyse.maze.rt_extrapolation.experiment_config import resolved_config, run_identity, write_json
from hrm.analyse.maze.extrapolation.maze_dataset import RepeatedMazeDataset, _read_verified_csv, _loader_kwargs
from hrm.analyse.maze.rt_extrapolation.maze_metrics import METRIC_NAME, METRIC_VERSION, score_predictions
from hrm.analyse.maze.rt_extrapolation.rope_scaling import apply_scaling, describe

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_manifest():
    manifest = json.loads((ROOT / 'eval_manifest.json').read_text())
    if manifest['cycles'] != 16 or manifest['metric_version'] != METRIC_VERSION or manifest['variants'] != ['original'] or (manifest['world_size'] != 8) or (len(manifest['datasets']) != 12) or (manifest['checkpoint_epoch_zero_based'] != 19):
        raise ValueError('Wrong frozen evaluation protocol')
    if sha256(RUNTIME / 'maze_metrics.py') != manifest['scorer_sha256']:
        raise ValueError('Scorer differs from the fixed shortest-path criterion')
    for path, digest in manifest['frozen_files_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Frozen source or data changed: {path}')
    for parent in manifest['HRM_D_baselines'].values():
        if sha256(parent['path']) != parent['sha256']:
            raise ValueError('Previously verified HRM-D result changed')
    for name, spec in manifest['datasets'].items():
        path = Path(spec['path'])
        if path.resolve() != (Path(spec['data_root']) / spec['relative_path']).resolve() or sha256(path) != spec['sha256'] or spec['count'] != 1000:
            raise ValueError(f'Frozen evaluation source changed: {name}')
    return manifest

def load_context(seed, run_dir):
    manifest = load_manifest()
    freeze = json.loads((ROOT / 'launch_freeze.json').read_text())
    for path, digest in freeze['sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Frozen campaign source changed: {path}')
    run_dir = Path(run_dir).resolve()
    if run_dir != (ROOT / 'runs' / f'RT_D_seed{seed}').resolve():
        raise ValueError('Use the corresponding frozen RT-D seed run directory')
    config = resolved_config('D', seed)
    metadata = json.loads((run_dir / 'run_metadata.json').read_text())
    identity = run_identity('D', seed=seed)
    if metadata['identity'] != identity:
        raise ValueError('Training identity differs; no historical migration is allowed')
    if json.loads((run_dir / 'resolved_config.json').read_text()) != config:
        raise ValueError('Training resolved config differs')
    if config['architecture_name'] != 'RT' or config['training']['cycles_per_data'] != 16 or config['architecture']['rope_type'] != '2d' or (config['architecture']['cycles'] != 7) or (config['training']['lr'] != 0.0001) or (config['training']['weight_decay'] != 0.5):
        raise ValueError('Expected the approved RT-D configuration')
    checkpoint = run_dir / 'checkpoints/epoch_19.pt'
    job = dict(seed=seed, run_dir=str(run_dir), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    architecture = dict(config['architecture'])
    architecture['allowed_sizes'] = sorted(set(architecture['allowed_sizes']) | {45, 50, 60, 80})
    return (manifest, job, config, architecture, identity)

@torch.inference_mode()
def evaluate(model, manifest, config, output, rank, world_size, variant):
    results = {}
    for name, spec in manifest['datasets'].items():
        started = time.monotonic()
        path = Path(spec['path'])
        if sha256(path) != spec['sha256']:
            raise ValueError(f'Dataset changed: {name}')
        rows, _ = _read_verified_csv(Path(spec['data_root']), spec['relative_path'])
        if len(rows) != 1000:
            raise ValueError(f'Expected the complete 1000-question test: {name}')
        size = spec['size']
        indices = np.arange(rank, 1000, world_size)
        loader = DataLoader(RepeatedMazeDataset(rows, [size] * len(rows)), batch_size=spec['local_batch_size'], sampler=indices.tolist(), drop_last=False, **_loader_kwargs(config['runtime']['num_workers']))
        saved = {key: [] for key in ('inputs', 'targets', 'predictions', 'shortest_path_correct', 'valid_path_correct', 'canonical_correct', 'format_correct', 'shortest_distance', 'predicted_steps')}
        counts = np.zeros(5, dtype=np.int64)
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            print(f"RT_D_EVAL_START variant={variant} dataset={name} T=16 local_batch={spec['local_batch_size']} examples=1000", flush=True)
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
                print(f'RT_D_EVAL_PROGRESS dataset={name} batch={batch + 1}/{len(loader)} elapsed_seconds={time.monotonic() - started:.1f}', flush=True)
        destination = output / 'metrics_predictions' / name
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / f'rank_{rank}.npz').open('xb') as stream:
            np.savez_compressed(stream, indices=indices, rank=rank, world_size=world_size, metric_version=METRIC_VERSION, metric_name=METRIC_NAME, rope_variant=variant, cycles=16, **{key: np.concatenate(values) for key, values in saved.items()})
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
            print(f'RT_D_EVAL_RESULT variant={variant} dataset={name} shortest={correct}/1000 canonical={canonical}/1000', flush=True)
        del loader, saved
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=list(range(4, 11)), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=['original'], required=True)
    args = parser.parse_args()
    context = load_context(args.seed, args.run_dir)
    manifest, job, config, architecture, identity = context
    if os.environ.get('WORLD_SIZE') != '8':
        raise ValueError('The established evaluation layout requires eight ranks')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    rank, world = (dist.get_rank(), dist.get_world_size())
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / 'run_metadata.json', dict(seed=args.seed, checkpoint=job['checkpoint'], rope_variant=args.variant, rope=describe(args.variant, [s['size'] for s in manifest['datasets'].values()]), checkpoint_sha256=job['checkpoint_sha256'], architecture=architecture, original_architecture=config['architecture'], evaluation_identity=identity, training_identity=json.loads((Path(job['run_dir']) / 'run_metadata.json').read_text())['identity'], manifest_sha256=sha256(ROOT / 'eval_manifest.json'), launch_freeze_sha256=sha256(ROOT / 'launch_freeze.json'), evaluation_source_sha256={name: sha256(ROOT / name) for name in ('evaluate_sizes.py', 'rope_scaling.py')}, created_utc=datetime.now(timezone.utc).isoformat(), torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), world_size=world, float32_matmul_precision=torch.get_float32_matmul_precision(), cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))
        dist.barrier()
        configure_compiler(config['runtime'])
        with torch.device('cuda'):
            model = RecurrentTransformer(architecture)
        weights = torch.load(job['checkpoint'], map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        apply_scaling(model, args.variant, architecture['allowed_sizes'])
        actual = model.state_dict()
        if set(actual) != set(weights) or any((not torch.equal(actual[k].cpu(), v) for k, v in weights.items())):
            raise ValueError('Evaluation changed the checkpoint state_dict')
        del actual, weights
        model.eval()
        model = torch.compile(model, dynamic=False, fullgraph=True)
        metrics = evaluate(model, manifest, config, args.output, rank, world, args.variant)
        if rank == 0:
            write_json(args.output / 'metrics.json', dict(experiment='RT_D', seed=args.seed, epoch=19, architecture_name='RT', rope_variant=args.variant, weights='EMA', T=16, checkpoint=job['checkpoint'], checkpoint_sha256=job['checkpoint_sha256'], metric_version=METRIC_VERSION, metric_name=METRIC_NAME, metrics=metrics, manifest_sha256=sha256(ROOT / 'eval_manifest.json')))
            write_json(args.output / 'completion.json', dict(status='completed', seed=args.seed, rope_variant=args.variant, checkpoint_sha256=job['checkpoint_sha256'], manifest_sha256=sha256(ROOT / 'eval_manifest.json'), files={str(path.relative_to(args.output)): sha256(path) for path in sorted(args.output.rglob('*')) if path.is_file()}))
        dist.barrier()
    finally:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
