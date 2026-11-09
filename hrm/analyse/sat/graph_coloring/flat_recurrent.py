"""Frozen HRM/RT inference on SATLIB flat200-479; no model/source edits."""
from hrm.paths import analysis_directory
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
import torch

from .flat_metrics import audit_rank, check_predictions, digest, next_tasks, require, validate_original_cnf

ROOT = analysis_directory(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('rank', type=int)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--cycles', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--eager', action='store_true')
    args = parser.parse_args()
    model_dir, rank = args.model_dir.resolve(), args.rank
    cfg = json.loads((model_dir/'config.json').read_text())
    cfg_sha, worker_sha, metrics_sha = digest(model_dir/'config.json'), digest(__file__), digest(Path(__file__).resolve().parent / 'flat_metrics.py')
    world, starts, cycles = cfg['world_size'], cfg['starts'], cfg['inference_cycles']
    size, n_cases = cfg['microbatch'], cfg['cases_per_task']
    require(world == 8 and size in (64, 128, 256) and starts == 256 and cycles == 256 and n_cases == 100, 'Unexpected formal settings')
    require(0 <= rank < world, 'Rank out of range')
    require(args.preflight or (args.cycles is None and args.limit is None and not args.eager), 'Overrides are restricted to preflight')
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(bool(cuda_visible) and ',' not in cuda_visible, 'Worker requires one visible GPU')
    checkpoint = Path(cfg['checkpoint'])
    require(digest(checkpoint) == cfg['checkpoint_sha256'], 'Checkpoint hash differs')
    data_file = Path(cfg['data_file'])
    data_sha = digest(data_file)
    require(data_sha == cfg['data_sha256'], 'Dataset hash differs')
    with np.load(data_file, allow_pickle=False) as z:
        clauses, answers = z['clauses'], z['answers']
    require(clauses.shape == (100, 2237, 3) and answers.shape == (100, 600), 'Unexpected dataset shape')
    require(np.issubdtype(clauses.dtype, np.integer) and np.all(clauses != 0) and np.max(np.abs(clauses)) <= 600, 'Invalid dataset literals')
    require(np.all(answers == 0), 'Expected target-free dummy answers')
    n_variables = answers.shape[1]
    validate_original_cnf(cfg, clauses, n_variables)
    arrays = dict(clauses=clauses, answers=answers)
    assigned = np.arange(rank, n_cases, world)
    if args.preflight:
        cycles = args.cycles if args.cycles is not None else 4
        limit = args.limit if args.limit is not None else len(assigned)
        require(0 < cycles <= cfg['inference_cycles'] and 0 < limit <= len(assigned), 'Invalid preflight budget')
        assigned = assigned[:limit]
        out = model_dir/'preflight'/f'rank{rank:02d}'
        verified = dict(solved=np.zeros(n_cases, bool), attempts=np.zeros(n_cases, np.int64), environments={})
    else:
        out = model_dir/f'rank{rank:02d}'
        verified = audit_rank(model_dir, rank, cfg, clauses, data_sha, worker_sha, metrics_sha)
    out.mkdir(parents=True, exist_ok=True)
    solved, attempts = verified['solved'], verified['attempts']
    began = time.time()

    def status(**details):
        info = dict(model=cfg['model'], rank=rank, world_size=world, pid=os.getpid(),
                    preflight=args.preflight, cuda_visible_devices=cuda_visible, config_sha256=cfg_sha,
                    elapsed_seconds=time.time()-began, time=time.time(), total=len(assigned),
                    full_solved=int(solved[assigned].sum()), executed_starts=int(attempts[assigned].sum()), **details)
        temporary = out/'status.tmp.json'
        temporary.write_text(json.dumps(info, indent=2)+'\n')
        temporary.replace(out/'status.json')
        print(json.dumps(info), flush=True)

    if not args.preflight and np.all(solved[assigned] | (attempts[assigned] == starts)):
        status(state='complete', completion_reason='Raw predictions independently rechecked; all assigned cases complete')
        return
    from hrm.tasks.sat_flow import SATTask
    training_config=json.loads(Path(cfg['training_config']).read_text())
    training_config['model_family']=cfg['model']
    cm=SATTask(training_config, Path(cfg['data_root']))
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    environment = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                       cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu_name=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
                       float32_matmul_precision=torch.get_float32_matmul_precision(),
                       cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                       cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                       compile=dict(enabled=not args.eager, dynamic=False, fullgraph=True),
                       environment={key: os.environ.get(key) for key in ('CUBLAS_WORKSPACE_CONFIG', 'TORCHINDUCTOR_CACHE_DIR')})
    if cfg.get('expected_torch_version'):
        require(environment['torch'] == cfg['expected_torch_version'], 'Torch version differs from pinned runtime')
    for previous in verified['environments'].values():
        for key in ('python', 'torch', 'numpy', 'cuda', 'cudnn', 'gpu_capability', 'float32_matmul_precision',
                    'cuda_matmul_allow_tf32', 'cudnn_allow_tf32', 'deterministic_algorithms', 'compile'):
            require(previous[key] == environment[key], f'Resume runtime changed: {key}')
    environment_json = json.dumps(environment, sort_keys=True, separators=(',', ':'))
    environment_sha = hashlib.sha256(environment_json.encode()).hexdigest()
    environment_tmp = out/'execution_environment.tmp.json'
    environment_tmp.write_text(json.dumps(dict(sha256=environment_sha, **environment), indent=2)+'\n')
    environment_tmp.replace(out/'execution_environment.json')
    status(state='initializing')
    model = cm.model()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    del state
    model.eval()
    run = model if args.eager else torch.compile(model, dynamic=False, fullgraph=True)
    torch.cuda.reset_peak_memory_stats()
    batch_index = len(verified.get('records', []))
    while True:
        ids, task_starts = next_tasks(assigned, solved, attempts, starts, size)
        count = len(ids)
        if not count:
            break
        if (model_dir/'PAUSE').exists():
            status(state='paused')
            raise SystemExit(75)
        padded_ids = np.pad(ids, (0, size-count), mode='edge')
        padded_starts = np.pad(task_starts, (0, size-count), mode='edge')
        seeds = cfg['seed'] + padded_starts*100000 + padded_ids
        batches = [cm.batch(arrays, [int(i)], int(seed), False) for i, seed in zip(padded_ids, seeds)]
        batch = {key: torch.cat([item[key] for item in batches]) for key in ('x', 'mask', 'mix')}
        del batches
        require(batch['x'].shape == (size, 2*n_variables+len(clauses[0])+1), 'Input sequence length differs')
        expected_degrees = np.asarray([[len(set(map(int, clause))) for clause in clauses[i]] for i in padded_ids])
        degrees = batch['mask'][:, 0, 2*n_variables+1:, 1:2*n_variables+1].sum(-1).cpu().numpy()
        require(np.array_equal(degrees, expected_degrees), 'Relation mask differs from original clause incidence')
        cm.bind(model, batch['mask'])
        carry = model.initial_carry | {'mixture_weights': batch['mix']}
        predictions, flags, violation_records, event_pairs = [], [], [], []
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for block in range(0, cycles, 16):
            block_preds, finite_flags = [], []
            for _ in range(block, min(block+16, cycles)):
                event_begin, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                event_begin.record()
                carry, logits = run(carry, batch['x'])
                finite_flags.append(torch.isfinite(logits).all())
                block_preds.append(logits[:count, 1:2*n_variables+1:2].argmax(-1).to(torch.uint8))
                event_end.record()
                event_pairs.append((event_begin, event_end))
            pp = torch.stack(block_preds).cpu().numpy()
            require(bool(torch.stack(finite_flags).all().item()), f'Non-finite logits in batch {batch_index}, block {block}')
            good, violations = check_predictions(pp, clauses[ids])
            predictions.append(pp)
            flags.append(good)
            violation_records.append(violations)
            status(state='running', batch_index=batch_index, current_tasks=count,
                   completed_cycles=min(block+16, cycles), total_cycles=cycles,
                   peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-begin
        cycle_seconds = np.asarray([a.elapsed_time(b)/1000 for a, b in event_pairs])
        pp, ff, vv = np.concatenate(predictions), np.concatenate(flags), np.concatenate(violation_records)
        require(len(pp) == cycles, 'Unexpected incomplete trajectory')
        full_hit = ff.any(0)
        prefix = f'preflight_{"eager" if args.eager else "compiled"}_t{cycles:03d}_' if args.preflight else ''
        destination = out/f'{prefix}b{batch_index:04d}.npz'
        require(not destination.exists(), f'Refusing to overwrite record: {destination}')
        temporary = destination.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, case_ids=ids, starts=task_starts,
                            seeds=seeds[:count], batch_index=batch_index,
                            predictions=pp, success=ff, full_ok=full_hit, violated_clauses=vv,
                            elapsed_seconds=elapsed, executed_cycles=cycles, cycle_cuda_seconds=cycle_seconds,
                            all_logits_finite=True, preflight=args.preflight,
                            checkpoint_sha256=cfg['checkpoint_sha256'], data_sha256=data_sha,
                            config_sha256=cfg_sha, worker_sha256=worker_sha, metrics_sha256=metrics_sha,
                            execution_environment_json=environment_json, execution_environment_sha256=environment_sha)
        temporary.replace(destination)
        # Repeated case IDs denote independent starts; indexed += would lose increments.
        np.add.at(attempts, ids, 1)
        np.logical_or.at(solved, ids, full_hit)
        status(state='running', batch_index=batch_index, last_tasks=count,
               start_min=int(task_starts.min())+1, start_max=int(task_starts.max())+1,
               last_executed_cycles=cycles, last_batch_seconds=elapsed,
               cycle_cuda_seconds=cycle_seconds.tolist(), peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
        batch_index += 1
        del batch, carry, logits, predictions, flags, violation_records, pp, ff, vv, good, block_preds, finite_flags
        if args.preflight:
            break
    require(args.preflight or np.all(solved[assigned] | (attempts[assigned] == starts)), 'Unfinished cases remain')
    status(state='complete', peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())


if __name__ == '__main__':
    main()
