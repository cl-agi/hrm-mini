"""Paired fixed-budget and stable-16 TTS on the final epoch, with disjoint shards."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from hrm.training.trainer import load_module
from hrm.evaluation.config import load_config
from hrm.analyse.sudoku.mixer.eval import select_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    args = parser.parse_args()
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    ckpt = select_checkpoint(args.ckpt, 'latest')
    config = load_config(ckpt)
    loader, metadata = load_module(f'dataset.{config.data.name}@create_dataloader')(
        config.data.eval_split, config.local_batch_size, rank=rank, world_size=world,
        **config.data.__pydantic_extra__)
    with torch.device('cuda'):
        model = load_module(f'arch.{config.arch.name}')(config.model_config_dict(metadata))
        model.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=True), assign=True)
        model.eval()
        model = torch.compile(model, fullgraph=True, dynamic=False)
    args.output.mkdir(parents=True, exist_ok=True)
    records = {str(b): {'fixed': [], 'stable16': [], 'cycles': []} for b in args.budgets}
    indices = []
    offset = 0
    started = time.monotonic()
    with torch.inference_mode():
        for x, y in loader:
            n = len(x)
            indices.extend(rank + world * np.arange(offset, offset + n))
            offset += n
            x, y = x.cuda(), y.cuda()
            if n < config.local_batch_size:
                x = torch.cat([x, x.new_zeros((config.local_batch_size - n, x.shape[1]))])
            carry = {k: v.expand(len(x), x.shape[1], -1).clone() for k, v in model.initial_carry.items()}
            previous = torch.full_like(x, -1)
            stable = torch.zeros(len(x), dtype=torch.long, device='cuda')
            stopped = torch.zeros_like(stable)
            frozen_correct = torch.zeros(n, dtype=torch.bool, device='cuda')
            for cycle in range(1, max(args.budgets) + 1):
                carry, logits = model(carry, x)
                pred = logits.argmax(-1)
                correct = (pred[:n] == y).all(-1)
                stable = torch.where((pred == previous).all(-1), stable + 1, 0)
                new_stop = (stopped == 0) & (stable >= 16)
                frozen_correct = torch.where(new_stop[:n], correct, frozen_correct)
                stopped = torch.where(new_stop, cycle, stopped)
                previous = pred
                if cycle in args.budgets:
                    rec = records[str(cycle)]
                    rec['fixed'].extend(correct.cpu().tolist())
                    rec['stable16'].extend(torch.where(stopped[:n] > 0, frozen_correct, correct).cpu().tolist())
                    rec['cycles'].extend(torch.where(stopped[:n] > 0, stopped[:n], cycle).cpu().tolist())
            print(json.dumps({'rank': rank, 'processed': offset, 'seconds': time.monotonic() - started}), flush=True)
    arrays = {'indices': np.array(indices)}
    for b, rec in records.items():
        arrays.update({f'{b}_{k}': np.array(v) for k, v in rec.items()})
    np.savez_compressed(args.output / f'rank_{rank}.npz', **arrays)
    dist.barrier()
    if rank == 0:
        shards = [np.load(args.output / f'rank_{r}.npz') for r in range(world)]
        ids = np.concatenate([s['indices'] for s in shards])
        assert sorted(ids.tolist()) == list(range(len(loader.dataset)))
        merged = {key: np.concatenate([s[key] for s in shards])[np.argsort(ids)] for key in arrays}
        np.savez_compressed(args.output / 'per_example.npz', **merged)
        results = []
        for b in args.budgets:
            for policy in ['fixed', 'stable16']:
                correct = int(merged[f'{b}_{policy}'].sum())
                results.append({'max_cycles': b, 'policy': policy, 'correct': correct, 'total': len(ids),
                    'exact_match': correct / len(ids),
                    'mean_cycles': b if policy == 'fixed' else float(merged[f'{b}_cycles'].mean())})
        summary = {'checkpoint': str(ckpt), 'selection': 'final epoch (no test-based selection)',
            'elapsed_seconds': time.monotonic() - started, 'results': results}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
