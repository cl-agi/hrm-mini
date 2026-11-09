"""Original Sudoku HRM: sparse width search with epoch-boundary continuation."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from hrm.training.trainer import TrainConfig, load_module, train_step, run_inference, update_lr
from hrm.training.optimizer import AdamATan2
from hrm.training.mup import parameter_groups, group_metadata
from hrm.training.state import prepare_loader, save_checkpoint, restore_training
from hrm.training.extension import trend_decision


class SearchConfig(TrainConfig):
    boundary_lr_factor: float = 1.0
    boundary_weight_decay: float = 1.0


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--benchmark-updates', type=int, default=0)
    parser.add_argument('--stop-after-epochs', type=int, default=0)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    config = SearchConfig(**job['config'])
    assert config.arch.name.split('@')[0].split('.')[-1] in {'hrm','trm','rt'}
    assert len(config.seeds) == 1 and config.lr_min_ratio == 1.0
    assert not any('flow' in key or 'consistency' in key for key in config.arch.model_dump())
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    assert world == job['gpus']
    assert config.local_batch_size * world == job['global_batch']
    seed = config.seeds[0]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f'training_rank{rank}.pt'
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if args.resume else None
    if not args.resume and checkpoint_path.exists():
        raise FileExistsError(f'Refusing to overwrite {checkpoint_path}')
    make_loader = load_module(f'dataset.{config.data.name}@create_dataloader')
    data_args = config.data.__pydantic_extra__ or {}
    train_loader, metadata = make_loader('train', config.local_batch_size, rank, world, seed=seed, **data_args)
    eval_loader, _ = make_loader('test_hard', config.local_batch_size, rank, world, seed=seed, **data_args)
    prepare_loader(train_loader, seed + rank, saved['loader'] if saved else None)
    eval_loader.generator = torch.Generator().manual_seed(seed + 100 + rank)
    if saved:
        eval_loader.generator.set_state(saved['test_generator'])
    steps_per_epoch = config.cycles_per_data * len(train_loader)
    model_cls = load_module(f'arch.{config.arch.name}')
    with torch.device('cuda'):
        native = model_cls((config.arch.__pydantic_extra__ or {}) | metadata)
        groups = parameter_groups(native, config.weight_decay, config.hidden_lr_factor)
        groups[1]['lr_multiplier'] = config.boundary_lr_factor
        groups[1]['weight_decay'] = config.boundary_weight_decay
        model = DDP(torch.compile(native, dynamic=False, fullgraph=True), static_graph=True)
    optimizer = AdamATan2(groups, lr=torch.tensor(0.0), betas=(config.beta1, config.beta2),
                          weight_decay=config.weight_decay, ema=config.ema)
    epoch, step, history, budget = 0, 0, [], config.epochs
    if saved:
        restore_training(saved, model.module, optimizer, config.model_dump())
        epoch, step, history, budget = saved['epoch'], saved['step'], saved['eval_history'], saved['budget']
        if saved['stopped']:
            raise ValueError('Terminal checkpoint: explicitly review budget/selection before extending it')
    if rank == 0:
        write_json(output / 'resolved.json', {'job': job, 'world_size': world, 'parameters': sum(p.numel() for p in native.parameters()),
                   'steps_per_epoch': steps_per_epoch, 'optimizer_groups': group_metadata(optimizer, config.lr),
                   'torch_version': torch.__version__, 'training_source': 'TRM tied H/L, XS muP, CE only'})
    started = time.time()
    first_resume_iterator = bool(saved)
    benchmark_start = None
    run_updates = 0
    while epoch < budget:
        model.train()
        iterator = iter(train_loader)
        if first_resume_iterator:
            train_loader.generator.set_state(saved['loader']['generator'])
            first_resume_iterator = False
        for x, y, worker_rng in iterator:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            carry = model.module.initial_carry
            for _ in range(config.cycles_per_data):
                step += 1
                update_lr(config, optimizer, step, max(budget * steps_per_epoch, config.lr_warmup_steps + 1))
                carry, metrics = train_step(model, carry, optimizer, x, y)
                run_updates += 1
                if args.benchmark_updates and run_updates == 16:
                    torch.cuda.synchronize(); benchmark_start = time.time()
                    torch.cuda.reset_peak_memory_stats()
                if args.benchmark_updates and run_updates == args.benchmark_updates + 16:
                    torch.cuda.synchronize()
                    seconds = torch.tensor(time.time() - benchmark_start, device='cuda')
                    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
                    if rank == 0:
                        write_json(output / 'benchmark.json', {'timed_updates': args.benchmark_updates,
                                   'seconds': seconds.item(), 'seconds_per_update': seconds.item()/args.benchmark_updates,
                                   'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                                   'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                                   'loss': metrics['loss'].item(), 'world_size': world, 'global_batch': job['global_batch']})
                    dist.destroy_process_group()
                    return
            if step % config.log_interval == 0:
                loss = metrics['loss'].float()
                dist.all_reduce(loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                if rank == 0:
                    record = {'epoch': epoch, 'step': step, 'loss': loss.item()/world, 'elapsed_seconds': time.time()-started}
                    write_json(output/'progress.json', record)
                    with (output/'train_metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps(record)+'\n')
            del carry, x, y, metrics
        model.eval()
        optimizer.swap_ema()
        totals = torch.zeros(2, dtype=torch.long, device='cuda')
        for x, y in eval_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            carry = model.module.initial_carry
            for _ in range(config.cycles_per_data):
                carry, prediction = run_inference(model, carry, x)
            totals[0] += prediction.eq(y).all(dim=-1).sum()
            totals[1] += len(y)
        dist.all_reduce(totals)
        accuracy = totals[0].item()/totals[1].item()
        assert totals[1].item() == len(eval_loader.dataset)
        if rank == 0:
            weights = {k.removeprefix('_orig_mod.'):v for k,v in model.module.state_dict().items()}
            if not history or accuracy > max(history):
                torch.save(weights, output/'best.pt')
            torch.save(weights, output/'last.pt')
        optimizer.swap_ema()
        history.append(accuracy)
        epoch += 1
        decision = None
        stopped = False
        if epoch >= budget:
            decision = trend_decision(history, min_gain=0.0005, max_drop=0.002)
            if decision['extend'] and budget < job['max_epochs']:
                budget = min(budget+10, job['max_epochs'])
            else:
                stopped = True
                if decision['extend']:
                    decision['reason'] = 'budget_cap_with_positive_trend_requires_review'
        save_checkpoint(checkpoint_path, model.module, optimizer, epoch, step, budget, history,
                        config.model_dump(), train_loader, worker_rng, eval_loader, stopped)
        dist.barrier()
        if rank == 0:
            record = {'epoch': epoch, 'step': step, 'correct': totals[0].item(), 'total': totals[1].item(),
                      'exact_match': accuracy, 'best_exact_match': max(history), 'best_epoch': history.index(max(history))+1,
                      'budget': budget, 'decision': decision, 'stopped': stopped, 'world_size': world,
                      'elapsed_seconds': time.time()-started}
            with (output/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            write_json(output/'status.json', record)
            print(json.dumps(record), flush=True)
        if stopped or (args.stop_after_epochs and epoch >= args.stop_after_epochs):
            break
    if rank == 0 and stopped:
        write_json(output/'complete.json', record)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
