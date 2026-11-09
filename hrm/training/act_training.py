"""Continuous ACT pool for native train.py; epoch boundaries never drain it."""
import json
import time
from pathlib import Path
import torch
import torch.distributed as dist
import wandb
from hrm.training.outer_schedule import outer_for_epoch


class ACTPool:
    def __init__(self, model, batch, length):
        initial = model.module.initial_carry
        self.device = next(iter(initial.values())).device
        self.x = torch.zeros(batch, length, device=self.device, dtype=torch.int32)
        self.y = torch.zeros_like(self.x)
        self.active = torch.zeros(batch, device=self.device, dtype=torch.bool)
        self.age = torch.zeros(batch, device=self.device, dtype=torch.long)
        self.streak = torch.zeros_like(self.age)
        self.carry = {key: value.expand(batch, length, -1).clone()
                      for key, value in initial.items()}
        self.pending_x = self.pending_y = None
        self.offset = 0
        self.admitted = self.completed = self.solved = self.timed_out = self.outer_sum = 0

    def refill(self, iterator, model, skip=0):
        """Return False immediately on StopIteration, keeping the live pool.

        Native distributed loader batches form one global queue. All ranks
        consume that queue in lockstep while assigning rows to whichever slots
        finish; a fast rank cannot exhaust only its own shard prematurely.
        """
        world, rank = dist.get_world_size(), dist.get_rank()
        slots = (~self.active).nonzero().flatten()
        sizes = [torch.zeros((), device=self.device, dtype=torch.long) for _ in range(world)]
        dist.all_gather(sizes, torch.tensor(len(slots), device=self.device))
        sizes = [int(n) for n in sizes]
        prefix, needed = sum(sizes[:rank]), sum(sizes)
        filled = 0
        while filled < needed or skip:
            if self.pending_x is None or self.offset == len(self.pending_x):
                try:
                    x, y = next(iterator)
                except StopIteration:
                    self.pending_x = self.pending_y = None
                    self.offset = 0
                    return False
                x, y = x.to(self.device), y.to(self.device)
                xs, ys = [torch.empty_like(x) for _ in range(world)], [torch.empty_like(y) for _ in range(world)]
                dist.all_gather(xs, x)
                dist.all_gather(ys, y)
                self.pending_x, self.pending_y = torch.cat(xs), torch.cat(ys)
                self.offset = 0
            if skip:
                take = min(skip, len(self.pending_x) - self.offset)
                skip -= take
                self.offset += take
                continue
            take = min(needed - filled, len(self.pending_x) - self.offset)
            lo, hi = max(filled, prefix), min(filled + take, prefix + len(slots))
            if hi > lo:
                selected = slots[lo - prefix:hi - prefix]
                source = slice(self.offset + lo - filled, self.offset + hi - filled)
                self.x[selected], self.y[selected] = self.pending_x[source], self.pending_y[source]
                for key, initial in model.module.initial_carry.items():
                    self.carry[key][selected] = initial
                self.age[selected] = self.streak[selected] = 0
                self.active[selected] = True
                self.admitted += hi - lo
            filled += take
            self.offset += take
        return True

    def state_dict(self):
        # Only save at loader exhaustion: no opaque DataLoader iterator to restore.
        assert self.pending_x is None
        return {key: ({k: v.detach().cpu() for k, v in value.items()} if key == 'carry'
                      else value.detach().cpu() if isinstance(value, torch.Tensor) else value)
                for key, value in self.__dict__.items() if key != 'device'}

    def load_state_dict(self, saved):
        for key, value in saved.items():
            setattr(self, key, {k: v.to(self.device) for k, v in value.items()} if key == 'carry'
                    else value.to(self.device) if isinstance(value, torch.Tensor) else value)


def train_act_epoch(config, model, optim, loader, epoch, step, total_steps,
                    rank, world, train_step, update_lr, checkpoint_dir, pool, local_skip=0, symmetry_rng=None):
    train_outer = outer_for_epoch(config, epoch)
    loader.sampler.set_epoch(epoch)
    iterator = iter(loader)
    base_admitted = pool.admitted - local_skip
    first_step = step + 1
    started = time.monotonic()
    log = (Path(checkpoint_dir) / 'metrics.jsonl').open('a', buffering=1) if rank == 0 else None
    skip = torch.tensor(local_skip, device=pool.device)
    dist.all_reduce(skip)
    skip = int(skip)
    while pool.refill(iterator, model, skip):
        skip = 0
        # Every optimizer update has a full global batch. Boundary refills that
        # encounter StopIteration return to save/evaluate without another update.
        assert bool(pool.active.all())
        pool.carry['active'] = pool.active.clone()
        step += 1
        lr = update_lr(config, optim, step, total_steps)
        pool.carry, metrics = train_step(model, pool.carry, optim, pool.x, pool.y)
        correct = metrics.pop('_correct')
        pool.age += 1
        pool.streak = torch.where(correct, pool.streak + 1, 0)
        success = pool.streak >= config.act_patience
        timeout = (pool.age >= train_outer) & ~success
        done = success | timeout
        pool.completed += int(done.sum())
        pool.solved += int(success.sum())
        pool.timed_out += int(timeout.sum())
        pool.outer_sum += int(pool.age[done].sum())
        pool.active &= ~done
        if config.outer_symmetry:
            from hrm.training.outer_symmetry import augment_active
            assert symmetry_rng is not None
            pool.x, pool.y, pool.carry = augment_active(
                pool.x, pool.y, pool.carry, pool.active, symmetry_rng)
        if step == first_step or step % config.log_interval == 0:
            keys = sorted(metrics)
            values = torch.stack([metrics[k].float() for k in keys])
            dist.all_reduce(values)
            values /= world
            stats = torch.tensor([pool.admitted-base_admitted, pool.completed, pool.solved,
                                  pool.timed_out, pool.outer_sum], device=pool.device, dtype=torch.long)
            dist.all_reduce(stats)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite training metrics')
            if rank == 0:
                record = dict(zip(keys, values.cpu().tolist()))
                a, c, s, t, o = stats.cpu().tolist()
                record.update(step=step, epoch=epoch, data_epoch=epoch + a/(len(loader)*config.local_batch_size*world),
                              admitted_this_epoch=a, act_completed=c, act_solved=s,
                              act_timed_out=t, act_outer_sum=o, mean_completed_outers=o/max(c, 1),
                              global_batch=config.local_batch_size*world, world_size=world, lr=lr,
                              elapsed=time.monotonic()-started, continuous_act=True, train_outer=train_outer)
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record)+'\n')
                wandb.log({f'train/{k}': v for k, v in record.items() if k != 'step'}, step=step)
    boundary = torch.tensor([int(pool.active.sum()), int(pool.age.sum()), pool.admitted-base_admitted], device=pool.device)
    dist.all_reduce(boundary)
    if rank == 0:
        print(json.dumps(dict(event='loader_exhausted', epoch=epoch, step=step,
                              active_preserved=int(boundary[0]), age_sum_preserved=int(boundary[1]),
                              admitted_this_epoch=int(boundary[2]), drain_updates=0)), flush=True)
        log.close()
    return step
