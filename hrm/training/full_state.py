"""Epoch-boundary checkpoints; one worker per loader, as in vanilla HRM."""
import os
import pickle
import random
import multiprocessing as mp
import numpy as np
import torch
import torch.distributed as dist


def rng_state(cuda=False):
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state())
    if cuda:
        state['torch_cuda'] = torch.cuda.get_rng_state()
    return state


class TrackedCollate:
    def __init__(self, collate):
        self.collate = collate
        self.buffer = mp.Array('B', 32768)
        self.size = mp.Value('i', 0)

    def __call__(self, batch):
        output = self.collate(batch)
        payload = pickle.dumps(rng_state())
        assert len(payload) <= len(self.buffer)
        with self.buffer.get_lock():
            self.buffer[:len(payload)] = payload
            self.size.value = len(payload)
        return output

    def state_dict(self):
        with self.buffer.get_lock():
            if not self.size.value:
                return None
            return pickle.loads(bytes(self.buffer[:self.size.value]))


def track_loader(loader):
    assert loader.num_workers == 1 and loader.persistent_workers
    loader.collate_fn = TrackedCollate(loader.collate_fn)


def loader_state(loader):
    sampler = loader.sampler
    return dict(worker_rng=loader.collate_fn.state_dict(), num_workers=loader.num_workers,
                sampler=dict(epoch=sampler.epoch, seed=sampler.seed, rank=sampler.rank,
                             num_replicas=sampler.num_replicas, shuffle=sampler.shuffle,
                             drop_last=sampler.drop_last),
                generator_state=loader.generator.get_state() if loader.generator is not None else None,
                iterator_base_seed=loader._iterator._base_seed,
                iterator_num_yielded=loader._iterator._num_yielded,
                batches_per_epoch=len(loader), at_epoch_boundary=True)


def save_checkpoint(path, model, optim, config, seed, epoch, step, total_steps, loaders, rank, world_size):
    local = dict(rank=rank, rng=rng_state(cuda=True),
                 loaders={name: loader_state(loader) for name, loader in loaders.items()})
    ranks = [None] * world_size if rank == 0 else None
    dist.gather_object(local, ranks, dst=0)
    if rank == 0:
        bare = getattr(model.module, '_orig_mod', model.module)
        checkpoint = dict(format_version=1, model=bare.state_dict(), optimizer=optim.state_dict(),
                          model_weight_state='raw', ema_location='optimizer.state.*.param_ema',
                          config=config.model_dump(), seed=seed, completed_epochs=epoch + 1,
                          optimizer_steps=step, outer_steps=step * config.cycles_per_data,
                          scheduler=dict(kind='functional_update_lr', step=step, total_steps=total_steps,
                                         lr_warmup_steps=config.lr_warmup_steps, lr=config.lr,
                                         lr_min_ratio=config.lr_min_ratio),
                          rank_states=ranks, world_size=world_size,
                          grad_scaler=None, gradients=None, carry=None,
                          boundary='after_eval_and_ema_swap_back; before_next_epoch',
                          torch_version=str(torch.__version__))
        temporary = path + '.tmp'
        with open(temporary, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        print(f'FULL_CHECKPOINT_SAVED {path}', flush=True)
    dist.barrier()
