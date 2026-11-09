"""Serializable RNG state and a sampler cursor for exact batch-boundary restart."""
import itertools
import random
import numpy as np
import torch
from torch.utils.data import Sampler


def capture_rng(include_cuda=True):
    n = np.random.get_state()
    state = dict(python=random.getstate(), numpy=dict(kind=n[0], keys=n[1].tolist(), pos=n[2], has_gauss=n[3], cached_gauss=n[4]), torch_cpu=torch.get_rng_state())
    if include_cuda and torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state()  # This DDP rank's device only.
    return state


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['kind'], np.array(n['keys'], dtype=np.uint32), n['pos'], n['has_gauss'], n['cached_gauss']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if 'torch_cuda' in state:
        torch.cuda.set_rng_state(state['torch_cuda'].cpu())


class ResumeSampler(Sampler):
    def __init__(self, base, batch_size, skip_batches=0):
        self.base = base
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        return itertools.islice(iter(self.base), self.skip_batches * self.batch_size, None)

    def __len__(self):
        # Keep full-epoch length for budgets and epoch-fraction accounting.
        return len(self.base)

    def state_dict(self):
        return dict(seed=self.base.seed, epoch=self.base.epoch, rank=self.base.rank,
                    num_replicas=self.base.num_replicas, shuffle=self.base.shuffle,
                    drop_last=self.base.drop_last, batch_size=self.batch_size,
                    dataset_length=len(self.base.dataset))
