"""Original Sudoku augmentation; exact padded distributed evaluation counts."""
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from hrm.tasks.sudoku import collate_fn, _worker_init_fn


class Puzzles(Dataset):
    def __init__(self, path, repeat=1):
        with np.load(path) as data:
            self.q, self.y = data['puzzles'], data['solutions']
        self.repeat = repeat

    def __len__(self):
        return len(self.q) * self.repeat

    def __getitem__(self, i):
        valid = i < len(self)
        i %= len(self.q)
        return dict(question=bytes(self.q[i].astype(np.uint8) + 48).decode(),
                    answer=bytes(self.y[i].astype(np.uint8) + 48).decode(), valid=valid)


def eval_collate(batch):
    x, y = collate_fn(batch, augment=False)
    return x, y, torch.tensor([r['valid'] for r in batch], dtype=torch.bool)


def create_dataloader(split, batch_size, rank, world_size, dataset_name,
                      augment=False, repeat=1, seed=42):
    train = split == 'train'
    dataset = Puzzles(Path(dataset_name) / f'{split}.npz', repeat if train else 1)
    if train:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True, seed=seed)
    else:
        # Fixed full batches avoid torch.compile recompilation for each band's tail.
        # All padded rows are excluded from both correct and total by valid.
        global_batch = batch_size * world_size
        padded = ((len(dataset) + global_batch - 1) // global_batch) * global_batch
        sampler = list(range(rank, padded, world_size))
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=partial(collate_fn, augment=augment) if train else eval_collate,
        drop_last=train, num_workers=1, pin_memory=True, persistent_workers=True,
        worker_init_fn=partial(_worker_init_fn, base_seed=seed), prefetch_factor=2)
    return loader, dict(vocab_size=10, seq_len=82, is_causal=False)
