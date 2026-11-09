from hrm.tasks.io import load_dataset
from typing import Optional
from functools import partial
import numpy as np
import torch
from torch import Tensor
from datasets import Features, Value
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from hrm.tasks.sudoku import shuffle_sudoku
from hrm.tasks.sudoku import collate_fn
from hrm.tasks.sudoku import _worker_init_fn

def create_dataloader(split: str, batch_size: int, rank: int, world_size: int, dataset_name: str, augment: bool=False, repeat: int=1, seed: int=42, samples_per_epoch: Optional[int]=None):
    is_train = split == 'train'
    dataset: Dataset = load_dataset(dataset_name, split=split, features=Features({'question': Value('string'), 'answer': Value('string')})).repeat(repeat if is_train else 1)
    if is_train and samples_per_epoch is not None:
        dataset = dataset.select(np.arange(samples_per_epoch) % len(dataset))
    return (DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, augment=augment and is_train), sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=is_train, drop_last=True, seed=seed), drop_last=True, pin_memory=True, persistent_workers=True, worker_init_fn=partial(_worker_init_fn, base_seed=seed), num_workers=1, prefetch_factor=2), {'vocab_size': 10, 'seq_len': 82, 'is_causal': False})
