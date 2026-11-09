from hrm.tasks.io import load_dataset
from functools import partial
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from datasets import Features, Value, load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
SUDOKU_FEATURES = Features({'source': Value('string'), 'question': Value('string'), 'answer': Value('string'), 'rating': Value('int64')})

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = np.random.rand() < 0.5
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    positions = np.ix_(row_perm, col_perm)

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        return digit_map[x[positions]]
    return (apply_transformation(board), apply_transformation(solution))
from hrm.tasks.sudoku import collate_fn
from hrm.tasks.sudoku import _worker_init_fn

def create_dataloader(split: str, batch_size: int, rank: int, world_size: int, dataset_name: str, augment: bool=False, repeat: int | float=1, seed: int=42, train_size: int | None=None, test_size: int | None=None, subsample_seed: int=42, prepared_data_path: str | None=None):
    is_train = split == 'train'
    size = train_size if is_train else test_size
    if prepared_data_path is not None:
        root = Path(prepared_data_path)
        with (root / 'metadata.json').open() as f:
            prepared = json.load(f)
        if prepared['dataset'] != dataset_name or prepared['subsample_seed'] != subsample_seed:
            raise ValueError('Prepared dataset name/subsample_seed does not match the requested data config')
        info = prepared['splits'][split]
        dataset = load_from_disk(root / info['directory'])
        if len(dataset) != info['subset_size']:
            raise ValueError(f'Prepared {split} row count does not match its metadata')
        if size is None and len(dataset) != info['full_size']:
            raise ValueError(f'Prepared {split} contains a subset; an explicit size is required')
    else:
        dataset = load_dataset(dataset_name, split=split, features=SUDOKU_FEATURES)
    if size is not None:
        if not 1 <= size <= len(dataset):
            raise ValueError(f'{split} size must be between 1 and {len(dataset)}, got {size}')
        if prepared_data_path is None:
            dataset = dataset.shuffle(seed=subsample_seed)
        dataset = dataset.select(range(size))
    dataset = dataset.select_columns(['question', 'answer'])
    if is_train:
        repeated_rows = len(dataset) * repeat
        if not math.isfinite(repeated_rows) or repeated_rows < 1 or int(repeated_rows) != repeated_rows:
            raise ValueError('Training repeat must produce a positive integer number of rows')
        dataset = dataset.repeat(math.ceil(repeat))
        if len(dataset) != int(repeated_rows):
            dataset = dataset.select(range(int(repeated_rows)))
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True, drop_last=True, seed=seed) if is_train else range(rank, len(dataset), world_size)
    return (DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, augment=augment and is_train), sampler=sampler, drop_last=is_train, pin_memory=True, persistent_workers=True, worker_init_fn=partial(_worker_init_fn, base_seed=seed), num_workers=1, prefetch_factor=2), {'vocab_size': 10, 'seq_len': 82, 'is_causal': False})
