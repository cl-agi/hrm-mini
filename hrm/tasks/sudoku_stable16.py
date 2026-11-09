from hrm.tasks.io import load_dataset
from functools import partial
import json
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from datasets import Features, Value, load_from_disk
from torch.utils.data import DataLoader
from hrm.tasks.sampling_stable16 import InfiniteSampler
SUDOKU_FEATURES = Features({'question': Value('string'), 'answer': Value('string')})

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

def _worker_init_fn(worker_id: int):
    np.random.seed(torch.initial_seed() % 2 ** 32)

def create_dataloader(split: str, batch_size: int, rank: int, world_size: int, dataset_name: str, augment: bool=False, seed: int=42, train_size: int | None=None, test_size: int | None=None, subsample_seed: int=42, prepared_data_path: str | None=None, training: bool=False, num_workers: int=1):
    if num_workers < 0: raise ValueError('num_workers must be nonnegative')
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
    sampler = InfiniteSampler(len(dataset), rank, world_size, seed) if training else range(rank, len(dataset), world_size)
    return (DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, augment=augment and training), sampler=sampler, drop_last=False, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0, worker_init_fn=_worker_init_fn, generator=torch.Generator().manual_seed(seed + rank), num_workers=num_workers, prefetch_factor=2 if num_workers else None), {'vocab_size': 10, 'seq_len': 82, 'is_causal': False})
