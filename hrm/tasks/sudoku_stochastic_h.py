from hrm.tasks.io import load_dataset
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from datasets import Features, Value
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from hrm.tasks.sudoku import shuffle_sudoku

def randomly_fill_solution_digits(board: np.ndarray, solution: np.ndarray, remaining_blanks: int | None=None) -> np.ndarray:
    """Reveal solution digits while preserving at least one original blank."""
    blank_indices = np.flatnonzero(board.reshape(-1) == 0)
    if blank_indices.size <= 1:
        return board
    if remaining_blanks is None:
        num_to_fill = np.random.randint(0, blank_indices.size)
    else:
        if remaining_blanks < 1:
            raise ValueError('remaining_blanks must be at least 1')
        num_to_fill = max(0, blank_indices.size - remaining_blanks)
    selected = np.random.choice(blank_indices, size=num_to_fill, replace=False)
    filled = board.copy().reshape(-1)
    filled[selected] = solution.reshape(-1)[selected]
    return filled.reshape(board.shape)

def collate_fn(batch: list[dict[str, str]], augment: bool, partial_fill_probability: float=0.0, partial_fill_remaining_blanks: int | None=None) -> tuple[Tensor, Tensor]:
    xs, ys = ([], [])
    for item in batch:
        board = np.frombuffer(item['question'].replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        solution = np.frombuffer(item['answer'].encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        if augment:
            board, solution = shuffle_sudoku(board, solution)
        if partial_fill_probability > 0 and np.random.rand() < partial_fill_probability:
            board = randomly_fill_solution_digits(board, solution, remaining_blanks=partial_fill_remaining_blanks)
        board = board.flatten().astype(np.int32)
        solution = solution.flatten().astype(np.int32)
        xs.append(np.pad(board, (1, 0)))
        ys.append(np.pad(solution, (1, 0)))
    return (torch.from_numpy(np.stack(xs, axis=0)), torch.from_numpy(np.stack(ys, axis=0)))
from hrm.tasks.sudoku import _worker_init_fn

def create_dataloader(split: str, batch_size: int, rank: int, world_size: int, dataset_name: str, augment: bool=False, repeat: int=1, seed: int=42, partial_fill_probability: float=0.0, partial_fill_remaining_blanks: int | None=None, partial_fill_on_eval: bool=False):
    is_train = split == 'train'
    if not 0.0 <= partial_fill_probability <= 1.0:
        raise ValueError('partial_fill_probability must be between 0 and 1')
    dataset_path = Path(dataset_name)
    if dataset_path.is_dir():
        dataset_source = 'csv'
        data_files = {'train': str(dataset_path / 'train.csv'), 'test_hard': str(dataset_path / 'test_hard.csv')}
    else:
        dataset_source, data_files = (dataset_name, None)
    dataset: Dataset = load_dataset(dataset_source, data_files=data_files, split=split, features=Features({'question': Value('string'), 'answer': Value('string')})).repeat(repeat if is_train else 1)
    return (DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, augment=augment and is_train, partial_fill_probability=partial_fill_probability if is_train or partial_fill_on_eval else 0.0, partial_fill_remaining_blanks=partial_fill_remaining_blanks), sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=is_train, drop_last=is_train, seed=seed), drop_last=is_train, pin_memory=True, persistent_workers=is_train, worker_init_fn=partial(_worker_init_fn, base_seed=seed), num_workers=1, prefetch_factor=2), {'vocab_size': 10, 'seq_len': 82, 'is_causal': False})
