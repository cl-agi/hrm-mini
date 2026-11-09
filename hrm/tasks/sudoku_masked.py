from hrm.tasks.io import load_dataset
from functools import partial
import numpy as np
import torch
from torch import Tensor
from datasets import Features, Value
from torch.utils.data import Dataset, DataLoader, DistributedSampler

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = np.random.rand() < 0.5
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        return digit_map[new_board]
    return (apply_transformation(board), apply_transformation(solution))

def collate_fn(batch: list[dict[str, str]], augment: bool) -> dict[str, Tensor]:
    xs, ys = ([], [])
    for item in batch:
        board = np.frombuffer(item['question'].replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        solution = np.frombuffer(item['answer'].encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        if augment:
            board, solution = shuffle_sudoku(board, solution)
        board = board.flatten().astype(np.int32)
        solution = solution.flatten().astype(np.int32)
        xs.append(np.pad(board, (1, 0)))
        ys.append(np.pad(solution, (1, 0)))
    return {'x': torch.from_numpy(np.stack(xs, axis=0)), 'y': torch.from_numpy(np.stack(ys, axis=0))}

def create_dataloader(split: str, batch_size: int, rank: int, world_size: int, dataset_name: str, augment: bool=False, repeat: int=1):
    is_train = split == 'train'
    dataset_split = 'test_hard' if split == 'test' else split
    dataset: Dataset = load_dataset(dataset_name, split=dataset_split, features=Features({'question': Value('string'), 'answer': Value('string')})).repeat(repeat if is_train else 1)
    return (DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, augment=augment and is_train), sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=is_train, drop_last=True), drop_last=True, pin_memory=True, persistent_workers=True, num_workers=1, prefetch_factor=2), {'vocab_size': 10, 'seq_len': 82, 'is_causal': False, 'enable_pe': True})
