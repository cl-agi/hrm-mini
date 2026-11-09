from hrm.tasks.io import load_dataset
import numpy as np
import torch
from torch import Tensor
from datasets import Features, Value
from torch.utils.data import DataLoader
from hrm.tasks.sampling_stable16 import InfiniteSampler
VOCAB = '~# SGo'
CHAR_TO_ID = {char: idx for idx, char in enumerate(VOCAB)}

def collate_fn(batch: list[dict[str, str]]) -> tuple[Tensor, Tensor]:
    xs, ys = ([], [])
    for item in batch:
        board_ids = [CHAR_TO_ID[char] for char in '~' + item['question']]
        solution_ids = [CHAR_TO_ID[char] for char in '~' + item['answer']]
        board = np.array(board_ids, dtype=np.int32)
        solution = np.array(solution_ids, dtype=np.int32)
        xs.append(board)
        ys.append(solution)
    return (torch.from_numpy(np.stack(xs, axis=0)), torch.from_numpy(np.stack(ys, axis=0)))

def create_dataloader(split: str, batch_size: int, rank: int, world_size: int, dataset_name: str, seed: int=42, train_size: int | None=None, test_size: int | None=None, subsample_seed: int=42, training: bool=False):
    is_train = split == 'train'
    dataset = load_dataset(dataset_name, split=split, features=Features({'question': Value('string'), 'answer': Value('string')}))
    size = train_size if is_train else test_size
    if size is not None:
        if not 1 <= size <= len(dataset):
            raise ValueError(f'{split} size must be between 1 and {len(dataset)}, got {size}')
        dataset = dataset.shuffle(seed=subsample_seed).select(range(size))
    return (DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=InfiniteSampler(len(dataset), rank, world_size, seed) if training else range(rank, len(dataset), world_size), drop_last=False, pin_memory=True, persistent_workers=True, num_workers=1, prefetch_factor=2), {'vocab_size': len(VOCAB), 'seq_len': 901, 'is_causal': False})
