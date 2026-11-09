from hrm.tasks.io import load_dataset
from hrm.paths import repo_path, work_path
from pathlib import Path
import torch
from torch import Tensor
from torch.nn import functional as F
from datasets import load_from_disk
import math
from functools import partial
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os

def pad_clause(clauses_in: list, M: int) -> list:
    assert len(clauses_in) <= M, f'M={M} is not large enough for {len(clauses_in)} clauses.'
    clauses_in = clauses_in * math.ceil(M / len(clauses_in))
    clauses_in = clauses_in[:M]
    return clauses_in

def expand_not(variable: Tensor) -> Tensor:
    return torch.stack([variable, ~variable], dim=-1).flatten(-2)

def get_clauses_state(clause: Tensor, answer: Tensor) -> Tensor:
    pos_values = torch.gather(answer.bool(), -1, clause.abs().flatten(1))
    values = torch.where(clause.flatten(-2) > 0, pos_values, ~pos_values).view(clause.shape)
    return values.any(dim=-1)

def is_valid_3sat(clauses: Tensor, answer: Tensor) -> Tensor:
    return get_clauses_state(clauses, answer).all(dim=-1)

def augmentation(clauses: Tensor, answer: Tensor, N: int, M: int) -> tuple[Tensor, Tensor]:
    B, _, K = clauses.shape
    device = clauses.device
    clause_perm = torch.argsort(torch.rand(B, M, device=device), dim=1)
    clauses = clauses.gather(1, clause_perm.unsqueeze(-1).expand(B, M, K))
    return (clauses, answer)

def build_attn_mask(clauses: Tensor, N: int, M: int) -> Tensor:
    B, _, K = clauses.shape
    clauses_x = torch.where(clauses > 0, clauses * 2, -clauses * 2 - 1).flatten(1)
    clauses_y = torch.arange(M).repeat_interleave(K).expand(B, -1) + 2 * N + 1
    seq_len = N * 2 + M + 1
    attn_mask = torch.zeros(B, seq_len * seq_len)
    attn_mask.scatter_(1, clauses_x * seq_len + clauses_y, 1).scatter_(1, clauses_y * seq_len + clauses_x, 1)
    attn_mask = attn_mask.view(B, seq_len, seq_len)
    digonal_ind = torch.arange(N * 2 + 1)
    attn_mask[..., digonal_ind, digonal_ind] = 1
    digonal_ind_2 = torch.arange(N) * 2 + 2
    attn_mask[..., digonal_ind_2, digonal_ind_2 - 1] = 1
    attn_mask[..., digonal_ind_2 - 1, digonal_ind_2] = 1
    attn_mask[..., 0] = 1
    attn_mask[..., 0, :] = 1
    return attn_mask.view(-1, 1, seq_len, seq_len).bool()

def collate_fn(batch: list[dict], N: int, M: int, puzzle_type: str, augment: bool=False) -> dict[str, Tensor]:
    assert puzzle_type == 'sat_unsat' or puzzle_type == 'exact_solution', 'puzzle_type must be sat_unsat for classification or exact_solution for assignments'
    with torch.no_grad():
        BS = len(batch)
        clause, answer, answer_mask = ([], [], [])
        for item in batch:
            clause.append(pad_clause(item['clauses'], M))
            if item['assignment'] is None:
                answer.append([1] * (N + 1))
                answer_mask.append([1] + [0] * (2 * N + M))
            else:
                answer.append([0] + item['assignment'])
                answer_mask.append([1] * (2 * N + 1) + [0] * M)
        clause = torch.tensor(clause)
        answer = torch.tensor(answer)
        answer_mask = torch.tensor(answer_mask)
        if augment:
            clause, answer = augmentation(clause, answer, N=N, M=M)
        if puzzle_type != 'sat_unsat':
            assert is_valid_3sat(clause, answer).all(), 'Wrong data'
        x_var = F.pad(torch.randint(0, 2, size=(BS, N)), (1, 0), mode='constant', value=0).bool()
        x = torch.cat([torch.zeros(BS, 1), expand_not(x_var[:, 1:].bool()).int() + 1, torch.ones(BS, M).int() + 2], dim=1)
        y = torch.cat([torch.zeros(BS, 1), expand_not(answer[:, 1:].bool()).int() + 1, torch.ones(BS, M).int() + 2], dim=1)
        return {'x': x.long(), 'y': y.long(), 'y_mask': answer_mask, 'attn_mask': build_attn_mask(clause, N=N, M=M), 'clauses': clause}

def create_dataloader(split: str, batch_size: int, rank: int, world_size: int, N: int, M: int, type: str, dataset_name: str=str(repo_path('data/sat')), cache_dir: str='/dev/shm/datasets', augment: bool=False, repeat: int=1, num_workers: int=1):
    is_train = 'train' in split
    if os.path.exists(dataset_name):
        dataset: Dataset = load_from_disk(dataset_name, keep_in_memory=True)[split].repeat(repeat if is_train else 1)
    else:
        dataset: Dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir).repeat(repeat if is_train else 1)
    return (DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=partial(collate_fn, augment=augment and is_train, N=N, M=M, puzzle_type=type), sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=is_train, drop_last=True), drop_last=True, pin_memory=True, persistent_workers=True, num_workers=num_workers, prefetch_factor=2), {'vocab_size': 4, 'seq_len': 2 * N + M + 1, 'is_causal': False, 'enable_pe': True})
