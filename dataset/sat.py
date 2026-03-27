import math
from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset, load_dataset


def pad_clause(clauses_in: list[list[int]], max_clauses: int) -> list[list[int]]:
    assert len(clauses_in) <= max_clauses, f"M={max_clauses} is not large enough for {len(clauses_in)} clauses."
    clauses_in *= math.ceil(max_clauses / len(clauses_in))
    return clauses_in[:max_clauses]


def expand_not(variable: Tensor) -> Tensor:
    return torch.stack([variable, ~variable], dim=-1).flatten(-2)


def get_clauses_state(clause: Tensor, answer: Tensor) -> Tensor:
    pos_values = torch.gather(answer.bool(), -1, clause.abs().flatten(1))
    values = torch.where(clause.flatten(-2) > 0, pos_values, ~pos_values).view(clause.shape)
    return values.any(dim=-1)


def is_valid_3sat(clauses: Tensor, answer: Tensor) -> Tensor:
    return get_clauses_state(clauses, answer).all(dim=-1)


def augmentation(clauses: Tensor, answer: Tensor, num_variables: int, max_clauses: int) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("Augmentation for 3-SAT is not implemented.")


def build_attn_mask(clauses: Tensor, num_variables: int, max_clauses: int) -> Tensor:
    batch_size, _, clause_width = clauses.shape

    clauses_x = torch.where(clauses > 0, clauses * 2, -clauses * 2 - 1).flatten(1)
    clauses_y = torch.arange(max_clauses).repeat_interleave(clause_width).expand(batch_size, -1) + 2 * num_variables + 1
    seq_len = num_variables * 2 + max_clauses + 1

    attn_mask = torch.zeros(batch_size, seq_len * seq_len, dtype=torch.bool)
    attn_mask.scatter_(1, clauses_x * seq_len + clauses_y, True)
    attn_mask.scatter_(1, clauses_y * seq_len + clauses_x, True)
    attn_mask = attn_mask.view(batch_size, seq_len, seq_len)

    diagonal = torch.arange(num_variables * 2 + 1)
    attn_mask[..., diagonal, diagonal] = True

    negation_links = torch.arange(num_variables) * 2 + 2
    attn_mask[..., negation_links, negation_links - 1] = True
    attn_mask[..., negation_links - 1, negation_links] = True

    attn_mask[..., 2 * num_variables + 1 :, 2 * num_variables + 1 :] = True
    attn_mask[..., 0] = True
    attn_mask[..., 0, :] = True

    return attn_mask.view(-1, 1, seq_len, seq_len)


def collate_fn(
    batch: list[dict],
    num_variables: int,
    max_clauses: int,
    augment: bool = False,
) -> dict[str, Tensor]:
    batch_size = len(batch)

    clauses = [pad_clause(item["clauses"], max_clauses) for item in batch]
    answers = [[0] + item["assignment"] for item in batch]
    answer_masks = [[1] * (2 * num_variables + 1) + [0] * max_clauses for _ in batch]

    clauses = torch.tensor(clauses)
    answers = torch.tensor(answers)
    answer_masks = torch.tensor(answer_masks)

    if augment:
        clauses, answers = augmentation(clauses, answers, num_variables=num_variables, max_clauses=max_clauses)

    assert is_valid_3sat(clauses, answers).all(), "Wrong data"

    x_var = F.pad(torch.randint(0, 2, size=(batch_size, num_variables)), (1, 0), value=0).bool()
    x = torch.cat(
        [
            torch.zeros(batch_size, 1),
            expand_not(x_var[:, 1:]).int() + 1,
            get_clauses_state(clauses, x_var).int() + 1,
        ],
        dim=1,
    )
    y = torch.cat(
        [
            torch.zeros(batch_size, 1),
            expand_not(answers[:, 1:].bool()).int() + 1,
            torch.ones(batch_size, max_clauses).int() + 1,
        ],
        dim=1,
    )

    return {
        "x": x.long(),
        "y": y.long(),
        "y_mask": answer_masks,
        "attn_mask": build_attn_mask(clauses, num_variables=num_variables, max_clauses=max_clauses),
    }


def create_dataloader(
    split: str,
    batch_size: int,
    rank: int,
    world_size: int,
    dataset_name: str,
    num_variables: int,
    max_clauses: int,
    augment: bool = False,
    repeat: int = 1,
    num_workers: int = 1,
    seed: int = 42,
):
    is_train = "train" in split

    dataset: Dataset = load_dataset(dataset_name)[split].repeat(repeat if is_train else 1)  # type: ignore[index]

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=partial(
            collate_fn,
            augment=augment and is_train,
            num_variables=num_variables,
            max_clauses=max_clauses,
        ),
        sampler=DistributedSampler(
            dataset,
            rank=rank,
            num_replicas=world_size,
            shuffle=is_train,
            drop_last=True,
            seed=seed,
        ),
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    ), {
        "vocab_size": 3,
        "seq_len": 2 * num_variables + max_clauses + 1,
        "is_causal": False,
    }
