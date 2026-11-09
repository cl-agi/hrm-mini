from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from datasets import Features, Value
from hrm.tasks.io import load_dataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from hrm.training.checkpointing import capture_rng, restore_rng, ResumeSampler

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)

def randomly_fill_solution_digits(
    board: np.ndarray,
    solution: np.ndarray,
    remaining_blanks: int | None = None,
) -> np.ndarray:
    """Reveal solution digits while preserving at least one original blank."""
    blank_indices = np.flatnonzero(board.reshape(-1) == 0)
    if blank_indices.size <= 1:
        return board

    if remaining_blanks is None:
        num_to_fill = np.random.randint(0, blank_indices.size)
    else:
        if remaining_blanks < 1:
            raise ValueError("remaining_blanks must be at least 1")
        num_to_fill = max(0, blank_indices.size - remaining_blanks)

    selected = np.random.choice(blank_indices, size=num_to_fill, replace=False)
    filled = board.copy().reshape(-1)
    filled[selected] = solution.reshape(-1)[selected]
    return filled.reshape(board.shape)


def collate_fn(
    batch: list[dict[str, str]],
    augment: bool,
    partial_fill_probability: float = 0.0,
    partial_fill_remaining_blanks: int | None = None,
    return_rng_state: bool = False,
):
    xs, ys = [], []
    for item in batch:
        board = np.frombuffer(item["question"].replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        solution = np.frombuffer(item["answer"].encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
        if augment:
            board, solution = shuffle_sudoku(board, solution)
        if partial_fill_probability > 0 and np.random.rand() < partial_fill_probability:
            board = randomly_fill_solution_digits(
                board,
                solution,
                remaining_blanks=partial_fill_remaining_blanks,
            )

        # Convert and flatten
        board = board.flatten().astype(np.int32)
        solution = solution.flatten().astype(np.int32)
        # Pad a BOS token
        xs.append(np.pad(board, (1, 0)))
        ys.append(np.pad(solution, (1, 0)))

    batch = (torch.from_numpy(np.stack(xs, axis=0)), torch.from_numpy(np.stack(ys, axis=0)))
    return (*batch, capture_rng(include_cuda=False)) if return_rng_state else batch

def _worker_init_fn(worker_id: int, base_seed: int, resume_rng=None):
    if resume_rng is None:
        np.random.seed(base_seed + worker_id)
    else:
        restore_rng(resume_rng)

def create_dataloader(
    split: str,
    batch_size: int,
    rank: int,
    world_size: int,
    dataset_name: str,
    augment: bool = False,
    repeat: int = 1,
    seed: int = 42,
    partial_fill_probability: float = 0.0,
    partial_fill_remaining_blanks: int | None = None,
    partial_fill_on_eval: bool = False,
    resume_worker_state=None,
    skip_batches: int = 0,
):
    is_train = split == "train"
    if not 0.0 <= partial_fill_probability <= 1.0:
        raise ValueError("partial_fill_probability must be between 0 and 1")
    dataset_path = Path(dataset_name)
    if dataset_path.is_dir():
        dataset_source = "csv"
        data_files = {
            "train": str(dataset_path / "train.csv"),
            "test_hard": str(dataset_path / "test_hard.csv"),
        }
    else:
        dataset_source, data_files = dataset_name, None
    dataset: Dataset = load_dataset(dataset_source, data_files=data_files, split=split, features = Features({
        "question": Value("string"),
        "answer": Value("string"),
    })).repeat(repeat if is_train else 1)  # pyright: ignore[reportAssignmentType]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(
            collate_fn,
            augment=augment and is_train,
            return_rng_state=is_train,
            partial_fill_probability=(
                partial_fill_probability if is_train or partial_fill_on_eval else 0.0
            ),
            partial_fill_remaining_blanks=partial_fill_remaining_blanks,
        ),

        sampler=ResumeSampler(DistributedSampler(dataset,
                                   rank=rank, num_replicas=world_size,
                                   shuffle=is_train, drop_last=is_train,
                                   seed=seed), batch_size, skip_batches) if is_train else list(range(rank, len(dataset), world_size)),
        drop_last=is_train,

        pin_memory=True,
        persistent_workers=is_train,
        worker_init_fn=partial(_worker_init_fn, base_seed=seed, resume_rng=resume_worker_state),

        num_workers=1,
        prefetch_factor=2
    ), {
        # Dataset metadata
        "vocab_size": 10,
        "seq_len": 82,
        "is_causal": False
    }
