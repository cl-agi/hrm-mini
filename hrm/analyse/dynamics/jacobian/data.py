"""Sudoku Extreme CSV loading with deterministic board selection."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SudokuRecord:
    board_id: int
    question: str
    answer: str


def resolve_test_csv(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = (path / "test_hard.csv", path / "test.csv")
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not path.is_file():
        raise ValueError(
            f"Sudoku Extreme test CSV does not exist: {path} "
            "(expected test_hard.csv or test.csv)"
        )
    return path


def read_test_records(path: Path) -> list[SudokuRecord]:
    csv_path = resolve_test_csv(path)
    records: list[SudokuRecord] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"question", "answer"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain question and answer columns")
        for board_id, row in enumerate(reader):
            question = str(row["question"]).replace("0", ".")
            answer = str(row["answer"])
            if len(question) != 81 or any(char not in ".123456789" for char in question):
                raise ValueError(f"Invalid question at CSV row {board_id + 2}")
            if len(answer) != 81 or any(char not in "123456789" for char in answer):
                raise ValueError(f"Invalid answer at CSV row {board_id + 2}")
            records.append(SudokuRecord(board_id, question, answer))
    if not records:
        raise ValueError("Test CSV is empty")
    return records


def select_board_ids(
    population_size: int,
    *,
    boards: int,
    offset: int,
    selection: str,
    selection_seed: int,
    explicit: Sequence[int] | None,
) -> tuple[int, ...]:
    if explicit is not None:
        selected = tuple(int(value) for value in explicit)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("Explicit board IDs must be non-empty and unique")
    elif selection == "all":
        selected = tuple(range(population_size))
    elif selection == "sequential":
        selected = tuple(range(offset, min(offset + boards, population_size)))
        if len(selected) != boards:
            raise ValueError("Requested sequential range exceeds the test set")
    elif selection == "random":
        if boards > population_size:
            raise ValueError("Requested more random boards than the test population")
        generator = np.random.default_rng(selection_seed)
        permutation = generator.permutation(population_size)
        selected = tuple(int(value) for value in permutation[offset : offset + boards])
        if len(selected) != boards:
            raise ValueError("Random offset plus board count exceeds the test population")
    else:
        raise ValueError(f"Unknown selection mode: {selection}")
    if min(selected) < 0 or max(selected) >= population_size:
        raise ValueError(f"Board IDs must lie in [0, {population_size})")
    return selected


class SelectedSudokuDataset(Dataset[tuple[Tensor, Tensor, int]]):
    def __init__(self, records: Sequence[SudokuRecord], board_ids: Sequence[int]) -> None:
        self.records = records
        self.board_ids = tuple(board_ids)

    def __len__(self) -> int:
        return len(self.board_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        record = self.records[self.board_ids[index]]
        question = [0] + [0 if char == "." else int(char) for char in record.question]
        answer = [0] + [int(char) for char in record.answer]
        return (
            torch.tensor(question, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            record.board_id,
        )


def build_loader(
    records: Sequence[SudokuRecord], board_ids: Sequence[int], batch_size: int
) -> DataLoader:
    return DataLoader(
        SelectedSudokuDataset(records, board_ids),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def board_ids_sha256(board_ids: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(board_ids, dtype=np.int64).tobytes()).hexdigest()


def dataset_sha256(records: Sequence[SudokuRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.question.encode("ascii"))
        digest.update(b"\0")
        digest.update(record.answer.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
