import random
from functools import partial

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class KQueenDataset(Dataset):
    def __init__(
        self,
        split: str,
        k: int,
        size: int,
        min_clues: int,
        max_clues: int,
        seed: int,
        mark_attacks: bool = False,
    ) -> None:
        if k == 2 or k == 3:
            raise ValueError("k-queen has no solution for k=2 or k=3.")
        if min_clues < 0 or max_clues > k or min_clues > max_clues:
            raise ValueError("Require 0 <= min_clues <= max_clues <= k.")
        self.split = split
        self.k = k
        self.size = size
        self.min_clues = min_clues
        self.max_clues = max_clues
        self.seed = seed + (0 if split == "train" else 1_000_000)
        self.mark_attacks = mark_attacks

    def __len__(self) -> int:
        return self.size

    def _solve(self, rng: random.Random) -> list[int]:
        cols = [-1] * self.k
        used_cols: set[int] = set()
        used_diag_a: set[int] = set()
        used_diag_b: set[int] = set()

        def backtrack(row: int) -> bool:
            if row == self.k:
                return True
            candidates = list(range(self.k))
            rng.shuffle(candidates)
            for col in candidates:
                diag_a = row - col
                diag_b = row + col
                if col in used_cols or diag_a in used_diag_a or diag_b in used_diag_b:
                    continue
                cols[row] = col
                used_cols.add(col)
                used_diag_a.add(diag_a)
                used_diag_b.add(diag_b)
                if backtrack(row + 1):
                    return True
                used_cols.remove(col)
                used_diag_a.remove(diag_a)
                used_diag_b.remove(diag_b)
                cols[row] = -1
            return False

        if not backtrack(0):
            raise RuntimeError(f"Failed to generate a {self.k}-queen solution.")
        return cols

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = random.Random(self.seed + index)
        cols = self._solve(rng)
        solution = torch.ones(self.k * self.k, dtype=torch.long)
        for row, col in enumerate(cols):
            solution[row * self.k + col] = 2

        board = torch.zeros(self.k * self.k, dtype=torch.long)
        clue_count = rng.randint(self.min_clues, self.max_clues)
        clue_rows = rng.sample(range(self.k), clue_count)
        if self.mark_attacks:
            attacked = torch.zeros(self.k, self.k, dtype=torch.bool)
            for row in clue_rows:
                col = cols[row]
                attacked[row, :] = True
                attacked[:, col] = True
                for r in range(self.k):
                    dc = abs(r - row)
                    if col - dc >= 0:
                        attacked[r, col - dc] = True
                    if col + dc < self.k:
                        attacked[r, col + dc] = True
            board[attacked.flatten()] = 1
        for row in clue_rows:
            board[row * self.k + cols[row]] = 2

        return {"board": board, "solution": solution}


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    boards = torch.stack([item["board"] for item in batch], dim=0)
    solutions = torch.stack([item["solution"] for item in batch], dim=0)
    bos = torch.zeros(boards.shape[0], 1, dtype=torch.long)
    x = torch.cat([bos, boards], dim=1)
    y = torch.cat([bos, solutions], dim=1)
    y_mask = (x == 0).long()
    y_mask[:, 0] = 0
    return {"x": x, "y": y, "y_mask": y_mask}



def solve_k_queen(k: int, fixed_cols: dict[int, int], col_order: list[list[int]] | None = None) -> list[int] | None:
    cols = [-1] * k
    used_cols: set[int] = set()
    used_diag_a: set[int] = set()
    used_diag_b: set[int] = set()

    for row, col in fixed_cols.items():
        if row < 0 or row >= k or col < 0 or col >= k:
            return None
        diag_a = row - col
        diag_b = row + col
        if col in used_cols or diag_a in used_diag_a or diag_b in used_diag_b:
            return None
        cols[row] = col
        used_cols.add(col)
        used_diag_a.add(diag_a)
        used_diag_b.add(diag_b)

    row_order = sorted(range(k), key=lambda row: 0 if cols[row] != -1 else 1)

    def backtrack(order_index: int) -> bool:
        if order_index == k:
            return True
        row = row_order[order_index]
        if cols[row] != -1:
            return backtrack(order_index + 1)
        candidates = col_order[row] if col_order is not None else list(range(k))
        for col in candidates:
            diag_a = row - col
            diag_b = row + col
            if col in used_cols or diag_a in used_diag_a or diag_b in used_diag_b:
                continue
            cols[row] = col
            used_cols.add(col)
            used_diag_a.add(diag_a)
            used_diag_b.add(diag_b)
            if backtrack(order_index + 1):
                return True
            used_cols.remove(col)
            used_diag_a.remove(diag_a)
            used_diag_b.remove(diag_b)
            cols[row] = -1
        return False

    return cols if backtrack(0) else None


def verify_k_queen_tokens(tokens: Tensor, x: Tensor | None = None) -> Tensor:
    """Return a per-sample validity mask for tokenized k-queen boards."""
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
        if x is not None and x.dim() == 1:
            x = x.unsqueeze(0)
    if tokens.dim() != 2:
        raise ValueError("k_queen tokens must have shape [batch, k*k + 1].")
    board_len = tokens.shape[1] - 1
    k = int(board_len ** 0.5)
    if k * k != board_len:
        raise ValueError("k_queen sequence length must be k*k + 1.")
    if x is not None and x.shape != tokens.shape:
        raise ValueError("k_queen clue tokens must have the same shape as predictions.")

    board = tokens[:, 1:].view(tokens.shape[0], k, k)
    queens = board == 2

    valid = tokens[:, 0] == 0
    valid &= ((board == 1) | queens).flatten(1).all(dim=1)
    valid &= queens.sum(dim=2).eq(1).all(dim=1)
    valid &= queens.sum(dim=1).eq(1).all(dim=1)

    diag_count = 2 * k - 1
    rows = torch.arange(k, device=tokens.device).view(k, 1)
    cols = torch.arange(k, device=tokens.device).view(1, k)
    diag_a = (rows - cols + k - 1).flatten().expand(tokens.shape[0], -1)
    diag_b = (rows + cols).flatten().expand(tokens.shape[0], -1)
    queen_values = queens.flatten(1).long()
    diag_a_counts = torch.zeros(tokens.shape[0], diag_count, dtype=torch.long, device=tokens.device)
    diag_b_counts = torch.zeros_like(diag_a_counts)
    diag_a_counts.scatter_add_(1, diag_a, queen_values)
    diag_b_counts.scatter_add_(1, diag_b, queen_values)
    valid &= diag_a_counts.le(1).all(dim=1)
    valid &= diag_b_counts.le(1).all(dim=1)

    if x is not None:
        clue_mask = x[:, 1:] == 2
        valid &= (queens.flatten(1) | ~clue_mask).all(dim=1)

    return valid


validate_k_queen_tokens = verify_k_queen_tokens


def complete_k_queen_tokens(x: Tensor, logits: Tensor | None = None) -> Tensor:
    boards = x[:, 1:].long()
    k = int(boards.shape[1] ** 0.5)
    if k * k != boards.shape[1]:
        raise ValueError("k_queen sequence length must be k*k + 1.")
    completed = torch.ones_like(boards)
    for batch_index in range(boards.shape[0]):
        clue_positions = torch.nonzero(boards[batch_index].view(k, k) == 2, as_tuple=False)
        fixed_cols = {int(row): int(col) for row, col in clue_positions.tolist()}
        col_order = None
        if logits is not None:
            queen_scores = logits[batch_index, 1:, 2].view(k, k)
            col_order = [torch.argsort(queen_scores[row], descending=True).tolist() for row in range(k)]
        solution = solve_k_queen(k, fixed_cols, col_order=col_order)
        if solution is None:
            completed[batch_index] = boards[batch_index]
            continue
        completed[batch_index].fill_(1)
        for row, col in enumerate(solution):
            completed[batch_index, row * k + col] = 2
    bos = torch.zeros(x.shape[0], 1, dtype=torch.long, device=x.device)
    return torch.cat([bos, completed], dim=1)


def enumerate_k_queen_solutions(k: int) -> list[list[int]]:
    solutions: list[list[int]] = []

    def collect(fixed_cols: dict[int, int], start_col_order: list[list[int]] | None = None) -> None:
        cols = [-1] * k
        used_cols: set[int] = set()
        used_diag_a: set[int] = set()
        used_diag_b: set[int] = set()

        def backtrack(row: int) -> None:
            if row == k:
                solutions.append(cols.copy())
                return
            for col in range(k):
                diag_a = row - col
                diag_b = row + col
                if col in used_cols or diag_a in used_diag_a or diag_b in used_diag_b:
                    continue
                cols[row] = col
                used_cols.add(col)
                used_diag_a.add(diag_a)
                used_diag_b.add(diag_b)
                backtrack(row + 1)
                used_cols.remove(col)
                used_diag_a.remove(diag_a)
                used_diag_b.remove(diag_b)
                cols[row] = -1

        backtrack(0)

    collect({})
    return solutions

def create_dataloader(
    split: str,
    batch_size: int,
    rank: int,
    world_size: int,
    k: int = 4,
    size: int = 4096,
    test_size: int = 512,
    min_clues: int = 1,
    max_clues: int = 2,
    seed: int = 42,
    num_workers: int = 0,
    mark_attacks: bool = False,
):
    dataset = KQueenDataset(
        split=split,
        k=k,
        size=size if split == "train" else test_size,
        min_clues=min_clues,
        max_clues=max_clues,
        seed=seed,
        mark_attacks=mark_attacks,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn),
        sampler=DistributedSampler(
            dataset,
            rank=rank,
            num_replicas=world_size,
            shuffle=split == "train",
            drop_last=True,
        ),
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        num_workers=num_workers,
    ), {
        "vocab_size": 3,
        "seq_len": k * k + 1,
        "is_causal": False,
        "enable_pe": True,
    }
