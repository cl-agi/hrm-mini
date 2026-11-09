"""Differentiable constraint errors for 9x9 Sudoku boards."""

from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F


Reduction = Literal["none", "sum", "mean"]


def _sudoku_peer_mask(device: torch.device) -> Tensor:
    """Return the 81x81 mask of distinct cells sharing a row, column, or box."""
    positions = torch.arange(81, device=device)
    rows = positions // 9
    columns = positions % 9
    boxes = (rows // 3) * 3 + columns // 3

    peers = (
        (rows[:, None] == rows[None, :])
        | (columns[:, None] == columns[None, :])
        | (boxes[:, None] == boxes[None, :])
    )
    return peers & (positions[:, None] != positions[None, :])


def sudoku_soft_error(
    logits: Tensor,
    *,
    from_logits: bool = True,
    temperature: float = 1.0,
    reduction: Reduction = "none",
) -> Tensor:
    """Estimate conflicts under independent per-cell digit distributions.

    Args:
        logits: Tensor shaped ``(..., 81, 9)``. By default these are raw
            logits. Set ``from_logits=False`` to pass probabilities directly.
        from_logits: Apply a softmax over the nine digits when true.
        temperature: Softmax temperature. Only used for raw logits.
        reduction: ``"none"`` returns one error per cell. ``"sum"`` returns
            the expected number of unique conflicting pairs per board, while
            ``"mean"`` returns the mean number of conflicts incident to a cell.

    For a one-hot probability board, ``reduction="none"`` is exactly the
    number of distinct conflicting peers of every cell. A peer shared through
    both a row/column and a box is counted only once. For soft probabilities,
    the dot product of two cells is the probability that their digits agree.
    """
    if logits.ndim < 2 or logits.shape[-2:] != (81, 9):
        raise ValueError(
            f"expected input shape (..., 81, 9), got {tuple(logits.shape)}"
        )
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"unsupported reduction: {reduction!r}")
    if from_logits:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        probabilities = F.softmax(logits.float() / temperature, dim=-1)
    else:
        if not logits.is_floating_point():
            probabilities = logits.float()
        else:
            probabilities = logits

    # P(two cells conflict) = sum_d P(cell_i=d) P(cell_j=d).
    agreement = probabilities @ probabilities.transpose(-1, -2)
    peer_mask = _sudoku_peer_mask(logits.device)
    per_cell = (agreement * peer_mask).sum(dim=-1)

    if reduction == "none":
        return per_cell
    if reduction == "mean":
        return per_cell.mean(dim=-1)
    # Every conflicting pair contributes to both of its endpoint cells.
    return per_cell.sum(dim=-1) / 2
