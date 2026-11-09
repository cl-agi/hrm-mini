"""Score a predicted Maze board by the path it actually marks, without repair.

Token IDs follow ``~# SGo``. The primary metric accepts any shortest path and
ignores the leading BOS prediction. All actual board cells must be valid.
"""

from collections import deque
from functools import lru_cache
from math import isqrt

import numpy as np


METRIC_NAME = "shortest_path_accuracy"
METRIC_VERSION = "maze_shortest_path_v1"


def _neighbors(index, size):
    row, col = divmod(index, size)
    if row:
        yield index - size
    if row + 1 < size:
        yield index + size
    if col:
        yield index - 1
    if col + 1 < size:
        yield index + 1


@lru_cache(maxsize=4096)
def _input_details(board_bytes, size):
    """Cache only immutable validated problem boards, never model predictions."""
    board = np.frombuffer(board_bytes, dtype=np.uint8)
    start = int(np.flatnonzero(board == 3)[0])
    goal = int(np.flatnonzero(board == 4)[0])
    distances = [-1] * board.size
    distances[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == goal:
            return start, goal, distances[current]
        for neighbor in _neighbors(current, size):
            if board[neighbor] != 1 and distances[neighbor] < 0:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    raise ValueError("Input Maze has no S-to-G path")


def _chain_steps(answer, start, goal, size):
    """Return edge count for a single simple S-to-G chain, or -1 otherwise."""
    selected = (answer == 3) | (answer == 4) | (answer == 5)
    grid = selected.reshape(size, size)
    degree = np.zeros((size, size), dtype=np.uint8)
    degree[1:] += grid[:-1]
    degree[:-1] += grid[1:]
    degree[:, 1:] += grid[:, :-1]
    degree[:, :-1] += grid[:, 1:]
    required = np.full(selected.size, 2, dtype=np.uint8)
    required[start] = required[goal] = 1
    if not np.all(degree.ravel()[selected] == required[selected]):
        return -1
    # Degree constraints alone also permit disconnected cycles. Require every
    # marked cell to belong to the component of the actual start.
    seen = {start}
    queue = deque([start])
    while queue:
        for neighbor in _neighbors(queue.popleft(), size):
            if selected[neighbor] and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if goal not in seen or len(seen) != int(selected.sum()):
        return -1
    return len(seen) - 1


def _integer_array(name, values):
    array = np.asarray(values)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a two-dimensional integer token array")
    return array


def score_predictions(inputs, predictions, targets=None, size=None):
    """Return per-question NumPy arrays for ``(N, size**2 + 1)`` tokens.

    ``shortest_path_correct`` and ``valid_path_correct`` ignore predicted BOS;
    ``format_correct`` also requires BOS=0. Board walls and S/G must remain
    exactly as in the question, and only empty cells may be replaced with o.
    Validity requires all selected cells to form one unbranched simple chain.
    ``shortest_distance`` and ``predicted_steps`` count edges, not cells; the
    latter is -1 for invalid board predictions. Out-of-vocabulary board
    predictions are scored incorrect, never repaired or treated as input data.

    Input shape/dtype, invalid question tokens/BOS/endpoints, and unreachable
    questions raise ValueError. Optional targets must have matching shape and
    integer dtype; they are used only for full-sequence ``canonical_correct``
    and never consulted for path validity or shortest distance. With no targets,
    ``canonical_correct`` is None. An empty batch returns empty arrays.
    """
    inputs = _integer_array("inputs", inputs)
    predictions = _integer_array("predictions", predictions)
    if size is None:
        cell_count = inputs.shape[1] - 1
        if cell_count < 4:
            raise ValueError("inputs must encode a square Maze of size at least 2")
        size = isqrt(cell_count)
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 2:
        raise ValueError("size must be an integer of at least 2")
    if inputs.shape[1] != size * size + 1:
        raise ValueError("inputs must have shape (N, size**2 + 1)")
    if predictions.shape != inputs.shape:
        raise ValueError("predictions must have the same shape as inputs")
    if targets is not None:
        targets = _integer_array("targets", targets)
        if targets.shape != inputs.shape:
            raise ValueError("targets must have the same shape as inputs")

    board, answer = inputs[:, 1:], predictions[:, 1:]
    if np.any(inputs[:, 0] != 0) or np.any((board < 1) | (board > 4)):
        raise ValueError("Input Maze must have BOS 0 and only wall/empty/S/G board tokens")
    if np.any((board == 3).sum(axis=1) != 1) or np.any((board == 4).sum(axis=1) != 1):
        raise ValueError("Each input Maze must contain exactly one S and one G")
    board_format = np.where(
        board == 2, (answer == 2) | (answer == 5), answer == board
    ).all(axis=1)
    format_correct = board_format & (predictions[:, 0] == 0)
    distances = np.empty(inputs.shape[0], dtype=np.int32)
    predicted_steps = np.full(inputs.shape[0], -1, dtype=np.int32)
    for index, question in enumerate(board):
        start, goal, distance = _input_details(question.astype(np.uint8).tobytes(), int(size))
        distances[index] = distance
        if board_format[index]:
            predicted_steps[index] = _chain_steps(answer[index], start, goal, int(size))
    valid = predicted_steps >= 0
    return {
        "shortest_path_correct": valid & (predicted_steps == distances),
        "valid_path_correct": valid,
        "canonical_correct": None if targets is None else (predictions == targets).all(axis=1),
        "format_correct": format_correct,
        "shortest_distance": distances,
        "predicted_steps": predicted_steps,
    }
