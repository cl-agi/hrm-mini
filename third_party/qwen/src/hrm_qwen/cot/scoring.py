"""Recovered teacher-CoT grading and hint curriculum; no model imports."""
from __future__ import annotations
import re
import numpy as np
from hrm_qwen.cot.prompt import make_prompt
from hrm_qwen.sft.data import read_split
from hrm.tasks.sudoku import shuffle_sudoku
THINK_END = "</think>"
def is_legal_solution(grids: np.ndarray) -> np.ndarray:
    """`[n]` bool: does each `[n, 81]` grid satisfy the row, column and box constraints?"""
    g = grids.reshape(-1, 9, 9)
    boxes = g.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    want = np.arange(1, 10)
    return (
        (np.sort(g, axis=2) == want).all(axis=(1, 2))
        & (np.sort(g, axis=1).transpose(0, 2, 1) == want).all(axis=(1, 2))
        & (np.sort(boxes, axis=2) == want).all(axis=(1, 2))
    )

def parse_grid(text: str, reasoning: bool = False) -> np.ndarray | None:
    """Pull an 81-cell grid out of free-form model output, or return None.

    `reasoning=True` means the chat template opened a `<think>` block, so a completion without a
    closing `</think>` is a model that ran out of budget mid-thought and never gave an answer.
    Scraping digits out of unfinished reasoning would score it as a confident wrong answer, which
    is worse than useless -- it looks like a real 0% instead of a truncated run.
    """
    if reasoning and "</think>" not in text:
        return None
    tail = text.rsplit("</think>", 1)[-1]  # ignore the reasoning block if there is one

    # 81 consecutive digits, tolerating whitespace the model may have inserted
    runs = re.findall(r"\d{81}", re.sub(r"\s+", "", tail))
    if runs:
        return np.array([int(c) for c in runs[-1]], dtype=np.int64)

    # or nine lines of nine digits, which is how models usually like to print a grid
    rows = [re.sub(r"\D", "", line) for line in tail.splitlines()]
    rows = [r for r in rows if len(r) == 9]
    if len(rows) >= 9:
        return np.array([int(c) for r in rows[-9:] for c in r], dtype=np.int64)

    return None

def augmented_rows(rows: list[tuple[str, str]], indices: range, seed: int, augment: bool = True):
    """The same (seed, index)-addressed augmentation as `llm_sft.data.SudokuSFTDataset`."""
    out = []
    for idx in indices:
        question, answer = rows[idx % len(rows)]
        if augment:
            np.random.seed((seed * 1_000_003 + idx) % (2**31 - 1))
            board = np.frombuffer(question.encode(), dtype=np.uint8).reshape(9, 9) - ord("0")
            solution = np.frombuffer(answer.encode(), dtype=np.uint8).reshape(9, 9) - ord("0")
            board, solution = shuffle_sudoku(board, solution)
            question = "".join(map(str, board.flatten().tolist()))
            answer = "".join(map(str, solution.flatten().tolist()))
        out.append((idx, question, answer))
    return out

def add_hints(question: str, answer: str, n_hints: int, rng: np.random.Generator) -> str:
    """Reveal `n_hints` empty cells of `question` from `answer` (fewer if the puzzle has fewer)."""
    if n_hints <= 0:
        return question
    empty = [i for i, c in enumerate(question) if c == "0"]
    reveal = rng.choice(empty, size=min(n_hints, len(empty)), replace=False)
    q = list(question)
    for i in reveal:
        q[i] = answer[i]
    return "".join(q)

def hinted_puzzle(question: str, answer: str, n_hints: int, seed: int, index: int) -> str:
    """`add_hints` addressed by (seed, index, n_hints) so every rank builds the same prompt."""
    rng = np.random.default_rng([seed, index, n_hints])
    return add_hints(question, answer, n_hints, rng)

def puzzle_with_empties(question: str, answer: str, n_empty: int, seed: int, index: int) -> str:
    """Reveal cells until at most `n_empty` blanks remain (the curriculum's difficulty knob).
    A puzzle with fewer blanks than that (the split goes down to 46) is left untouched."""
    return hinted_puzzle(question, answer, max(0, question.count("0") - n_empty), seed, index)

def load_puzzles(data_dir: str, split: str) -> list[tuple[str, str]]:
    return read_split(data_dir, split)

def eval_rows(data_dir: str, n: int, layout: str = "rows") -> list[dict]:
    """The first `n` raw `test_hard` puzzles -- the fixed subset the reasoning baseline reports on."""
    rows = load_puzzles(data_dir, "test_hard")[:n]
    return [{"prompt": make_prompt(q, layout), "question": q, "answer": a, "empties": q.count("0"), "index": i}
            for i, (q, a) in enumerate(rows)]

def train_rows(data_dir: str, seed: int, indices: range, n_empty: int, layout: str = "rows") -> list[dict]:
    """Augmented training puzzles cut down to `n_empty` blanks (for the diagnostic sweep)."""
    rows = load_puzzles(data_dir, "train")
    out = []
    for idx, q, a in augmented_rows(rows, indices, seed):
        hq = puzzle_with_empties(q, a, n_empty, seed, idx)
        out.append({"prompt": make_prompt(hq, layout), "question": hq, "answer": a,
                    "empties": hq.count("0"), "index": idx})
    return out

def grade(text: str, question: str, answer: str) -> dict:
    """Score one completion. `closed`: the model ended its reasoning by itself (the chat template
    opens `<think>`, so an answer only exists after `</think>`)."""
    closed = THINK_END in text
    grid = parse_grid(text, reasoning=True)
    gold = np.array([int(c) for c in answer])
    given = np.array([int(c) for c in question])
    if grid is None:
        return {"closed": closed, "parsed": False, "exact": False, "legal": False,
                "clues_kept": 0.0, "cell_acc": 0.0}
    clue = given > 0
    return {
        "closed": closed,
        "parsed": True,
        "exact": bool((grid == gold).all()),
        "legal": bool(is_legal_solution(grid[None])[0]),
        "clues_kept": float((grid[clue] == given[clue]).mean()),
        "cell_acc": float((grid == gold).mean()),
    }

def summarize(grades: list[dict], n_tokens: list[int] | None = None) -> dict:
    """Aggregate `grade` outputs the way `llm_reason.rollout.merge` reports them."""
    n = len(grades)
    closed = np.array([g["closed"] for g in grades])
    exact = np.array([g["exact"] for g in grades])
    m = {
        "n": n,
        "exact_match": float(exact.mean()),
        "legal_grid": float(np.mean([g["legal"] for g in grades])),
        "parse_rate": float(np.mean([g["parsed"] for g in grades])),
        "closed_rate": float(closed.mean()),
        "cell_accuracy": float(np.mean([g["cell_acc"] for g in grades])),
        "exact_match_when_closed": float(exact[closed].mean()) if closed.any() else 0.0,
    }
    if n_tokens is not None:
        t = np.asarray(n_tokens)
        m.update(tokens_mean=float(t.mean()), tokens_p50=float(np.median(t)), tokens_p90=float(np.percentile(t, 90)))
    return m
