"""Build a deterministic, source-balanced 10k Sudoku OOD evaluation set.

Sources puzzles2 through puzzles6 each contribute the same number of rows.
Sources with too few distinct rows are expanded with solution-preserving Sudoku
symmetries rather than duplicated verbatim.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


SOURCES = (
    "puzzles2_17_clue",
    "puzzles3_magictour_top1465",
    "puzzles4_forum_hardest_1905",
    "puzzles5_forum_hardest_1905_11+",
    "puzzles6_forum_hardest_1106",
)
FIELDNAMES = ("source", "question", "answer", "rating")


def transform_grid(value: str, rng: random.Random) -> str:
    grid = [list(value[row * 9 : (row + 1) * 9]) for row in range(9)]
    digits = list("123456789")
    rng.shuffle(digits)
    digit_map = dict(zip("123456789", digits, strict=True))
    digit_map["."] = "."
    digit_map["0"] = "0"

    bands = list(range(3))
    stacks = list(range(3))
    rng.shuffle(bands)
    rng.shuffle(stacks)
    row_order = []
    col_order = []
    for band in bands:
        within = list(range(3))
        rng.shuffle(within)
        row_order.extend(band * 3 + offset for offset in within)
    for stack in stacks:
        within = list(range(3))
        rng.shuffle(within)
        col_order.extend(stack * 3 + offset for offset in within)

    transformed = [[grid[row][col] for col in col_order] for row in row_order]
    if rng.random() < 0.5:
        transformed = [list(row) for row in zip(*transformed, strict=True)]
    return "".join(digit_map[cell] for row in transformed for cell in row)


def augment_row(row: dict[str, str], rng: random.Random) -> dict[str, str]:
    state = rng.getstate()
    question = transform_grid(row["question"], rng)
    rng.setstate(state)
    answer = transform_grid(row["answer"], rng)
    return {**row, "question": question, "answer": answer}


def validate(row: dict[str, str]) -> None:
    question = row["question"].replace("0", ".")
    answer = row["answer"]
    if len(question) != 81 or len(answer) != 81:
        raise ValueError("Sudoku strings must contain 81 cells")
    for index, clue in enumerate(question):
        if clue != "." and clue != answer[index]:
            raise ValueError("Question clue does not match its answer")
    expected = set("123456789")
    for index in range(9):
        row_values = set(answer[index * 9 : (index + 1) * 9])
        col_values = set(answer[index::9])
        if row_values != expected or col_values != expected:
            raise ValueError("Invalid completed Sudoku grid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-template", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=4300)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.input.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source"] in SOURCES:
                by_source[row["source"]].append(row)

    selected = []
    metadata_sources = {}
    for source in SOURCES:
        population = by_source[source]
        if not population:
            raise ValueError(f"No input rows found for {source}")
        if len(population) >= args.per_source:
            source_rows = rng.sample(population, args.per_source)
            augmented = 0
        else:
            source_rows = list(population)
            seen = {(row["question"], row["answer"]) for row in source_rows}
            while len(source_rows) < args.per_source:
                candidate = augment_row(rng.choice(population), rng)
                key = (candidate["question"], candidate["answer"])
                if key not in seen:
                    seen.add(key)
                    source_rows.append(candidate)
            augmented = args.per_source - len(population)
        for row in source_rows:
            validate(row)
        selected.extend(source_rows)
        metadata_sources[source] = {
            "available_distinct": len(population),
            "selected_original": min(len(population), args.per_source),
            "augmented": augmented,
            "final": len(source_rows),
        }

    rng.shuffle(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "test_hard.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDNAMES)
        writer.writeheader()
        writer.writerows(selected)
    shutil.copyfile(args.train_template, args.output_dir / "train.csv")
    metadata = {
        "seed": args.seed,
        "input": str(args.input),
        "per_source": args.per_source,
        "total": len(selected),
        "augmentation": "digit, band/row, stack/column, and transpose symmetries",
        "sources": metadata_sources,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
