"""Build controlled Sudoku OOD datasets from sudoku-extreme.

The two training sets contain puzzles1_unbiased examples only.  They share an
identical hard test set containing every puzzles5/puzzles6 example and a
uniform random sample of puzzles4 examples to reach one million rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path


TRAIN_SOURCE = "puzzles1_unbiased"
HARD_SAMPLE_SOURCE = "puzzles4_forum_hardest_1905"
HARD_ALL_SOURCES = {
    "puzzles5_forum_hardest_1905_11+",
    "puzzles6_forum_hardest_1106",
}
FIELDNAMES = ("source", "question", "answer", "rating")


def rows(paths: list[Path]):
    for path in paths:
        with path.open(newline="") as handle:
            yield from csv.DictReader(handle)


def select_uniformly(
    rng: random.Random,
    remaining_population: int,
    remaining_needed: int,
) -> bool:
    """Online uniform sampling when population size is known."""
    if remaining_needed == 0:
        return False
    if rng.random() < remaining_needed / remaining_population:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-test-size", type=int, default=1_000_000)
    args = parser.parse_args()

    inputs = [args.input / "train.csv", args.input / "test.csv"]
    counts = Counter(row["source"] for row in rows(inputs))

    train_total = counts[TRAIN_SOURCE]
    hard_all_total = sum(counts[source] for source in HARD_ALL_SOURCES)
    hard_sample_needed = args.hard_test_size - hard_all_total
    if train_total < 1_000_000:
        raise ValueError(f"Need 1,000,000 {TRAIN_SOURCE} rows, found {train_total}")
    if not 0 <= hard_sample_needed <= counts[HARD_SAMPLE_SOURCE]:
        raise ValueError("Requested hard-test size cannot be constructed")

    out_1k = args.output_root / "sudoku-ood-puzzles1-1k"
    out_1m = args.output_root / "sudoku-ood-puzzles1-1m"
    out_1k.mkdir(parents=True, exist_ok=True)
    out_1m.mkdir(parents=True, exist_ok=True)

    rng_train = random.Random(args.seed)
    rng_test = random.Random(args.seed + 1)
    train_remaining = train_total
    train_1k_needed = 1_000
    hard_sample_remaining = counts[HARD_SAMPLE_SOURCE]
    selected_hard = 0

    with (
        (out_1m / "train.csv").open("w", newline="") as train_1m_handle,
        (out_1k / "train.csv").open("w", newline="") as train_1k_handle,
        (out_1m / "test_hard.csv").open("w", newline="") as test_handle,
    ):
        train_1m_writer = csv.DictWriter(train_1m_handle, FIELDNAMES)
        train_1k_writer = csv.DictWriter(train_1k_handle, FIELDNAMES)
        test_writer = csv.DictWriter(test_handle, FIELDNAMES)
        for writer in (train_1m_writer, train_1k_writer, test_writer):
            writer.writeheader()

        for row in rows(inputs):
            source = row["source"]
            if source == TRAIN_SOURCE:
                train_1m_writer.writerow(row)
                if select_uniformly(rng_train, train_remaining, train_1k_needed):
                    train_1k_writer.writerow(row)
                    train_1k_needed -= 1
                train_remaining -= 1
            elif source in HARD_ALL_SOURCES:
                test_writer.writerow(row)
            elif source == HARD_SAMPLE_SOURCE:
                needed = hard_sample_needed - selected_hard
                if select_uniformly(rng_test, hard_sample_remaining, needed):
                    test_writer.writerow(row)
                    selected_hard += 1
                hard_sample_remaining -= 1

    if train_1k_needed or selected_hard != hard_sample_needed:
        raise RuntimeError("Online sampling ended with an incorrect row count")

    # A hard link guarantees byte-identical test sets without using extra disk.
    shared_test = out_1m / "test_hard.csv"
    second_test = out_1k / "test_hard.csv"
    if second_test.exists():
        second_test.unlink()
    os.link(shared_test, second_test)

    metadata = {
        "seed": args.seed,
        "inputs": [str(path) for path in inputs],
        "training_source": TRAIN_SOURCE,
        "train_sizes": {"sudoku-ood-puzzles1-1k": 1_000, "sudoku-ood-puzzles1-1m": train_total},
        "hard_test_size": args.hard_test_size,
        "hard_test_sources": {
            HARD_SAMPLE_SOURCE: hard_sample_needed,
            **{source: counts[source] for source in sorted(HARD_ALL_SOURCES)},
        },
    }
    for output in (out_1k, out_1m):
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
