"""Build a deterministic Sudoku holdout that is never used for model selection.

Selection is order-independent: for each source, retain the ``per_source``
eligible questions with the smallest SHA-256 priorities.  All questions in
the supplied exclusion CSVs are forbidden.  The resulting test file is meant
to be generated once, hashed, and kept sealed until the analysis is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import shutil
from collections import Counter
from pathlib import Path


FIELDNAMES = ("source", "question", "answer", "rating")
DEFAULT_SOURCES = ("puzzles2_17_clue", "puzzles4_forum_hardest_1905")


def csv_rows(paths: list[Path]):
    for path in paths:
        with path.open(newline="") as handle:
            yield from csv.DictReader(handle)


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
        if set(answer[index * 9 : (index + 1) * 9]) != expected:
            raise ValueError("Invalid answer row")
        if set(answer[index::9]) != expected:
            raise ValueError("Invalid answer column")


def priority(seed: int, source: str, question: str) -> int:
    payload = f"{seed}\0{source}\0{question}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-template", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--per-source", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    excluded = {row["question"].replace("0", ".") for row in csv_rows(args.exclude)}
    heaps: dict[str, list[tuple[int, str, dict[str, str]]]] = {
        source: [] for source in args.sources
    }
    selected_questions: dict[str, set[str]] = {source: set() for source in args.sources}
    populations = Counter()
    excluded_counts = Counter()

    for row in csv_rows(args.input):
        source = row["source"]
        if source not in heaps:
            continue
        question = row["question"].replace("0", ".")
        populations[source] += 1
        if question in excluded:
            excluded_counts[source] += 1
            continue
        score = priority(args.seed, source, question)
        heap = heaps[source]
        chosen = selected_questions[source]
        if question in chosen:
            continue
        item = (-score, question, row)
        if len(heap) < args.per_source:
            heapq.heappush(heap, item)
            chosen.add(question)
        elif score < -heap[0][0]:
            removed = heapq.heapreplace(heap, item)
            chosen.remove(removed[1])
            chosen.add(question)

    selected: list[dict[str, str]] = []
    source_counts = {}
    for source in args.sources:
        rows = [item[2] for item in heaps[source]]
        if len(rows) != args.per_source:
            raise RuntimeError(f"Only found {len(rows)} eligible rows for {source}")
        rows.sort(key=lambda row: priority(args.seed, source, row["question"]))
        for row in rows:
            validate(row)
        selected.extend(rows)
        source_counts[source] = {
            "population_rows": populations[source],
            "excluded_rows": excluded_counts[source],
            "selected": len(rows),
        }

    # Deterministic interleaving prevents source-contiguous evaluation batches.
    selected.sort(key=lambda row: priority(args.seed + 1, row["source"], row["question"]))
    if len({row["question"] for row in selected}) != len(selected):
        raise RuntimeError("Selected questions are not globally unique")
    if any(row["question"].replace("0", ".") in excluded for row in selected):
        raise RuntimeError("Exclusion leakage detected")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    test_path = args.output_dir / "test_hard.csv"
    with test_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDNAMES)
        writer.writeheader()
        writer.writerows(selected)
    shutil.copyfile(args.train_template, args.output_dir / "train.csv")

    metadata = {
        "status": "SEALED_FINAL_TEST_DO_NOT_USE_FOR_SELECTION",
        "selection": "smallest SHA-256 priorities per source",
        "seed": args.seed,
        "inputs": [str(path) for path in args.input],
        "exclusions": [str(path) for path in args.exclude],
        "sources": source_counts,
        "total": len(selected),
        "test_hard_sha256": sha256_file(test_path),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
