"""Command-line interface for the standalone JTM estimator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from .compute import compute_jtm, sha256_file
from .data import (
    board_ids_sha256,
    build_loader,
    dataset_sha256,
    read_test_records,
    resolve_test_csv,
    select_board_ids,
)
from .models import build_model


OUTPUT_FILENAMES = (
    "accuracy_by_step.csv",
    "jtm_long.csv",
    "manifest.json",
    "per_board_jtm.csv",
    "per_probe_jtm.csv",
    "selected_board_ids.csv",
    *(f"jtm_{state}{suffix}.csv"
      for state in ("H", "L", "Z")
      for suffix in ("", "_rms")),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute an output-Jacobian triangular matrix from a standard "
            "train.py HRM or recurrent-Transformer checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Defaults to model_config.json next to the checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Sudoku Extreme directory, test_hard.csv, or test.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boards", type=int, default=64)
    parser.add_argument("--board-offset", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=("random", "sequential", "all"),
        default="random",
        help="Random is the representative default; sequential is opt-in.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260827)
    parser.add_argument("--board-ids", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument(
        "--outer-steps",
        type=int,
        help="Defaults to cycles_per_data in model_config.json.",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--random-seed", type=int, default=20260828)
    parser.add_argument("--save-per-probe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else checkpoint.parent / "model_config.json"
    )
    if not config_path.is_file():
        parser.error(f"Config does not exist: {config_path}")
    if args.boards <= 0 or args.batch_size <= 0 or args.probes <= 0:
        parser.error("--boards, --batch-size and --probes must be positive")
    if args.board_offset < 0:
        parser.error("--board-offset must be non-negative")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(f"Output directory is non-empty: {output_dir}")
    if args.overwrite:
        for filename in OUTPUT_FILENAMES:
            path = output_dir / filename
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA device requested but CUDA is unavailable")
    model, config = build_model(checkpoint, config_path, device)
    configured_steps = config.get("cycles_per_data")
    outer_steps = args.outer_steps if args.outer_steps is not None else configured_steps
    if not isinstance(outer_steps, int) or outer_steps <= 0:
        parser.error(
            "A positive --outer-steps is required when cycles_per_data is absent"
        )

    csv_path = resolve_test_csv(args.dataset)
    records = read_test_records(csv_path)
    try:
        board_ids = select_board_ids(
            len(records),
            boards=args.boards,
            offset=args.board_offset,
            selection=args.selection,
            selection_seed=args.selection_seed,
            explicit=args.board_ids,
        )
    except ValueError as error:
        parser.error(str(error))
    loader = build_loader(records, board_ids, args.batch_size)
    architecture = str(config["arch"].get("name"))
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "architecture": architecture,
        "dataset_csv": str(csv_path),
        "dataset_records": len(records),
        "dataset_content_sha256": dataset_sha256(records),
        "board_ids": list(board_ids),
        "board_ids_sha256": board_ids_sha256(board_ids),
        "selection": {
            "mode": "explicit" if args.board_ids is not None else args.selection,
            "seed": args.selection_seed,
            "offset": args.board_offset,
        },
    }
    manifest = compute_jtm(
        model,
        loader,
        device=device,
        output_dir=output_dir,
        outer_steps=outer_steps,
        probes=args.probes,
        random_seed=args.random_seed,
        save_per_probe=args.save_per_probe,
        manifest_metadata=metadata,
    )
    if not manifest["checks"]["all_values_finite"]:
        raise RuntimeError("JTM contains NaN or infinite values")
    if not manifest["checks"]["probe_row_count_matches"]:
        raise RuntimeError("JTM output row count failed its integrity check")
    print(f"JTM complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
