#!/usr/bin/env python3
"""Estimate the 16x16 recurrent-state to future-readout Jacobian triangle for RT."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "00_shared_runtime"
if str(RUNTIME_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(RUNTIME_ROOT))

from hrm.analyse.dynamics.output_sensitivity.observational import (
    RunSpec,
    _load_config,
    _sha256_file,
    _sha256_mapping,
    build_fixed_dataset,
    load_board_indices,
    load_model,
)
from hrm.analyse.dynamics.output_sensitivity.h_l_state_rank_audit import atomic_csv, atomic_json

SCHEMA_VERSION = "rt-output-triangle-vjp-v1"
OUTER_STEPS = 16
OUTPUT_DIM = 81 * 10


def load_rt_run_spec(seed_dir: Path, epoch: int) -> RunSpec:
    """Resolve one RT run without widening the shared HRM-only loader."""
    seed_dir = seed_dir.expanduser().resolve()
    match = re.fullmatch(r"seed_(\d+)", seed_dir.name)
    if not seed_dir.is_dir() or match is None:
        raise ValueError(f"Expected existing seed_N directory, got {seed_dir}")
    seed = int(match.group(1))
    checkpoint = seed_dir / f"epoch_{epoch}.pt"
    from hrm.paths import checkpoint_config
    config_path = checkpoint_config(checkpoint)
    if not checkpoint.is_file() or not config_path.is_file():
        raise ValueError("Missing checkpoint or model_config.json")
    config, config_file_sha256 = _load_config(config_path)
    if config.get("arch", {}).get("name") != "rt@RecurrentTransformer":
        raise ValueError("Run is not rt@RecurrentTransformer")
    if seed not in config.get("seeds", []):
        raise ValueError(f"Seed {seed} is absent from configured seeds")
    epochs = config.get("epochs")
    if not isinstance(epochs, int) or epoch < 0 or epoch >= epochs:
        raise ValueError(f"Epoch {epoch} is outside configured epochs={epochs}")
    cycles = config.get("cycles_per_data")
    core_cycles = config["arch"].get("cycles")
    if not isinstance(cycles, int) or cycles <= 0:
        raise ValueError("cycles_per_data must be positive")
    if not isinstance(core_cycles, int) or core_cycles <= 0:
        raise ValueError("arch.cycles must be positive")
    checkpoint_stat = checkpoint.stat()
    return RunSpec(
        seed_dir=seed_dir,
        seed=seed,
        epoch=epoch,
        checkpoint=checkpoint,
        checkpoint_sha256=_sha256_file(checkpoint),
        checkpoint_bytes=checkpoint_stat.st_size,
        checkpoint_mtime_ns=checkpoint_stat.st_mtime_ns,
        config_path=config_path,
        config=config,
        config_sha256=_sha256_mapping(config),
        config_file_sha256=config_file_sha256,
        cycles=cycles,
    )


def rademacher_output(
    shape: torch.Size, device: torch.device, generator: torch.Generator
) -> Tensor:
    value = torch.randint(
        0, 2, shape, device=device, generator=generator, dtype=torch.int8
    ).float().mul_(2).sub_(1)
    return value - value.mean(dim=-1, keepdim=True)


def outer_transition(model: nn.Module, latent: Tensor, z: Tensor) -> Tensor:
    for _ in range(int(model.cycles)):  # type: ignore[attr-defined]
        z = model.core(z + latent)  # type: ignore[attr-defined]
    return z


def boundary_linearizations(model: nn.Module, x: Tensor):
    latent = model.embed(x).detach()  # type: ignore[attr-defined]
    z = torch.broadcast_to(model.initial_carry["z"], latent.shape).clone()  # type: ignore[attr-defined]
    with torch.no_grad():
        z = outer_transition(model, latent, z)
    states, vjps = [z.detach()], []
    for _ in range(OUTER_STEPS - 1):
        def mapping(current: Tensor) -> Tensor:
            return outer_transition(model, latent, current)

        next_z, vjp_fn = torch.func.vjp(mapping, states[-1])
        vjps.append(vjp_fn)
        states.append(next_z.detach())
    logits = [model.lm_head(state)[:, 1:].float() for state in states]  # type: ignore[attr-defined]
    return states, logits, vjps


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["board_id"]), int(row["source_step"]), int(row["readout_step"]))
        grouped.setdefault(key, []).append(row)
    result = []
    for key in sorted(grouped, key=lambda value: (value[0], value[2], value[1])):
        selected = grouped[key]
        values = np.asarray([float(value["gain_sq"]) for value in selected])
        mean_sq = float(values.mean())
        se_sq = (
            float(values.std(ddof=1) / math.sqrt(values.size))
            if values.size > 1 else 0.0
        )
        source = selected[0]
        result.append({
            "board_id": key[0],
            "source_step": key[1],
            "readout_step": key[2],
            "horizon": key[2] - key[1],
            "blank_count": int(source["blank_count"]),
            "final_exact": int(source["final_exact"]),
            "readout_exact": int(source["readout_exact"]),
            "probes": int(values.size),
            "mean_gain_sq": mean_sq,
            "mean_gain_sq_se": se_sq,
            "rms_gain": math.sqrt(max(mean_sq, 0.0)),
            "rms_gain_se_delta": 0.5 * se_sq / max(math.sqrt(max(mean_sq, 0.0)), 1e-30),
        })
    return result


def run_batch(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    board_ids: Sequence[int],
    probes: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    states, logits, transition_vjps = boundary_linearizations(model, x)
    target = y[:, 1:].long()
    exact = [value.argmax(-1).eq(target).all(-1) for value in logits]
    final_exact = exact[-1]
    blanks = x[:, 1:].eq(0).sum(-1)
    weight = model.lm_head.weight.to(torch.float32)  # type: ignore[attr-defined]
    rows: list[dict[str, Any]] = []
    diagonal_min, diagonal_max = math.inf, 0.0

    def append(source: int, endpoint: int, probe_index: int, gradient: Tensor) -> None:
        nonlocal diagonal_min, diagonal_max
        value = gradient[:, 1:].float()
        gain_sq = value.square().sum((1, 2)) / OUTPUT_DIM
        if source == endpoint:
            diagonal_min = min(diagonal_min, float(gain_sq.min()))
            diagonal_max = max(diagonal_max, float(gain_sq.max()))
        for batch_index, board_id in enumerate(board_ids):
            rows.append({
                "board_id": int(board_id),
                "source_step": source + 1,
                "readout_step": endpoint + 1,
                "horizon": endpoint - source,
                "probe": probe_index,
                "blank_count": int(blanks[batch_index]),
                "final_exact": int(final_exact[batch_index]),
                "readout_exact": int(exact[endpoint][batch_index]),
                "gain_sq": float(gain_sq[batch_index]),
            })

    for endpoint in range(OUTER_STEPS):
        for probe_index in range(probes):
            probe_parts = []
            for board_id in board_ids:
                generator = torch.Generator(device=x.device)
                generator.manual_seed(
                    random_seed + int(board_id) * 1_000_003
                    + endpoint * 10_007 + probe_index
                )
                probe_parts.append(
                    rademacher_output(logits[endpoint][:1].shape, x.device, generator)
                )
            probe = torch.cat(probe_parts, dim=0)
            cotangent = torch.zeros_like(states[endpoint], dtype=torch.float32)
            cotangent[:, 1:] = probe @ weight
            append(endpoint, endpoint, probe_index, cotangent)
            for source in range(endpoint - 1, -1, -1):
                cotangent = transition_vjps[source](cotangent)[0]
                append(source, endpoint, probe_index, cotangent)
    return rows, {"minimum_diagonal_gain_sq": diagonal_min,
                  "maximum_diagonal_gain_sq": diagonal_max}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boards", type=int, default=1)
    parser.add_argument("--board-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--explicit-board-ids", type=int, nargs="*")
    parser.add_argument(
        "--board-indices", type=Path,
        default=RUNTIME_ROOT / "experiments/hrm_mechanism_splits.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-seed", type=int, default=20_260_828)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if min(args.boards, args.batch_size, args.probes) <= 0 or args.board_offset < 0:
        parser.error("boards, batch-size, and probes must be positive")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error(f"Output is non-empty: {output}")
    device = torch.device(args.device)
    spec = load_rt_run_spec(args.seed_dir, args.epoch)
    if spec.config["arch"].get("name") != "rt@RecurrentTransformer":
        parser.error("Checkpoint is not rt@RecurrentTransformer")

    if args.explicit_board_ids is not None:
        board_ids = tuple(args.explicit_board_ids)
        if (not board_ids or len(board_ids) != len(set(board_ids))
                or min(board_ids) < 0 or max(board_ids) >= 20_000):
            parser.error("explicit-board-ids must be a unique subset of [0, 20000)")
        split_hash = "explicit-test-hard-board-ids"
        content_hash = "test-hard-20k-from-dataset"
    else:
        if args.full_test:
            population = tuple(range(20_000))
            split_hash = hashlib.sha256(
                np.asarray(population, dtype=np.int64).tobytes()
            ).hexdigest()
            content_hash = "test-hard-20k-from-dataset"
        else:
            population, split_hash, content_hash = load_board_indices(
                args.board_indices, max_boards=2048, group="test"
            )
        board_ids = population[args.board_offset:args.board_offset + args.boards]
        if len(board_ids) != args.boards:
            parser.error("Requested board range exceeds selected population")

    selected_hash = hashlib.sha256(
        np.asarray(board_ids, dtype=np.int64).tobytes()
    ).hexdigest()
    dataset = build_fixed_dataset(
        spec, board_ids, batch_size=args.batch_size,
        expected_dataset_size=20_000, device=device,
    )
    model = load_model(spec, dataset.metadata, device).float()
    model.requires_grad_(False)
    model.embed.cast_to = torch.float32  # type: ignore[attr-defined]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    checks: list[dict[str, float]] = []
    consumed = 0
    probe_rows = 0
    started = time.time()
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        for batch in dataset.loader:
            x, y = batch[0].to(device), batch[1].to(device)
            ids = board_ids[consumed:consumed + x.shape[0]]
            batch_rows, batch_checks = run_batch(
                model, x, y, ids, args.probes, args.random_seed
            )
            probe_rows += len(batch_rows)
            if args.summary_only:
                summaries.extend(summarize(batch_rows))
            else:
                rows.extend(batch_rows)
            checks.append(batch_checks)
            consumed += x.shape[0]
            print(json.dumps({"boards_done": consumed, "boards_total": len(board_ids),
                              "elapsed_sec": round(time.time() - started, 3)}), flush=True)

    if consumed != len(board_ids):
        raise RuntimeError(f"Consumed {consumed}, expected {len(board_ids)}")
    if not args.summary_only:
        atomic_csv(output / "per_probe.csv", rows)
    atomic_csv(output / "per_board_triangle.csv", summaries if args.summary_only else summarize(rows))
    expected_rows = len(board_ids) * OUTER_STEPS * (OUTER_STEPS + 1) // 2 * args.probes
    atomic_json(output / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "checkpoint": str(spec.checkpoint),
        "checkpoint_sha256": spec.checkpoint_sha256,
        "seed": spec.seed,
        "epoch": args.epoch,
        "board_ids": list(board_ids),
        "board_ids_sha256": selected_hash,
        "board_split_hash": split_hash,
        "board_content_hash": content_hash,
        "elapsed_seconds": time.time() - started,
        "checks": {
            "minimum_diagonal_gain_sq": min(value["minimum_diagonal_gain_sq"] for value in checks),
            "maximum_diagonal_gain_sq": max(value["maximum_diagonal_gain_sq"] for value in checks),
            "rows_expected": expected_rows,
            "rows_actual": probe_rows,
        },
        "protocol": {
            "outer_steps": OUTER_STEPS,
            "core_cycles_per_outer": int(model.cycles),  # type: ignore[attr-defined]
            "source_nodes": "independent recurrent boundary state z_i after each outer step; y_i=head(z_i)",
            "endpoint_reuse": "one direct readout VJP followed by reverse recurrence returns every earlier source",
            "output": "cellwise logits with per-cell class mean removed; BOS excluded",
            "gain_sq": "Hutchinson VJP estimate of ||d centered_logits_j / d z_i||_F^2 / (81*10)",
            "per_probe_saved": not args.summary_only,
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
