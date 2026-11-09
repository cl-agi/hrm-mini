"""Matrix-free VJP estimator for output-Jacobian triangular matrices."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


SCHEMA_VERSION = "standalone-jtm-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class CsvStream:
    """Atomically stream same-schema dictionaries without retaining every row."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.handle: Any = None
        self.writer: csv.DictWriter | None = None
        self.rows = 0

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.temporary.open("w", newline="")
            self.writer = csv.DictWriter(
                self.handle, fieldnames=list(rows[0].keys())
            )
            self.writer.writeheader()
        self.writer.writerows(rows)
        self.rows += len(rows)

    def close(self) -> None:
        if self.writer is None:
            raise ValueError(f"Refusing to write empty CSV: {self.path}")
        self.handle.close()
        self.temporary.replace(self.path)


def rademacher_output(
    shape: torch.Size, device: torch.device, generator: torch.Generator
) -> Tensor:
    value = torch.randint(
        0, 2, shape, device=device, generator=generator, dtype=torch.int8
    ).float()
    value = value.mul_(2).sub_(1)
    return value - value.mean(dim=-1, keepdim=True)


def hrm_outer_transition(
    model: nn.Module, embedded: Tensor, h_state: Tensor, l_state: Tensor
) -> tuple[Tensor, Tensor]:
    for _ in range(int(model.H_cycles)):  # type: ignore[attr-defined]
        old_h = h_state
        for _ in range(int(model.L_cycles)):  # type: ignore[attr-defined]
            l_state = model.L_level(l_state + h_state + embedded)  # type: ignore[attr-defined]
        h_state = model.H_level(old_h + l_state)  # type: ignore[attr-defined]
    return h_state, l_state


def rt_outer_transition(model: nn.Module, embedded: Tensor, state: Tensor) -> Tensor:
    for _ in range(int(model.cycles)):  # type: ignore[attr-defined]
        state = model.core(state + embedded)  # type: ignore[attr-defined]
    return state


def boundary_linearizations(
    model: nn.Module, inputs: Tensor, outer_steps: int
) -> tuple[
    dict[str, list[Tensor]],
    list[Tensor],
    list[Callable[..., Any]],
]:
    embedded = model.embed(inputs).detach()  # type: ignore[attr-defined]
    family = str(model.family)  # type: ignore[attr-defined]
    if family == "hrm":
        h_state = torch.broadcast_to(model.initial_carry["z_H"], embedded.shape).clone()  # type: ignore[attr-defined]
        l_state = torch.broadcast_to(model.initial_carry["z_L"], embedded.shape).clone()  # type: ignore[attr-defined]
        with torch.no_grad():
            h_state, l_state = hrm_outer_transition(model, embedded, h_state, l_state)
        states: dict[str, list[Tensor]] = {
            "H": [h_state.detach()],
            "L": [l_state.detach()],
        }
        vjps: list[Callable[..., Any]] = []
        for _ in range(outer_steps - 1):
            def mapping(h_value: Tensor, l_value: Tensor) -> tuple[Tensor, Tensor]:
                return hrm_outer_transition(model, embedded, h_value, l_value)

            (next_h, next_l), vjp_function = torch.func.vjp(
                mapping, states["H"][-1], states["L"][-1]
            )
            vjps.append(vjp_function)
            states["H"].append(next_h.detach())
            states["L"].append(next_l.detach())
        logits = [model.lm_head(value)[:, 1:].float() for value in states["H"]]  # type: ignore[attr-defined]
        return states, logits, vjps

    if family == "rt":
        state = torch.broadcast_to(model.initial_carry["z"], embedded.shape).clone()  # type: ignore[attr-defined]
        with torch.no_grad():
            state = rt_outer_transition(model, embedded, state)
        states = {"Z": [state.detach()]}
        vjps = []
        for _ in range(outer_steps - 1):
            def mapping(value: Tensor) -> Tensor:
                return rt_outer_transition(model, embedded, value)

            next_state, vjp_function = torch.func.vjp(mapping, states["Z"][-1])
            vjps.append(vjp_function)
            states["Z"].append(next_state.detach())
        logits = [model.lm_head(value)[:, 1:].float() for value in states["Z"]]  # type: ignore[attr-defined]
        return states, logits, vjps

    raise ValueError(f"Unknown model family: {family}")


def run_batch(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    board_ids: Sequence[int],
    *,
    outer_steps: int,
    probes: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states, logits, transition_vjps = boundary_linearizations(
        model, inputs, outer_steps
    )
    family = str(model.family)  # type: ignore[attr-defined]
    source_states = ("H", "L") if family == "hrm" else ("Z",)
    labels = targets[:, 1:].long()
    exact = [value.argmax(-1).eq(labels).all(-1) for value in logits]
    blanks = inputs[:, 1:].eq(0).sum(-1)
    final_exact = exact[-1]
    vocab_size = logits[0].shape[-1]
    output_dim = logits[0].shape[-2] * vocab_size
    weight = model.lm_head.weight.to(torch.float32)  # type: ignore[attr-defined]
    rows: list[dict[str, Any]] = []

    def append(
        source: int,
        endpoint: int,
        state_name: str,
        probe_index: int,
        gradient: Tensor,
    ) -> None:
        non_bos = gradient[:, 1:].float()
        gain_sq = non_bos.square().sum((1, 2)) / output_dim
        for batch_index, board_id in enumerate(board_ids):
            rows.append(
                {
                    "board_id": int(board_id),
                    "source_step": source + 1,
                    "readout_step": endpoint + 1,
                    "horizon": endpoint - source,
                    "source_state": state_name,
                    "probe": probe_index,
                    "blank_count": int(blanks[batch_index]),
                    "readout_exact": int(exact[endpoint][batch_index]),
                    "final_exact": int(final_exact[batch_index]),
                    "gain_sq": float(gain_sq[batch_index]),
                }
            )

    for endpoint in range(outer_steps):
        for probe_index in range(probes):
            probe_parts: list[Tensor] = []
            for board_id in board_ids:
                generator = torch.Generator(device=inputs.device)
                generator.manual_seed(
                    random_seed
                    + int(board_id) * 1_000_003
                    + endpoint * 10_007
                    + probe_index
                )
                probe_parts.append(
                    rademacher_output(
                        logits[endpoint][:1].shape, inputs.device, generator
                    )
                )
            probe = torch.cat(probe_parts, dim=0)
            cotangent = torch.zeros_like(states[source_states[0]][endpoint])
            cotangent[:, 1:] = probe @ weight
            if family == "hrm":
                cotangents: tuple[Tensor, ...] = (
                    cotangent,
                    torch.zeros_like(states["L"][endpoint]),
                )
                append(endpoint, endpoint, "H", probe_index, cotangents[0])
                append(endpoint, endpoint, "L", probe_index, cotangents[1])
                for source in range(endpoint - 1, -1, -1):
                    cotangents = transition_vjps[source](cotangents)
                    append(source, endpoint, "H", probe_index, cotangents[0])
                    append(source, endpoint, "L", probe_index, cotangents[1])
            else:
                append(endpoint, endpoint, "Z", probe_index, cotangent)
                for source in range(endpoint - 1, -1, -1):
                    cotangent = transition_vjps[source](cotangent)[0]
                    append(source, endpoint, "Z", probe_index, cotangent)

    accuracy_rows = [
        {
            "board_id": int(board_id),
            "readout_step": step + 1,
            "exact": int(exact[step][batch_index]),
        }
        for step in range(outer_steps)
        for batch_index, board_id in enumerate(board_ids)
    ]
    return rows, accuracy_rows


def summarize_per_board(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, str], list[float]] = {}
    metadata: dict[tuple[int, int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            int(row["board_id"]),
            int(row["source_step"]),
            int(row["readout_step"]),
            str(row["source_state"]),
        )
        grouped.setdefault(key, []).append(float(row["gain_sq"]))
        metadata[key] = row
    result: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: (value[3], value[0], value[2], value[1])):
        values = np.asarray(grouped[key], dtype=np.float64)
        row = metadata[key]
        result.append(
            {
                "board_id": key[0],
                "source_step": key[1],
                "readout_step": key[2],
                "horizon": key[2] - key[1],
                "source_state": key[3],
                "blank_count": int(row["blank_count"]),
                "readout_exact": int(row["readout_exact"]),
                "final_exact": int(row["final_exact"]),
                "probes": len(values),
                "mean_gain_sq": float(values.mean()),
                "rms_gain": math.sqrt(max(float(values.mean()), 0.0)),
            }
        )
    return result


def summarize_cohort(
    per_board: Sequence[Mapping[str, Any]], probes: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[float]] = {}
    for row in per_board:
        key = (
            int(row["source_step"]),
            int(row["readout_step"]),
            str(row["source_state"]),
        )
        grouped.setdefault(key, []).append(float(row["mean_gain_sq"]))
    return summarize_cohort_values(grouped, probes)


def summarize_cohort_values(
    grouped: Mapping[tuple[int, int, str], Sequence[float]], probes: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: (value[2], value[1], value[0])):
        values = np.asarray(grouped[key], dtype=np.float64)
        mean_sq = float(values.mean())
        se_sq = (
            float(values.std(ddof=1) / math.sqrt(values.size))
            if values.size > 1
            else 0.0
        )
        rms = math.sqrt(max(mean_sq, 0.0))
        board_gain = np.sqrt(np.maximum(values, 0.0))
        result.append(
            {
                "source_step": key[0],
                "readout_step": key[1],
                "horizon": key[1] - key[0],
                "source_state": key[2],
                "boards": len(values),
                "probes_per_board": probes,
                "mean_gain_sq": mean_sq,
                "mean_gain_sq_se_across_boards": se_sq,
                "rms_gain": rms,
                "rms_gain_se_delta": 0.5 * se_sq / max(rms, 1e-30),
                "median_board_gain": float(np.median(board_gain)),
                "q10_board_gain": float(np.quantile(board_gain, 0.10)),
                "q90_board_gain": float(np.quantile(board_gain, 0.90)),
                "q99_board_gain": float(np.quantile(board_gain, 0.99)),
                "max_board_gain": float(board_gain.max()),
                "mean_log10_board_gain": float(
                    np.log10(np.maximum(board_gain, 1e-30)).mean()
                ),
            }
        )
    return result


def matrix_rows(
    cohort_rows: Sequence[Mapping[str, Any]],
    state_name: str,
    outer_steps: int,
    metric: str,
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["source_step"]), int(row["readout_step"])): float(row[metric])
        for row in cohort_rows
        if str(row["source_state"]) == state_name
    }
    return [
        {
            "source_step": source,
            **{
                f"y_{endpoint}": (
                    lookup[(source, endpoint)] if endpoint >= source else ""
                )
                for endpoint in range(1, outer_steps + 1)
            },
        }
        for source in range(1, outer_steps + 1)
    ]


def accuracy_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[int]] = {}
    for row in rows:
        grouped.setdefault(int(row["readout_step"]), []).append(int(row["exact"]))
    return [
        {
            "readout_step": step,
            "correct": sum(values),
            "boards": len(values),
            "exact_accuracy": sum(values) / len(values),
        }
        for step, values in sorted(grouped.items())
    ]


def compute_jtm(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    output_dir: Path,
    outer_steps: int,
    probes: int,
    random_seed: int,
    save_per_probe: bool,
    manifest_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_board_stream = CsvStream(output_dir / "per_board_jtm.csv")
    probe_stream = (
        CsvStream(output_dir / "per_probe_jtm.csv") if save_per_probe else None
    )
    cohort_values: dict[tuple[int, int, str], list[float]] = {}
    accuracy_counts: dict[int, list[int]] = {}
    boards_done = 0
    probe_rows_count = 0
    per_board_rows_count = 0
    all_values_finite = True
    source_states = ("H", "L") if str(model.family) == "hrm" else ("Z",)  # type: ignore[attr-defined]
    diagonal_limits = {
        state_name: {"minimum": math.inf, "maximum": -math.inf}
        for state_name in source_states
    }
    started = time.time()
    context = (
        torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        )
        if device.type == "cuda"
        else nullcontext()
    )
    with context:
        for inputs, targets, batch_ids in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            board_ids = [int(value) for value in batch_ids]
            rows, accuracy = run_batch(
                model,
                inputs,
                targets,
                board_ids,
                outer_steps=outer_steps,
                probes=probes,
                random_seed=random_seed,
            )
            probe_rows_count += len(rows)
            all_values_finite = all_values_finite and all(
                math.isfinite(float(row["gain_sq"])) for row in rows
            )
            for row in rows:
                if int(row["source_step"]) == int(row["readout_step"]):
                    limits = diagonal_limits[str(row["source_state"])]
                    value = float(row["gain_sq"])
                    limits["minimum"] = min(limits["minimum"], value)
                    limits["maximum"] = max(limits["maximum"], value)
            if probe_stream is not None:
                probe_stream.write(rows)
            batch_per_board = summarize_per_board(rows)
            per_board_stream.write(batch_per_board)
            per_board_rows_count += len(batch_per_board)
            for row in batch_per_board:
                key = (
                    int(row["source_step"]),
                    int(row["readout_step"]),
                    str(row["source_state"]),
                )
                cohort_values.setdefault(key, []).append(float(row["mean_gain_sq"]))
            for row in accuracy:
                counts = accuracy_counts.setdefault(
                    int(row["readout_step"]), [0, 0]
                )
                counts[0] += int(row["exact"])
                counts[1] += 1
            boards_done += len(board_ids)
            print(
                json.dumps(
                    {
                        "boards_done": boards_done,
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                ),
                flush=True,
            )
    per_board_stream.close()
    if probe_stream is not None:
        probe_stream.close()
    cohort = summarize_cohort_values(cohort_values, probes)
    write_csv(output_dir / "jtm_long.csv", cohort)
    write_csv(
        output_dir / "accuracy_by_step.csv",
        [
            {
                "readout_step": step,
                "correct": counts[0],
                "boards": counts[1],
                "exact_accuracy": counts[0] / counts[1],
            }
            for step, counts in sorted(accuracy_counts.items())
        ],
    )
    board_ids = [int(value) for value in manifest_metadata["board_ids"]]
    write_csv(
        output_dir / "selected_board_ids.csv",
        [
            {"selection_order": order, "board_id": board_id}
            for order, board_id in enumerate(board_ids)
        ],
    )
    for state_name in source_states:
        write_csv(
            output_dir / f"jtm_{state_name}.csv",
            matrix_rows(cohort, state_name, outer_steps, "median_board_gain"),
        )
        write_csv(
            output_dir / f"jtm_{state_name}_rms.csv",
            matrix_rows(cohort, state_name, outer_steps, "rms_gain"),
        )
    expected_probe_rows = (
        boards_done
        * probes
        * outer_steps
        * (outer_steps + 1)
        // 2
        * len(source_states)
    )
    row_counts_match = expected_probe_rows == probe_rows_count
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        **dict(manifest_metadata),
        "elapsed_seconds": time.time() - started,
        "checks": {
            "boards_processed": boards_done,
            "probe_rows_expected": expected_probe_rows,
            "probe_rows_actual": probe_rows_count,
            "probe_row_count_matches": row_counts_match,
            "per_board_rows_actual": per_board_rows_count,
            "cohort_rows_actual": len(cohort),
            "diagonal_gain_sq": diagonal_limits,
            "all_values_finite": all_values_finite,
        },
        "protocol": {
            "outer_steps": outer_steps,
            "source_states": list(source_states),
            "source_boundary": "state after each complete outer step; no step-0 row",
            "readout": "lm_head at each source boundary",
            "output": "BOS excluded; logits centered over classes by the probe covariance",
            "estimator": "Rademacher VJP Hutchinson estimator",
            "gain_sq": "||d centered_non_BOS_logits_j / d state_i||_F^2 divided by output dimension",
            "rms_gain": "sqrt(mean gain_sq across probes and boards)",
            "jtm_default_matrix": "median per-board rms_gain, matching the historical JTM report; *_rms.csv contains sqrt(mean gain_sq across boards)",
            "endpoint_reuse": "one output VJP per endpoint/probe is propagated through cached transition VJPs to fill the entire causal column",
            "analysis_precision": "model lifted to float32; math SDPA on CUDA",
            "per_probe_saved": save_per_probe,
            "memory_behavior": "per-board and optional per-probe CSVs are streamed by batch; only one scalar per board/JTM-cell is retained for cohort quantiles",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
